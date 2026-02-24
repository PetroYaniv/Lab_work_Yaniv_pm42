import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


TRAIN_CSV = "emnist-letters-train.csv"
TEST_CSV = "emnist-letters-test.csv"
MAPPING_TXT = "emnist-letters-mapping.txt"
MODEL_PATH = "emnist_paint_best.pth"


mapping = {}
with open(MAPPING_TXT,"r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts)>=2:
            idx = int(parts[0])
            ascii_code = int(parts[1])
            mapping[idx] = chr(ascii_code)
classes = [mapping[i] for i in sorted(mapping.keys())]


class EMNISTDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        df = pd.read_csv(csv_file, header=None)
        self.labels = df.iloc[:,0].values - 1   # EMNIST Letters: 1=A
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.images = self.images.reshape(-1,28,28)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Перетворення як у EMNIST: повернути та віддзеркалити
        img = np.rot90(img,1)
        img = np.fliplr(img)
        
        
        if self.augment:
            # Поворот
            angle = random.uniform(-20,20)
            M = cv2.getRotationMatrix2D((14,14),angle,1)
            img = cv2.warpAffine(img,M,(28,28),borderValue=0)

            # Зсув
            tx,ty = random.randint(-3,3), random.randint(-3,3)
            M = np.float32([[1,0,tx],[0,1,ty]])
            img = cv2.warpAffine(img,M,(28,28),borderValue=0)

           
            if random.random() < 0.5:
                kernel = np.ones((2,2),np.uint8)
                img = cv2.dilate(img,kernel,iterations=1)

            
            if random.random() < 0.3:
                kernel = np.ones((2,2),np.uint8)
                img = cv2.erode(img,kernel,iterations=1)

            
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img,(3,3),0)

            
            img = img + np.random.normal(0,7,img.shape)
            img = np.clip(img,0,255)
        
        img = img.astype(np.float32)/255.0
        img = (img - 0.1307)/0.3081
        img = torch.tensor(img).unsqueeze(0)  # 1x28x28
        label = torch.tensor(self.labels[idx])
        return img,label


full_train_dataset = EMNISTDataset(TRAIN_CSV, augment=True)
test_dataset = EMNISTDataset(TEST_CSV, augment=False)


train_size = int(0.8*len(full_train_dataset))
val_size = len(full_train_dataset)-train_size
train_dataset, val_dataset = random_split(full_train_dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=128)
test_loader = DataLoader(test_dataset,batch_size=128)

print("Train:",len(train_dataset))
print("Validation:",len(val_dataset))
print("Test:",len(test_dataset))


class EMNIMT_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((3,3))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,26)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = EMNIMT_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
    model.eval()
    print("Модель завантажена")
else:
    epochs = 8
    best_val_acc = 0
    train_losses, val_accs = [],[]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        train_losses.append(avg_loss)

        # Валідація
        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
        val_acc = correct/total
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(),MODEL_PATH)
            print("Модель збережена")

    
    plt.figure()
    plt.plot(train_losses,label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("training_loss.svg",format="svg")
    plt.close()
    print("Навчання завершено")


model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)
print("Test Accuracy:", round(correct/total,4))


def preprocess_char(img):

    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Бінаризація
    _, img = cv2.threshold(img,0,255,
                           cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

   
    kernel = np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

    coords = cv2.findNonZero(img)
    if coords is None:
        return None

    x,y,w,h = cv2.boundingRect(coords)
    img = img[y:y+h,x:x+w]

    # Нормалізація 
    scale = 20/max(h,w)
    img = cv2.resize(img,(int(w*scale),int(h*scale)))

    canvas = np.zeros((28,28),dtype=np.float32)

    y_offset = (28-img.shape[0])//2
    x_offset = (28-img.shape[1])//2

    canvas[y_offset:y_offset+img.shape[0],
           x_offset:x_offset+img.shape[1]] = img/255.0

    tensor = torch.tensor(canvas)

    # orientation fix
    tensor = torch.rot90(tensor,1,[0,1])
    tensor = torch.flip(tensor,[1])

    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = (tensor-0.1307)/0.3081

    return tensor.to(device)

def recognize_char(img_path):
    img = cv2.imread(img_path)
    tensor = preprocess_char(img)
    if tensor is None: return ""
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return classes[pred]


if os.path.exists("imgEMNIST"):
    files = glob("imgEMNIST/*.png")+glob("imgEMNIST/*.jpg")
    for f in files:
        print(f,"->",recognize_char(f))
