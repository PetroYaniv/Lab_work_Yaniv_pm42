import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize, rotate
import cv2
import numpy as np
import os
from glob import glob
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.RandomRotation(15),           # випадкове обертання ±15°
    transforms.ToTensor(),                    # у тензор [0,1]
    transforms.Lambda(lambda x: torch.clamp(x + 0.05*torch.randn_like(x), 0, 1))  # шум і обмежуємо [0,1]
])



train_dataset = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="mnist_data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



import matplotlib.pyplot as plt

# ------------------ тренування ------------------
epochs = 5
train_losses = []  # список для збереження loss

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# ------------------ графік ------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, epochs+1), train_losses, marker='o', color='blue')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.show()
plt.savefig("training_loss.png", dpi=300)
plt.close()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
print(f"Test Accuracy: {correct/total:.4f}")


from skimage import io, color, transform, filters
import numpy as np

def preprocess_image(image_path):
    img = io.imread(image_path)

  
    if len(img.shape) == 3:
        if img.shape[2] == 4:        
            img = img[:, :, :3]      
        img = color.rgb2gray(img)   

    
    
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0

   
    img = 1.0 - img

    thresh = filters.threshold_otsu(img)
    img = img > thresh
    img = img.astype(np.float32)

    coords = np.column_stack(np.where(img > 0))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1+1, x0:x1+1]

    img = transform.resize(img, (20, 20), anti_aliasing=True)

    padded = np.zeros((28, 28))
    padded[4:24, 4:24] = img

    return padded

def center_and_resize(img):
    img_uint8 = (img * 255).astype(np.uint8)
    
    
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img_uint8[y:y+h, x:x+w]
    digit_resized = resize(digit, (28, 28), anti_aliasing=True)
    digit_resized = digit_resized.astype(np.float32) / 255.0  # назад у [0,1]
    return digit_resized



def recognize_digit(image_path, model):
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    best_prob = 0
    best_digit = None
    img = preprocess_image(image_path)
    for angle in rotations:
        img_rot = rotate(img, angle, resize=False)
        tensor = torch.tensor(img_rot, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1).max().item()
            digit = output.argmax(dim=1).item()
            if prob > best_prob:
                best_prob = prob
                best_digit = digit
    return best_digit, best_prob


image_files = glob("images/*.png") + glob("images/*.jpg")
os.makedirs("results", exist_ok=True)

results = {}
for img_path in image_files:
    digit, prob = recognize_digit(img_path, model)
    results[os.path.basename(img_path)] = {"digit": int(digit), "probability": float(prob)}
    print(f"{img_path}: Digit={digit}, Probability={prob:.3f}")



