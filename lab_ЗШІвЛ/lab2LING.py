import os
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader
from glob import glob
import matplotlib.pyplot as plt  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


nato_letters = [
"Alpha","Bravo","Charlie","Delta","Echo","Foxtrot","Golf","Hotel",
"India","Juliett","Kilo","Lima","Mike","November","Oscar","Papa",
"Quebec","Romeo","Sierra","Tango","Uniform","Victor","Whiskey",
"Xray","Yankee","Zulu"
]

digits = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

classes = nato_letters + digits
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}



class SpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for cls in classes:
            path = os.path.join(root_dir, cls)
            for file in glob(os.path.join(path,"*.mp3")):
                self.files.append((file, class_to_idx[cls]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        return mfcc, torch.tensor(label)

def collate_fn(batch):
    features = [item[0].T for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    padded = []
    for f in features:
        pad = max_len - f.shape[0]
        if pad > 0:
            f = torch.nn.functional.pad(f,(0,0,0,pad))
        padded.append(f)
    padded = torch.stack(padded)
    return padded.to(device), labels.to(device)



class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(40,64,3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(128,128,batch_first=True)
        self.fc = nn.Linear(128,num_classes)

    def forward(self,x):
        x = x.permute(0,2,1)  # B x 40 x T
        x = self.cnn(x)
        x = x.permute(0,2,1)  # B x T x C
        _, (h,_) = self.lstm(x)
        h = h[-1]
        return self.fc(h)



dataset = SpeechDataset("dataset")
loader = DataLoader(dataset,batch_size=16,shuffle=True,collate_fn=collate_fn)

model = SpeechModel(len(classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()



model_file = "speech_nato.pth"
loss_history = []
acc_history = []

if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {model_file}")
else:
    print("⚡ Training model...")
    epochs = 7
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x,y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)

        epoch_loss = total_loss/len(loader)
        epoch_acc = correct/total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f"Epoch {epoch+1} | Loss={epoch_loss:.4f} | Acc={epoch_acc:.4f}")

    torch.save(model.state_dict(), model_file)
    print(f"✅ Model saved to {model_file}")


if loss_history and acc_history:  
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.plot(acc_history, marker='o', color='orange')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

def word_to_char(word):
    if word in nato_letters:
        return word[0]
    if word in digits:
        return str(digits.index(word))
    return ""

def recognize_sequence(file_list):

    model.eval()
    result = ""

    for file in file_list:
        
        y, sr = librosa.load(file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = torch.tensor(mfcc,dtype=torch.float32).T.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(mfcc)
            pred = out.argmax(1).item()

        word = idx_to_class[pred]
        result += word_to_char(word)

    return result


files = ["audio/One.mp3","audio/November.mp3","audio/Eight.mp3","audio/Juliett.mp3"]
print("Flight number:", recognize_sequence(files))
