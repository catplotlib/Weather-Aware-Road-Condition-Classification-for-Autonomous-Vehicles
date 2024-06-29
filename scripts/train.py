import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.feature_extraction.text import CountVectorizer
import cv2

road_classes = {'clean': 0, 'dirty': 1}

def encode_label(road_condition):
    return road_classes[road_condition]

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.vectorizer = CountVectorizer(max_features=100)
        self.vectorizer.fit(texts)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Image not found at {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        
        text = self.texts[idx]
        text_vector = self.vectorizer.transform([text]).toarray().squeeze()
        text_tensor = torch.tensor(text_vector, dtype=torch.float32)
        
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image, text_tensor, label_tensor

class MultimodalNN(nn.Module):
    def __init__(self, num_text_features, num_classes):
        super(MultimodalNN, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_image_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        
        self.fc1 = nn.Linear(num_image_features + num_text_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, image, text):
        image_features = self.cnn(image)
        image_features = image_features.view(image_features.size(0), -1)
        combined_features = torch.cat((image_features, text), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, texts, labels in dataloader:
            images, texts, labels = images, texts, labels 
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def main():
    metadata_path = "../data/augmented_images/metadata.txt"
    
    image_paths = []
    texts = []
    labels = []
    
    with open(metadata_path, "r") as f:
        for line in f:
            image_path, text, weather, location, time_of_day, label = line.strip().split('\t')
            image_paths.append(image_path)
            texts.append(text)
            labels.append(int(label))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MultimodalDataset(image_paths, texts, labels, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    num_text_features = 100  
    num_classes = len(road_classes) #clean,dirty
    model = MultimodalNN(num_text_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    model_path = 'multimodal_road_condition_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")

if __name__ == "__main__":
    main()