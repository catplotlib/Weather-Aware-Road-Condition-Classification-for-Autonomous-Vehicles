import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.feature_extraction.text import CountVectorizer
import cv2

import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, transform=None, num_text_features=37):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.num_text_features = num_text_features
        self.vectorizer = CountVectorizer(max_features=num_text_features)
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
        
        # Pad or truncate the text vector to match the required number of features
        if len(text_vector) < self.num_text_features:
            text_vector = np.pad(text_vector, (0, self.num_text_features - len(text_vector)))
        elif len(text_vector) > self.num_text_features:
            text_vector = text_vector[:self.num_text_features]
        
        text_tensor = torch.tensor(text_vector, dtype=torch.float32)
        
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image, text_tensor, label_tensor

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texts, labels in dataloader:
            print("Images shape:", images.shape)
            print("Texts shape:", texts.shape)
            print("Labels shape:", labels.shape)
            
            outputs = model(images, texts)
            print("Outputs shape:", outputs.shape)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    print(f"Evaluation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return epoch_loss, epoch_acc


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

def main():
    metadata_path = "../data/evaluate_images/metadata.txt"
    
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model_path = 'multimodal_road_condition_model.pth'
    state_dict = torch.load(model_path)
    
    num_text_features = state_dict['fc1.weight'].shape[1] - 512 
    print(f"Number of text features in the trained model: {num_text_features}")
    
    dataset = MultimodalDataset(image_paths, texts, labels, transform, num_text_features=num_text_features)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = MultimodalNN(num_text_features=num_text_features, num_classes=2)
    model.load_state_dict(state_dict)
    
    criterion = nn.CrossEntropyLoss()
    
    evaluate_model(model, dataloader, criterion)

if __name__ == "__main__":
    main()