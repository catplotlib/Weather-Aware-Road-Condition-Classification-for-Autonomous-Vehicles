import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import cv2

from collect_data import collect_environmental_data
from text_augmentation import augment_text
from image_augmentation import apply_weather_effects

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

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

def preprocess_text(text, vectorizer, num_features):
    text_vector = vectorizer.transform([text]).toarray().squeeze()
    if len(text_vector) < num_features:
        text_vector = np.pad(text_vector, (0, num_features - len(text_vector)))
    elif len(text_vector) > num_features:
        text_vector = text_vector[:num_features]
    return torch.tensor(text_vector, dtype=torch.float32).unsqueeze(0)

def evaluate_input(model, image, text, vectorizer, num_text_features):
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image)
        text = preprocess_text(text, vectorizer, num_text_features)
        output = model(image, text)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

def main():
    weather_api_key = os.getenv("WEATHER_API_KEY")
    geocoding_api_key = os.getenv("GEOCODING_API_KEY")
    location_name = "Boston"

    if not weather_api_key or not geocoding_api_key:
        raise ValueError("API keys for weather and geocoding are not set in environment variables.")

    env_data = collect_environmental_data(weather_api_key, geocoding_api_key, location_name)
    weather = env_data["weather"]["weather"][0]["main"].lower()
    time_of_day = env_data["time_of_day"]

    print(f"Weather: {weather}")
    print(f"Time of Day: {time_of_day}")
    
    base_text = "Driving through the city"
    augmented_text = augment_text(base_text, weather, location_name, time_of_day)
    print("Original Text:", base_text)
    print("Augmented Text:", augmented_text)

    image = cv2.imread('../data/dirty.jpg')
    if image is None:
        raise FileNotFoundError("Input image not found.")
    augmented_image = apply_weather_effects(image, weather)
    cv2.imshow("Augmented Image", augmented_image)
    cv2.waitKey(5000)  # Display for 5 seconds
    cv2.destroyAllWindows()

    model_path = 'multimodal_road_condition_model.pth'
    state_dict = torch.load(model_path)
    
    num_text_features = state_dict['fc1.weight'].shape[1] - 512
    print(f"Number of text features in the trained model: {num_text_features}")
    
    model = MultimodalNN(num_text_features=num_text_features, num_classes=2)
    model.load_state_dict(state_dict)
    
    vectorizer = CountVectorizer(max_features=num_text_features)
    vectorizer.fit([augmented_text])
    
    predicted_class, confidence = evaluate_input(model, augmented_image, augmented_text, vectorizer, num_text_features)
    
    class_names = ["clean", "dirty"]  
    print(f"\nModel Prediction:")
    print(f"- Predicted class: {class_names[predicted_class]}")
    print(f"- Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
