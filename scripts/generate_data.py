import os
import random
import cv2
from datetime import datetime
from text_augmentation import augment_text
from image_augmentation import apply_weather_effects

weather_map = {
    'clear sky': 'sunny',
    'few clouds': 'cloudy',
    'scattered clouds': 'cloudy',
    'broken clouds': 'cloudy',
    'overcast clouds': 'cloudy',
    'mist': 'foggy',
    'smoke': 'foggy',
    'haze': 'foggy',
    'fog': 'foggy',
    'light rain': 'rainy',
    'moderate rain': 'rainy',
    'heavy intensity rain': 'rainy',
    'very heavy rain': 'rainy',
    'extreme rain': 'rainy',
    'freezing rain': 'rainy',
    'light snow': 'snowy',
    'snow': 'snowy',
    'heavy snow': 'snowy'
}
locations = ['New York', 'Los Angeles', 'Boston', 'Chicago']
times_of_day = ['morning', 'afternoon']

original_image_dir = '../data/test'
# augmented_image_dir = '../data/augmented_images'
augmented_image_dir = '../data/evaluate_images'
os.makedirs(augmented_image_dir, exist_ok=True)

road_condition_metadata_path = '../data/road_condition_metadata.csv'

def generate_synthetic_metadata(num_samples):
    metadata = []
    for _ in range(num_samples):
        weather = random.choice(list(weather_map.values()))
        location = random.choice(locations)
        time_of_day = random.choice(times_of_day)
        metadata.append((weather, location, time_of_day))
    return metadata

def augment_image(image_path, weather):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    augmented_image = apply_weather_effects(image, weather)
    return augmented_image

def main():
    num_samples = len(os.listdir(original_image_dir))
    metadata = generate_synthetic_metadata(num_samples)
    
    image_paths = [os.path.join(original_image_dir, img) for img in os.listdir(original_image_dir)]
    augmented_texts = []
    augmented_image_paths = []

    road_condition_labels = {}
    with open(road_condition_metadata_path, "r") as f:
        next(f)  
        for line in f:
            image_name, label = line.strip().split(',')
            road_condition_labels[image_name] = label

    for i, (image_path, (weather, location, time_of_day)) in enumerate(zip(image_paths, metadata)):
        original_image_name = os.path.basename(image_path)
        
        augmented_image = augment_image(image_path, weather)
        augmented_image_path = os.path.join(augmented_image_dir, f"augmented_{i}.jpg")
        cv2.imwrite(augmented_image_path, augmented_image)
        augmented_image_paths.append(augmented_image_path)
        
        text = "Driving through the city"
        augmented_text = augment_text(text, weather, location, time_of_day)
        augmented_texts.append(augmented_text)
        
        road_condition_label = road_condition_labels.get(original_image_name, "")
        
        print(f"Processed {i+1}/{num_samples} images")

    with open(os.path.join(augmented_image_dir, "metadata.txt"), "w") as f:
        for image_path, text, (weather, location, time_of_day), label in zip(augmented_image_paths, augmented_texts, metadata, [road_condition_labels.get(os.path.basename(path), "") for path in image_paths]):
            f.write(f"{image_path}\t{text}\t{weather}\t{location}\t{time_of_day}\t{label}\n")

if __name__ == "__main__":
    main()