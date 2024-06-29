# text_augmentation.py

from nltk.tokenize import word_tokenize
import random

def augment_text(text, weather=None, location=None, time_of_day=None):
    words = word_tokenize(text)
    augmented_words = words[:]

    if weather:
        weather_phrases = {
            'rainy': ['in the rain', 'on a rainy day', 'during a downpour'],
            'foggy': ['in the fog', 'on a foggy day', 'with limited visibility'],
            'sunny': ['on a sunny day', 'under clear skies'],
            'cloudy': ['on a cloudy day', 'with overcast skies'],
            'snowy': ['in the snow', 'on a snowy day', 'during snowfall']
        }
        if weather in weather_phrases:
            augmented_words.append(random.choice(weather_phrases[weather]))

    if location:
        augmented_words.append(f'in {location}')
    
    if time_of_day:
        time_phrases = {
            'morning': ['in the morning', 'at dawn', 'early in the day'],
            'afternoon': ['in the afternoon', 'during the day', 'later in the day'],
        }
        if time_of_day in time_phrases:
            augmented_words.append(random.choice(time_phrases[time_of_day]))
    
    return ' '.join(augmented_words)

if __name__ == "__main__":
    text = "Driving through the city"
    weather = "rainy"
    location = "Boston"
    time_of_day = "afternoon"
    
    augmented_text = augment_text(text, weather, location, time_of_day)
    print("Original Text:", text)
    print("Augmented Text:", augmented_text)
