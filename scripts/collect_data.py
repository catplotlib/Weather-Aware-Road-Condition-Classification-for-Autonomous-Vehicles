# collect_data.py

import requests
from datetime import datetime

def get_location(api_key, location_name):
    base_url = "https://api.opencagedata.com/geocode/v1/json"
    params = {'q': location_name, 'key': api_key}
    response = requests.get(base_url, params=params)
    data = response.json()
    if data['results']:
        latitude = data['results'][0]['geometry']['lat']
        longitude = data['results'][0]['geometry']['lng']
        return latitude, longitude
    else:
        raise Exception("Location not found")

def get_weather(api_key, latitude, longitude):
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'lat': latitude, 'lon': longitude, 'appid': api_key}
    response = requests.get(base_url, params=params)
    return response.json()

def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9/5 + 32

def get_time_of_day():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "morning"
    elif 12 <= current_hour < 17:
        return "afternoon"
    elif 17 <= current_hour < 20:
        return "evening"
    else:
        return "night"

def collect_environmental_data(weather_api_key, geocoding_api_key, location_name):
    latitude, longitude = get_location(geocoding_api_key, location_name)
    weather_data = get_weather(weather_api_key, latitude, longitude)
    time_of_day = get_time_of_day()
    
    weather_data['main']['temp'] = kelvin_to_fahrenheit(weather_data['main']['temp'])
    weather_data['main']['feels_like'] = kelvin_to_fahrenheit(weather_data['main']['feels_like'])
    weather_data['main']['temp_min'] = kelvin_to_fahrenheit(weather_data['main']['temp_min'])
    weather_data['main']['temp_max'] = kelvin_to_fahrenheit(weather_data['main']['temp_max'])

    environmental_data = {
        "weather": weather_data,
        "location": {"latitude": latitude, "longitude": longitude},
        "time_of_day": time_of_day
    }
    return environmental_data
