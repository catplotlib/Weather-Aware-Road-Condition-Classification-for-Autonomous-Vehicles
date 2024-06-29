import cv2
import numpy as np

def apply_weather_effects(image, weather):
    if weather == 'rainy':
        image = add_rain_effect(image)
    elif weather == 'clouds':
        image = add_cloud_effect(image)
    elif weather == 'foggy':
        image = add_fog_effect(image)
    elif weather == 'snowy':
        image = add_snow_effect(image)
    return image

def add_rain_effect(image):
    rain = cv2.imread('../data/rain.png', cv2.IMREAD_UNCHANGED)
    if rain is None:
        raise FileNotFoundError("Rain effect image not found.")
    rain = cv2.resize(rain, (image.shape[1], image.shape[0]))
    if rain.shape[2] == 4:
        alpha_rain = rain[:, :, 3] / 255.0
        for c in range(0, 3):
            image[:, :, c] = image[:, :, c] * (1 - alpha_rain) + rain[:, :, c] * alpha_rain
    else:
        for c in range(0, 3):
            image[:, :, c] = cv2.addWeighted(image[:, :, c], 0.8, rain[:, :, c], 0.2, 0)
    return image

def add_cloud_effect(image):
    clouds = cv2.imread('../data/clouds.png', cv2.IMREAD_UNCHANGED)
    if clouds is None:
        raise FileNotFoundError("Cloud effect image not found.")
    clouds = cv2.resize(clouds, (image.shape[1], image.shape[0]))
    if clouds.shape[2] == 4:
        alpha_clouds = clouds[:, :, 3] / 255.0
        for c in range(0, 3):
            image[:, :, c] = image[:, :, c] * (1 - alpha_clouds) + clouds[:, :, c] * alpha_clouds
    else:
        for c in range(0, 3):
            image[:, :, c] = cv2.addWeighted(image[:, :, c], 0.8, clouds[:, :, c], 0.2, 0)
    return image

def add_fog_effect(image):
    fog = cv2.imread('../data/fog.png', cv2.IMREAD_UNCHANGED)
    if fog is None:
        raise FileNotFoundError("Fog effect image not found.")
    fog = cv2.resize(fog, (image.shape[1], image.shape[0]))
    if fog.shape[2] == 4:
        alpha_fog = fog[:, :, 3] / 255.0
        for c in range(0, 3):
            image[:, :, c] = image[:, :, c] * (1 - alpha_fog) + fog[:, :, c] * alpha_fog
    else:
        for c in range(0, 3):
            image[:, :, c] = cv2.addWeighted(image[:, :, c], 0.8, fog[:, :, c], 0.2, 0)
    return image

def add_snow_effect(image):
    snow = cv2.imread('../data/snow.png', cv2.IMREAD_UNCHANGED)
    if snow is None:
        raise FileNotFoundError("Snow effect image not found.")
    snow = cv2.resize(snow, (image.shape[1], image.shape[0]))
    if snow.shape[2] == 4:
        alpha_snow = snow[:, :, 3] / 255.0
        for c in range(0, 3):
            image[:, :, c] = image[:, :, c] * (1 - alpha_snow) + snow[:, :, c] * alpha_snow
    else:
        for c in range(0, 3):
            image[:, :, c] = cv2.addWeighted(image[:, :, c], 0.8, snow[:, :, c], 0.2, 0)
    return image
