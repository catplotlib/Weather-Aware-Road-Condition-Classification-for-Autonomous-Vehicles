# Multimodal Road Condition Classification with Real-Time Weather Integration

## Project Overview

### Problem Statement

Autonomous vehicles rely heavily on accurate environmental perception to navigate safely and efficiently. One of the critical aspects of this perception is understanding road conditions, which can be significantly affected by weather and time of day. Traditional datasets often lack real-time weather data, limiting the ability of models to generalize well under varying environmental conditions.

### Project Solution

This project addresses the gap by augmenting an existing road image dataset with real-time weather and geolocation data. By integrating both image and text data modes, the model can better understand and classify road conditions under diverse weather scenarios. The key innovation is the incorporation of live weather data, enhancing the model's robustness and reliability in real-world applications.

### Key Features

* **Multimodal Approach**: Combines image data with text data derived from real-time weather information.
* **Real-Time Data Integration**: Uses live weather data to augment the dataset, ensuring the model is trained on realistic and current environmental conditions.
* **Comprehensive Augmentation**: Applies realistic weather effects to images to simulate different conditions like rain, fog, and snow.

## Dataset

We are using the Clean/Dirty Road Classification dataset from Kaggle. This dataset provides a variety of road images categorized as clean or dirty, which serves as the foundation for our project.

https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification

## Synthetic Data Creation

For demonstration purposes, we are creating synthetic data by augmenting the original road images with real-time weather effects. This process involves overlaying weather conditions such as rain, fog, and snow onto the images, and adding corresponding descriptive text data. This helps in simulating diverse weather scenarios, enhancing the training data for our model.

## Model Performance

Our trained model achieved a score of 93% on the test dataset, demonstrating its effectiveness in classifying road conditions under various weather scenarios.

## Acknowledgments

- Clean/Dirty Road Classification dataset for providing the base road images.
- OpenWeatherMap and OpenCage for their APIs used in this project.

Feel free to contribute to this project or use it as a basis for further research and development in the field of autonomous driving.# Weather-Aware-Road-Condition-Classification-for-Autonomous-Vehicles
