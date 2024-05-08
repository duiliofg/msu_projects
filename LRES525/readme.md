# Hyperspectral Image Analysis for Snow Albedo Estimation

This repository contains two Python scripts that utilize deep-learning techniques to analyze hyperspectral images to estimate snow albedo. These scripts demonstrate how to extract training data from hyperspectral imagery and geospatial vector data and how to construct and evaluate convolutional neural networks (CNNs) tailored for classifying various surface types based on their spectral signatures.

## Description

The project is structured around two main scripts:

1. **hyperspectral_cnn_model_v1.py** - This script demonstrates loading hyperspectral images, extracting labeled data using vector shapes, and training a convolutional neural network with dropout and batch normalization layers.

2. **hyperspectral_cnn_model_v2.py** - An advanced version that includes different configurations for the convolutional layers and explores over-sampling techniques to handle class imbalance. It also implements a rigorous evaluation strategy using precision and F1 score metrics.

Both scripts are designed to handle the challenges of hyperspectral image processing, such as high dimensionality and class imbalance, making them suitable for environmental and cryospheric studies.

## Features

- **Data Loading and Processing**: Utilize `rasterio` and `geopandas` to load and preprocess geospatial and hyperspectral data.
- **Deep Learning Model Construction**: Build custom CNN architectures using TensorFlow/Keras.
- **Class Imbalance Handling**: Implement random oversampling to balance the dataset.
- **Performance Evaluation**: Evaluate model performance using precision and F1 score metrics.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- TensorFlow 2.x
- Rasterio
- Geopandas
- Scikit-learn
- Imbalanced-learn


