# Deep Learning Drowsy Driver Detection

## Overview

The "Deep Learning Drowsy Driver Detection" project focuses on developing a deep learning-based system that detects whether a driver is drowsy by analyzing facial features, specifically the state of their eyes. This real-time system is designed to enhance road safety by alerting drivers who may be at risk of drowsiness-related accidents.

This project employs advanced deep learning models, including Convolutional Neural Networks (CNNs) and pre-trained architectures like YOLOv8 and EfficientNet, to achieve high accuracy in detecting drowsiness. The model's effectiveness is evaluated using various metrics, with a strong emphasis on real-time application and deployment.

## Table of Contents

- [Problem Objective](#problem-objective)
- [Data Gathering](#data-gathering)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Image Preprocessing](#image-preprocessing)
- [Modeling Approaches](#modeling-approaches)
  - [CNN for Object Detection](#cnn-for-object-detection)
  - [YOLOv8 Pre-trained Model](#yolov8-pre-trained-model)
  - [EfficientNet for Object Classification](#efficientnet-for-object-classification)
- [Results](#results)
- [Challenges and Future Work](#challenges-and-future-work)
- [How to Use](#how-to-use)

## Problem Objective

The primary goal of this project is to develop a deep learning system that can accurately detect drowsiness in drivers by analyzing their eye state in real-time. The system aims to reduce the risk of accidents caused by drowsy driving by providing timely alerts.

## Data Gathering

### Data Sources:

- **Video Data**: Collected from platforms like TikTok and YouTube, focusing on videos where driver eye states can be clearly observed.
- **Frame Extraction**: Frames were generated from the collected videos for further analysis and training.
- **Bounding Boxes**: Bounding boxes were created around the driver’s eyes using Roboflow to provide the necessary data for training object detection models.

## Exploratory Data Analysis (EDA)

### Dataset Overview:

- **Training Data**: 590 images
- **Validation Data**: 418 images
- **No Missing Values**: The dataset is complete, with no missing entries.

### Key Findings:

- **Histogram Analysis**: The distribution of bounding box sizes varies, with a slight right skew, indicating that most bounding boxes are relatively small.
- **Scatterplot Analysis**: Analyzed the relationship between x and y coordinates of bounding boxes, color-coded by width and height, providing insights into the spatial distribution of eye regions in images.

## Image Preprocessing

### Object Classification:

- **Image Conversion**: All images were converted to RGB format to standardize the input.
- **Resizing**: Images were resized to 244x244 pixels, preparing them for input into the CNN models.
- **Label Encoding**: Class labels (e.g., "Eyes Open," "Eyes Closed") were converted into integer format for classification tasks.

### Object Detection:

- **Aspect Ratio Maintenance**: Images were resized while maintaining the original aspect ratio to avoid distortion.
- **Bounding Box Adjustment**: Bounding boxes were recalculated to align with the resized image dimensions, ensuring accuracy in object detection.

## Modeling Approaches

### CNN for Object Detection

- **Model Architecture**:
  - **Input Layer**: Accepts 244x244 pixel images.
  - **Convolutional Layers**: Includes 64 filters with a 3x3 kernel size and ReLU activation.
  - **Output Layers**: Features fully connected layers that output bounding box coordinates.
  - **Regularization**: Dropout rate of 0.5 and L2 regularization applied to prevent overfitting.

### YOLOv8 Pre-trained Model

- **Model Architecture**:
  - **Convolutional Layers**: Multiple convolutional layers followed by Batch Normalization.
  - **C2f Layer**: A custom layer combining convolution and other operations.
  - **SPPF**: Spatial Pyramid Pooling, which enhances feature extraction.
  - **Upsampling and Concat Layers**: Used to increase resolution and combine feature maps.
  - **Detection Layer**: Final layer for object detection.

### EfficientNet for Object Classification

- **Model Architecture**:
  - **Base Model**: EfficientNetB0 pre-trained on ImageNet, used as the base model with custom top layers.
  - **Custom Layers**: Added layers include GlobalAveragePooling2D, Dense layers with ReLU activation, and a final Softmax layer for binary classification.
  - **Model Compilation**: The model was compiled using the Adam optimizer and Sparse Categorical Cross-entropy loss function, with accuracy as the primary metric.

## Results

### Object Detection - CNN

- **Validation IoU**: 0.7475
- **Test IoU**: 0.0993

### Object Detection - YOLOv8

- **MAP50**: Mean Average Precision when IoU overlap is greater than 50%.
- **MAP50-95**: Average of MAP values from 0.5 to 0.95 thresholds.

### Object Classification - CNN

- **Validation Accuracy**: 0.9659
- **Test Accuracy**: 0.9231
- **Precision for "Eyes Closed"**: 1.00
- **Recall for "Eyes Closed"**: 0.81
- **F1 Score for "Eyes Closed"**: 0.9

### Object Classification - EfficientNet

- **Train Accuracy**: 0.9974
- **Test Accuracy**: 0.9487
- **Precision for "Eyes Closed"**: 0.89
- **Recall for "Eyes Closed"**: 1.00
- **F1 Score for "Eyes Closed"**: 0.94

## Challenges and Future Work

### Challenges:

- **Integration of Models**: Challenges in integrating the YOLO model with other classification models for real-time streaming.
- **Data Labeling**: Issues with class naming consistency during bounding box creation in Roboflow.
- **Deployment**: Technical challenges in deploying the model for real-world use.
- **Maintenance**: Ensuring the model remains accurate and effective over time.

### Future Work:

- **Real-Time Deployment**: Implementing the model in real-time systems to provide immediate alerts to drivers.
- **Model Refinement**: Further improving model accuracy and robustness, particularly in diverse real-world scenarios.

## How to Use

Clone the repository:

```bash
git clone https://github.com/sameerab04/Deep_Learning_Drowsy_Driver.git
```
