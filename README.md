# Chest X-Ray Classification with ResNet50V2

This project uses a deep learning approach to classify chest X-ray images into two categories (Normal and Pneumonia). The model is built using TensorFlow and Keras, leveraging the ResNet50V2 pre-trained architecture, data augmentation, and K-fold cross-validation for robust evaluation.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Training and Evaluation](#training-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Features
- **Pre-trained Model**: ResNet50V2 for feature extraction.
- **Data Augmentation**: Includes rotation, zoom, and horizontal flip for robust training.
- **K-Fold Cross-Validation**: Ensures a robust performance evaluation across folds.
- **Visualization**: Includes training and validation accuracy/loss plots.

## Dataset
The dataset contains chest X-ray images categorized into two classes: 
- `NORMAL`
- `PNEUMONIA`

The dataset is structured into three directories:
- `train`: Training data
- `val`: Validation data
- `test`: Testing data

**Source**: [Chest X-Ray Images Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Model Architecture
- **Base Model**: ResNet50V2 pre-trained on ImageNet, used for feature extraction.
- **Additional Layers**:
  - Global Average Pooling
  - Dense Layer (128 units, ReLU activation)
  - Output Layer (1 unit, Sigmoid activation)

## Data Augmentation
The training data is augmented using the following techniques:
- Rescaling
- Rotation (up to 40 degrees)
- Width and Height Shift
- Shear Transformation
- Zoom
- Horizontal Flip

## K-Fold Cross-Validation
K-Fold Cross-Validation is applied with 5 splits to ensure robust evaluation. Each fold trains the model on 4 subsets and validates on the remaining subset.

## Training and Evaluation
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### Early Stopping
Training stops if validation loss does not improve for 5 consecutive epochs.

### Results
- **Train Accuracy**: 96.78%
- **Validation Accuracy**: 93.75%
- **Test Accuracy**: 90.54%

### Plots
Training and validation accuracy/loss plots are included to visualize performance.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chest-xray-classification.git
   cd chest-xray-classification
