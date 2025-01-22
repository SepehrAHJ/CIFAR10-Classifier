# CIFAR-10 Classifier

This project focuses on building a deep learning model to classify images from the CIFAR-10 dataset. The model is based on the ResNet101 architecture, which has been pretrained on ImageNet and fine-tuned on the CIFAR-10 dataset to improve its performance.

## Project Overview

The CIFAR-10 dataset contains 60,000 images, divided into 10 classes, with each class containing 6,000 images. The dataset is split into training (50,000 images) and testing (10,000 images) datasets. The main objective of this project is to train a neural network model on the CIFAR-10 dataset and achieve high classification accuracy.

## Model

The model used in this project is **ResNet101**, a convolutional neural network (CNN) that has been pretrained on the ImageNet dataset. The pretrained model is fine-tuned on the CIFAR-10 dataset by training only the fully connected layers while freezing the weights of the convolutional layers.

### Training Configuration
- **Input Size**: Images are resized to 224x224 pixels, as required by the ResNet101 architecture.
- **Batch Size**: 32
- **Epochs**: 16 epochs
- **Optimizer**: Adam optimizer with learning rate of 0.1 for fine-tuning the fully connected layers.
- **Loss Function**: Cross-Entropy Loss, which is commonly used for multi-class classification tasks.
- **Learning Rate Scheduler**: StepLR scheduler, reducing the learning rate by a factor of 10 every 10 epochs.

## Results

The model was trained for 50 epochs and achieved an accuracy of approximately **83%** on the validation set.

## Requirements

To run this project, you will need the following Python libraries:

- torch
- torchvision
- pandas
- matplotlib
- numpy
- scikit-learn
- seaborn

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/SepehrAHJ/CIFAR10-Classifier.git
