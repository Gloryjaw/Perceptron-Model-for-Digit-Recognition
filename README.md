# Perceptron Model for Digit Recognition

This repository contains a perceptron model designed to recognize handwritten digits from the MNIST dataset. The model was implemented from scratch and achieved an accuracy of 87.26% on the test data.

## Overview

The perceptron model is a simple, yet foundational, machine learning algorithm that serves as a building block for more complex neural networks. This project demonstrates the application of a single-layer perceptron for the task of digit recognition.

## Features

- **Implemented from Scratch**: The perceptron model was created without relying on machine learning libraries like TensorFlow or PyTorch, providing an in-depth understanding of how the perceptron algorithm works.
- **Accuracy**: The model achieves an accuracy of 87.26% on the MNIST test dataset.
- **MNIST Dataset**: The model is trained and tested on the MNIST dataset, which is a benchmark dataset in the field of machine learning for image recognition tasks.

## Model Architecture

- **Input Layer**: 784 neurons (28x28 pixels)
- **Output Layer**: 10 neurons (corresponding to digits 0-9)
- **Activation Function**: ArgMax
- **Learning Rate**: 0.01
- **Training Epochs**: 10

## Dataset

The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image of a handwritten digit (0-9).

## Results


- **Test Accuracy**: 87.26%


