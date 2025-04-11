# 🏡 House Price Classification with Neural Networks

This project trains a neural network using **TensorFlow/Keras** to predict whether a house is considered "expensive" or "not expensive" based on 10 input features such as size, rooms, and location-related data.
This project was used for testing purposes to find out how neuronal networks are working.

---

## 📊 Dataset

The dataset is stored in `data/housepricedata.csv` and contains:

- 10 numerical features per house (e.g., lot size, number of rooms, etc.)
- 1 binary target label:  
  - `1` → expensive  
  - `0` → not expensive

---

## 🧠 Model Overview

The model is a simple feedforward neural network built with Keras:

- Input Layer: 10 features
- Hidden Layers: 2 layers with 32 neurons each (ReLU)
- Output Layer: 1 neuron (Sigmoid activation)
- Loss Function: Mean Squared Error
- Optimizer: Adam
- Metric: Accuracy

---
