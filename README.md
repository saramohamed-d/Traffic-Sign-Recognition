# 🚦 Traffic Sign Recognition using CNN

## 💡 Overview
This deep learning project uses a **Convolutional Neural Network (CNN)** to recognize and classify traffic signs from images.  
It was trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset to identify different types of road signs.

## 🧠 Objective
To build a computer vision model that can automatically classify traffic signs with high accuracy, aiding self-driving car systems and intelligent transport solutions.

## 🧰 Tools & Technologies
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV  

## ⚙️ How It Works
1. Load and preprocess the dataset (resize, normalize, and label images).  
2. Split the dataset into training and testing sets.  
3. Design a **CNN architecture** for feature extraction and classification.  
4. Train the model and evaluate performance using accuracy metrics.  
5. Test on unseen images and display predicted labels.

## 📊 Results
- Achieved high accuracy on the test dataset.  
- Model can correctly recognize various traffic sign categories.  
- Visualization includes confusion matrix and example predictions.

## ⚠️ Dataset Notice
The **GTSRB dataset** used in this project is **not included** in this repository due to file size limitations on GitHub.  
You can download it from the official source here:  
👉 [GTSRB Dataset (Kaggle)](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-images)  

After downloading, place the dataset folders (`Train`, `Test`, etc.) in the project directory before running the script.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/saramohamed-d/Traffic-Sign-Recognition.git
