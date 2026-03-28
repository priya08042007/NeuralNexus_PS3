# 🌊 Microplastic Detection and Risk Analysis System

An AI-powered computer vision system that detects microplastic particles, analyzes their physical properties, and classifies environmental risk levels through an interactive Streamlit dashboard.

---

## 🚀 Project Overview

This project aims to automatically detect and analyze microplastic particles such as **fiber, fragment, film, and pellet** from images. It provides:
- Multiple images can be processed at a time
- Categorizes particles using deep learning
- Size estimation using contour detection in micrometer
- Ecological Threat Index based on morphology and size
- Interactive visualization dashboard
- A heatmap on the image showing regions contributing to classification decision
- Downloadable report of all the images

---

## 📊 1. Dataset Used & Preprocessing

### 📥 Dataset Sources:
- Kaggle 
- Datasets used :
- https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision
- https://universe.roboflow.com/iam/microplastics-m7mf5?utm_source=chatgpt.com

### 🧹 Preprocessing Steps:
- Removed corrupted images
- Removed missing or empty label files
- Validated YOLO annotation format
- Checked class balance across categories
- Split dataset into:
  - Training set
  - Validation set
  - Test set
- Created `data.yaml` configuration file for YOLO training

---

## 🤖 2. Model Used & Performance

### 🧠 Model:
- YOLOv8 (Ultralytics)
- Base model: `yolov8s.pt`

### ⚙️ Training Configuration:
- Image size: 640
- Epochs: 50
- Batch size: 128
- Optimizer: Auto (YOLO default)

### 📈 Performance Metrics:
- Precision: 0.943
- Recall: 0.954
- mAP@50: 0.92
- Fast inference suitable for real-time applications

---

Here are the **key features of your microplastic AI project** (clean, simple, with emojis and no code):

---

# 🌊 Key Features of the Project

##  1. AI-Based Object Detection

* Uses an advanced AI model to detect microplastics in images
* Identifies four types:
  * Fiber
  * Fragment
  * Film
  * Pellet
* Provides fast and accurate detection results


##  2. Multi-Image Upload

* Allows users to upload multiple images at once
* Processes all images together
* Generates both overall and image-wise results


##  3. Detection with Cropping

* Automatically extracts detected microplastic regions
* Each particle is analyzed individually


## 4. Particle Size Measurement

* Measures size of each detected particle
* Uses advanced image processing for accurate results


## 5. Smart Risk Scoring System

* Calculates risk based on:
  * Shape of microplastic
  * Size of particle
* Smaller particles → higher risk
* Fiber type → highest risk


## 🚦 6. Risk Classification

* Converts risk into easy categories:
  * 🔴 High Risk
  * 🟡 Medium Risk
  * 🟢 Low Risk

## 📊 7. Real-Time Dashboard

* Displays:
  * Total number of particles
  * Average size
  * Average risk score
* Updates instantly after upload


##  8. Ecological Threat Index

* Shows overall environmental impact in range 0 - 100


## 9. Data Visualization

* Interactive charts for better understanding:
  * Bar chart (class distribution)
  * Pie chart (percentage of types)


##  10. Detailed Image Analysis

* Provides detailed results for each image
* Shows:
  * Number of each type
  * Individual particle details
  * Size and risk information


## 11. User-Friendly Interface

* Clean and modern dashboard design
* Easy navigation and visualization
* Color-coded risk indicators


## 12. High Performance

* Fast processing and real-time results
* Optimized for smooth user experience
------
## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy & Pandas
- Plotly & Matplotlib
- Tensorflow
- BytesIO
- os
- PIL, shutil

---

