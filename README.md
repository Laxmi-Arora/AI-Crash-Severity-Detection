# 🚗 AI-Based Vehicle Crash Severity Detection

## 📌 Overview

This project is a deep learning-based system that automatically detects the severity of vehicle crashes from accident images. The model classifies crashes into three categories:

* Minor
* Moderate
* Major

The project uses transfer learning with Convolutional Neural Networks (CNNs) and provides a user-friendly Streamlit web application for real-time predictions.

---

## 🎯 Objectives

* Automatically classify accident severity from images.
* Compare the performance of different deep learning models.
* Deploy the best-performing model using Streamlit.
* Demonstrate the practical application of AI in road safety.

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* Scikit-learn
* Matplotlib

---

## 📂 Dataset

Dataset Used:
**Balanced Accident Video Dataset**

The dataset contains accident videos categorized into:

* Minor
* Moderate
* Major

Videos were converted into image frames using OpenCV before training the models.

---

## 🔄 Project Workflow

1. Load accident video dataset
2. Extract image frames using OpenCV
3. Preprocess images (Resize and Normalize)
4. Train deep learning models
5. Evaluate model performance
6. Select the best-performing model
7. Deploy the model using Streamlit
8. Predict crash severity from uploaded images

---

## 🧠 Deep Learning Models

The following transfer learning models were implemented:

* MobileNetV2
* EfficientNetB0
* VGG16

---

## 📊 Model Performance

| Model          | Test Accuracy |
| -------------- | ------------- |
| MobileNetV2    | ~58%          |
| EfficientNetB0 | ~34%          |
| VGG16          | ~35%          |

MobileNetV2 achieved the highest accuracy and was selected for deployment.

---

## 💻 Web Application

The project includes a Streamlit-based web application where users can:

* Upload an accident image
* Predict crash severity
* View confidence score
* Get a brief interpretation of the prediction

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Crash-Severity-Detection.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
AI-Crash-Severity-Detection
│
├── app.py
├── Crash_Severity_Detection.ipynb
├── mobilenet_model.h5
├── requirements.txt
├── README.md
├── images/
└── screenshots/
```

---

## 🔮 Future Improvements

* Real-time CCTV video analysis
* Integration with emergency alert systems
* Improved model accuracy using larger datasets
* Deployment on cloud platforms
* Support for live traffic monitoring

---

## 👩‍💻 Author

**Laxmi Arora**

B.Tech Artificial Intelligence & Data Science
