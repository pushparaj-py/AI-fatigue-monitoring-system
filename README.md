# Real-Time AI Sleep Detection 💤

## 📌 About

**Real-Time AI Sleep Detection (RASD)** is an AI-based workplace monitoring system designed to detect signs of employee drowsiness or sleep in real time using computer vision. The system analyzes facial and eye movements through a live camera feed, combining trained deep learning models to generate alerts when prolonged eye closure, yawning, or sleep-like behavior is detected.

The project also includes an **Admin/Manager Module** for organization-wide monitoring of employee alertness and detection records.

## 🚀 Features

* Real-time drowsiness and sleep detection from a live webcam feed
* Eye-state classification (open/closed) using a trained CNN
* Yawn detection using a separate trained CNN
* Temporal pattern analysis using LSTM to distinguish sustained eye closure from normal blinking
* 4-level alertness classification: **Alert → Early Drowsiness → Drowsy → Critical**
* Automatic audio and visual alerts when drowsiness is detected
* Employee registration and login
* Admin/Manager login and organization-wide employee management
* Session tracking and detection record logging

## 🛠️ Technologies Used

**Backend:** Python, Flask, MySQL
**Computer Vision:** OpenCV, MediaPipe (facial landmark detection, EAR/MAR calculation)
**Machine Learning:** TensorFlow / Keras — CNN (eye-state & yawn classification), LSTM (temporal drowsiness pattern analysis)
**Frontend:** HTML, CSS, JavaScript, Bootstrap

## 📊 Model Performance

| Model           | Task                                   | Accuracy |
| --------------- | -------------------------------------- | -------: |
| CNN (Eye-State) | Open vs. closed eye classification     |   99.06% |
| CNN (Yawn)      | Yawning vs. not-yawning classification |   98.87% |
| LSTM            | Sustained eye closure vs. normal blink |   85.04% |

The LSTM is tuned to prioritize recall for the drowsiness-indicator class, placing greater emphasis on detecting genuine sustained eye-closure events.

## 👥 Modules

### Employee Module

* Registration and login
* Real-time monitoring
* Sleep/drowsiness detection
* Audio and visual alerts
* Personal detection history

### Admin/Manager Module

* Admin login
* Organization-wide employee management
* Monitor detection records across employees
* View employee alertness information

## ⚙️ How It Works

```text
Camera
   ↓
Face & Eye Landmark Detection (MediaPipe)
   ↓
Eye-State CNN + Yawn CNN
   ↓
LSTM Temporal Pattern Analysis
   ↓
Alertness Classification
(Alert / Early Drowsiness / Drowsy / Critical)
   ↓
Alert + Database Logging
```

## 🎯 Purpose

The system is designed to help organizations monitor employee alertness and promote a safer and more productive workplace through real-time AI-based drowsiness detection.

## 🔭 Future Work

* Incorporate real drowsy-driving/fatigue datasets such as NTHU-DDD to move beyond proxy labels from alert-actor footage
* Improve model generalization using larger and more diverse datasets
* Add detailed analytics and reporting for managers
* Optional hardware extension using Raspberry Pi and an external buzzer for standalone deployment
