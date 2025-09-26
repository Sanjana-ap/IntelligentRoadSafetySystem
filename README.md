# Intelligent Road Safety Surveillance System

## Project Overview
This project detects potholes and analyzes traffic behavior using CCTV footage. It provides real-time alerts and visualizes data on interactive maps.

## Features
- Pothole detection using YOLOv8
- Real-time alerts with audio notifications
- Backend built with Flask and MongoDB
- Map visualization using Leaflet.js

## Technologies Used
- Python 3.x
- OpenCV (`opencv-python`)
- YOLOv8 (`ultralytics`)
- Flask
- PyMongo (`pymongo`)
- Flask-CORS (`flask-cors`)
- Pyttsx3
- Geopy

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/Intelligent-Road-Safety.git
cd Intelligent-Road-Safety

2.Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt

4.Dataset

The full dataset is not included in this repository due to size.

Download the dataset and place it in the datasets/IDD_FGVD folder:
[Provide link to dataset]

5.Models

YOLOv8 pretrained and custom-trained weights are not included due to size.

Download weights and place them in the models/ folder.

6. Usage
python app.py
