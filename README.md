AI-Powered Assistive Navigation System for the Visually Impaired
📌 Overview

This project is an AI-based assistive navigation system designed to help visually impaired individuals navigate their surroundings safely. It uses real-time object detection and converts visual information into audio feedback.

🚨 Problem Statement

Navigating independently is a major challenge for visually impaired individuals. Existing solutions are often expensive or limited in functionality. There is a need for an affordable, intelligent, and real-time navigation aid.

💡 Solution

This system uses a camera to capture live surroundings and applies a deep learning model to detect objects. The detected objects are then communicated to the user through voice alerts, enabling better awareness and safer navigation.

⚙️ Features
Real-time object detection
Audio feedback using text-to-speech
Helps identify nearby obstacles
Lightweight and easy to run
Can be extended to wearable systems
🛠️ Tech Stack
Python
OpenCV
YOLOv5 (Object Detection)
pyttsx3 (Text-to-Speech)
🧠 How It Works
Captures live video from the camera
Processes frames using YOLOv5 model
Detects objects in real time
Converts detected object names into speech
Alerts the user continuously
▶️ How to Run
Clone the repository

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py
Ensure camera access is enabled


🚀 Future Improvements
Distance estimation of objects
Direction-based alerts (left/right)
Mobile or wearable device integration
Improved detection accuracy
Emergency/SOS feature
📊 Applications
Assistive technology for visually impaired individuals
Smart navigation systems
AI-based safety tools
