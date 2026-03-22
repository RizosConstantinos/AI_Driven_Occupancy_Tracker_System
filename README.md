# Automated AI Identification System: A Deep Learning & Computer Vision Approach
### People Counter & Occupancy Tracker

---

## Overview
An end-to-end AI-powered system for real-time human detection and occupancy tracking using Computer Vision (OpenCV) and Deep Learning. Features automated counting and entry/exit monitoring.

---

## Core AI Concepts & Technologies

* **Domain:** Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision.
* **Deep Learning Models:** Integrated **OpenCV's DNN module** for accurate and robust person detection.
* **Tracking Algorithm:** Implementation of **Centroid Tracking**, maintaining unique IDs across video frames to ensure accurate counting even in challenging conditions.
* **Logic-Based Counting:** Vector-based analysis to determine entry/exit direction (In/Out) and calculate real-time building occupancy.

---

## System in Action (Showcase)

### 1. Detection & Identification Pipeline
The system processes each frame to detect human figures and assign unique tracking IDs using Centroid Tracking.

| **Initial Frame Detection** | **Unique ID Assignment** |
| :--- | :--- |
| ![Frame Detection](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(127).png) | ![ID Tracking](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(122).png) |
| *Step 1: Analyzing the input stream for human features using Deep Learning.* | *Step 2: Assigning persistent IDs to track movement across the scene.* |

---

### 2. Real-Time Counting & Occupancy Logic
By monitoring the displacement of centroids relative to a virtual boundary, the system triggers "In" or "Out" events.

| **Directional Monitoring** | **Live Occupancy Status** |
| :--- | :--- |
| ![In-Out Monitoring](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(120).png) | ![Final Status](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(54).png) |
| *Step 3: Calculating directional vectors to update entrance and exit counters.* | *Step 4: Providing a real-time 'Grand Total' for building occupancy management.* |


---

## Installation & Usage
1. Clone the repository:
   `git clone https://github.com/RizosConstantinos/AI_Identification_System.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the application:
   `python main.py`

---

## Key Results
The system successfully monitors multiple access points, providing a synchronized **Grand Total** of people within a restricted area, a critical feature for safety and security management.

---

## Author

**Rizos Constantinos**  
- LinkedIn: www.linkedin.com/in/constantinos-rizos-0589b5254  
- GitHub: https://github.com/RizosConstantinos
  
---

## ⭐ If you found this useful, feel free to star the repo!
