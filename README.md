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

| **Entry Monitoring (Frame Breakdown)** | **System Architecture & Logic** |
| :--- | :--- |
| <img src="https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(127).png" width="400" alt="Frame Breakdown"> | <img src="https://raw.githubusercontent.com/RizosConstantinos/AI_Identification_System/main/images/diagr.png" width="400" alt="Architectural Diagram"> |
| *Visualizing the multi-stage frame processing pipeline, from camera input to detection and classification.* | *The overall system diagram illustrating the data flow from frame capture to final count. (as seen in image_2.png)* |

---

| **Real-Time Entrance & Exit Analysis** | **Integrated Grand Total Logic** |
| :--- | :--- |
| <img src="https://raw.githubusercontent.com/RizosConstantinos/AI_Identification_System/main/images/Screenshot%20(51).jpg" width="400" alt="In/Out Counters"> | <img src="https://raw.githubusercontent.com/RizosConstantinos/AI_Identification_System/main/images/Screenshot%20(53).jpg" width="400" alt="Grand Total"> |
| *Screen with live "Entrance Counter" and "Exit Counter" overlay, demonstrating real-time direction detection. (as seen in image_5.png)* | *Live status showing the unique "Grand Total" logic which prevents double counting and data loss.* |


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
