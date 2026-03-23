# AI-Driven Real-Time Occupancy Tracking System | Deep Learning & Computer Vision
### Real-Time Human Analytics & Smart Occupancy Monitoring System

---

For a more in-depth understanding of the theoretical background, mathematical models, and experimental results of this system, you can access the following resources:

* **[Full Diploma Thesis (PDF)](https://github.com/RizosConstantinos/AI_Driven_Occupancy_Tracker_System/blob/main/docs/AI_Identification_System.pdf)**: A comprehensive 110+ pages analysis covering Deep Learning architectures, tracking algorithms, and performance evaluation.
* **[Project Presentation (PPTX)](https://github.com/RizosConstantinos/AI_Driven_Occupancy_Tracker_System/blob/main/docs/Final_Presentation.pdf)**: A concise slide summary used for the final thesis defense at the Hellenic Air Force Academy.

---

## Table of Contents
* [Overview](#overview)
* [Core AI Concepts & Technologies](#core-ai-concepts--technologies)
* [System in Action (Showcase)](#system-in-action-showcase)
* [Architecture & System Logic](#architecture--system-logic)
* [Project Structure](#project-structure)
* [Installation & Usage](#installation--usage)
* [Key Results](#key-results)
* [Author](#author)

---

## Overview
An end-to-end AI-powered system for real-time human detection and occupancy tracking using Computer Vision (OpenCV) and Deep Learning. The system addresses the critical need for automated, high-accuracy monitoring in restricted areas, providing a robust and scalable solution for real-time entry/exit analysis and facility management.

---

## Core AI Concepts & Technologies

* **Domain:** Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision.
* **Deep Learning Models:** Integrated **OpenCV's DNN module** for accurate and robust person detection, optimized for real-time inference.
* **Tracking Algorithm:** Implementation of **Centroid Tracking**, maintaining unique IDs across video frames to ensure accurate counting even in challenging conditions.
* **Logic-Based Counting:** Advanced vector-based analysis to determine entry/exit direction (In/Out) by analyzing the displacement of centroids relative to a pre-defined virtual boundary.

---

## System in Action (Showcase)

### 1. Detection & Identification Pipeline
The system processes each frame to detect human figures and assign unique tracking IDs using Centroid Tracking.

| **Initial Frame Detection** | **Unique ID Assignment** |
| :--- | :--- |
| ![Frame Detection](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(127).png) | ![ID Tracking](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(122).png) |
| *Step 1: Analyzing the input stream for human features using Deep Learning.* | *Step 2: Assigning persistent IDs to track movement across the scene.* |


### 2. Real-Time Counting & Occupancy Logic
By monitoring the displacement of centroids relative to a virtual boundary, the system triggers "In" or "Out" events.

| **Directional Monitoring** | **Live Occupancy Status** |
| :--- | :--- |
| ![In-Out Monitoring](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(120).png) | ![Final Status](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(54).png) |
| *Step 3: Calculating directional vectors to update entrance and exit counters.* | *Step 4: Providing a real-time 'Grand Total' for building occupancy management.* |

---

## Architecture & System Logic

To ensure high accuracy and data consistency, the system follows a structured pipeline. Below is the architectural flowchart and the resulting real-time monitoring interface.

### System Flowchart
This diagram illustrates the logic from the initial video capture to the final decision-making process for counting and occupancy tracking.

| **Flow Diagram** | **Logic** |
| :--- | :--- |
| ![Flow Diagram](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(51).png) | ![Logic](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/diagr.png) |

### Hardware System and Programming

| **Rasberry Pi - Programming** | **Final System** |
| :--- | :--- |
| ![Rasberry Pi - Programming](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(52).png) | ![Final System](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(53).png) |

---

## Project Structure

A clean and modular directory structure is maintained to ensure scalability, maintainability, and ease of deployment.

| **Repository Filesystem** | **Directory Breakdown** |
| :--- | :--- |
| ![Folder Structure](https://github.com/RizosConstantinos/AI_Identification_System/blob/main/images/Screenshot%20(55).png) | **- `models/`**: Pre-trained Deep Learning weights and configurations.<br>**- `images/`**: Project documentation assets and showcase visual data.<br>**- `src/`**: Core Python implementation and tracking logic.<br>**- `docs/`**: Full Diploma Thesis documentation (PDF). |

### File Descriptions:
* **`main.py`**: The entry point of the application.
* **`centroidtracker.py`**: Custom implementation of the Centroid Tracking algorithm.
* **`requirements.txt`**: List of Python dependencies for environment reproduction.
* **`diagr.png`**: High-level system architecture flowchart.

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
The system successfully monitors multiple access points, providing a synchronized **Grand Total** of people within a restricted area. This feature is critical for safety protocols, security auditing, and dynamic resource allocation. The project was developed and evaluated at the **Hellenic Air Force Academy**, receiving high marks for technical execution and real-world applicability.

---

## Author

**Rizos Constantinos**
- LinkedIn: [www.linkedin.com/in/rizosconstantinos](https://www.linkedin.com/in/rizosconstantinos/)
- GitHub: [https://github.com/RizosConstantinos](https://github.com/RizosConstantinos)

---

## If you found this useful, feel free to star the repo!
