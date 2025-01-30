# Deep-Learning-Based-Framework-for-Automated-Grasping-Behavior-Analysis-in-Non-Human-Primates

Deep Learning-Based Framework for Automated Grasping Behavior Analysis in Non-Human Primates
This project provides a framework for processing and analyzing hand motion data in non-human primates using deep learning-based keypoint detection models. The framework automates the generation of accurate CSV files, visualization of annotated videos, and landmark-only annotated videos. Additionally, it incorporates an error detection mechanism for efficient debugging.

Features
📌 CSV File Generation & Error Detection
generate_csv.py:
Extracts and processes detected keypoints from raw video data.
Generates structured CSV files for further kinematic analysis.
Implements an error detection mechanism to efficiently identify and rectify mislabeling or missing keypoints.
The framework is designed to process an entire folder of CSV files, enabling fast and efficient large-scale data analysis.
📌 Right-Hand Side Data Processing
analyze_right.py:
Adds a time axis to the raw data for accurate temporal analysis.
Generates a 3D motion trajectory of the hand, marking the start point, farthest point, and end point.
Computes finger-tip movement distance over time along with corresponding velocity and acceleration curves.
Includes an error detection mechanism to identify inconsistencies.
📌 Overhead View Data Processing
analyze_overhead.py:
Adds a time axis to the raw data for overall motion tracking.
Generates a 3D motion trajectory with key points: start, farthest, and end positions.
Computes overall hand movement trends over time, including velocity and acceleration analysis.
Incorporates an error detection mechanism to improve reliability.

Installation & Dependencies
To use this framework, install the following dependencies:
pip install numpy pandas matplotlib scipy opencv-python