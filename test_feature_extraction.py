"""
test_feature_extraction.py

Quick sanity check for detection/feature_extraction.py.
Captures a single frame from your webcam, runs the full feature
extraction pipeline, and prints the resulting feature vector.

Run this BEFORE building the full sequence-generation pipeline,
to confirm model loading, landmark detection, and CNN inference
are all working correctly together.
"""

import cv2
from detection.feature_extraction import extract_features

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("Capturing a frame in 2 seconds — look at the camera...")
cv2.waitKey(2000)

ret, frame = cap.read()
cap.release()

if not ret:
    print("[ERROR] Failed to capture frame from webcam.")
    exit()

print("Running feature extraction...")
features = extract_features(frame)

if features is None:
    print("[RESULT] No face detected in the captured frame. Try again with your face clearly visible.")
else:
    print("\n" + "=" * 60)
    print("FEATURE VECTOR EXTRACTED SUCCESSFULLY")
    print("=" * 60)
    print(f"Left EAR:        {features[0]:.4f}")
    print(f"Right EAR:       {features[1]:.4f}")
    print(f"Average EAR:     {features[2]:.4f}")
    print(f"MAR:             {features[3]:.4f}")
    print(f"Eye CNN score:   {features[4]:.4f}  (0=Closed, 1=Open)")
    print(f"Yawn CNN score:  {features[5]:.4f}  (0=NotYawning, 1=Yawning)")
    print("\nIf these numbers look reasonable (EAR roughly 0.2-0.35 for open eyes,")
    print("eye CNN score close to 1.0 if your eyes were open, yawn score close to 0.0")
    print("if you weren't yawning), the pipeline is working correctly.")