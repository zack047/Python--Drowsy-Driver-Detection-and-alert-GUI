#!/usr/bin/env python
# Debug script to test prediction on your own camera images

import pickle
import json
import cv2
import numpy as np
from pathlib import Path

print("=" * 70)
print("Debug: Testing Prediction Logic")
print("=" * 70)

# Load model and metadata
with open('eye_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('eye_class_names.json', 'r') as f:
    metadata = json.load(f)

print("\nClass Metadata:")
print(f"  Classes: {metadata['class_names']}")
print(f"  Open Index: {metadata['open_class_index']}")
print(f"  Closed Index: {metadata['closed_class_index']}")

def preprocess_eye_image(img, image_size=(86, 86)):
    """Preprocess eye image with CLAHE enhancement"""
    img = cv2.resize(img, image_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_eye_features(eye_patch):
    """Extract features from eye patch - MUST MATCH TRAINING"""
    gray = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2GRAY)

    # Color histogram features
    hist_b = cv2.calcHist([eye_patch], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([eye_patch], [1], None, [16], [0, 256])
    hist_r = cv2.calcHist([eye_patch], [2], None, [16], [0, 256])
    hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])

    # Edge features using Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_features = np.array([
        np.mean(edge_magnitude),
        np.std(edge_magnitude),
        np.sum(edge_magnitude),
    ])

    # Statistical features
    stats = np.array([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
    ])

    # Texture features using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.array([
        np.mean(np.abs(laplacian)),
        np.std(laplacian),
    ])

    features = np.concatenate([hist, edge_features, stats, texture])
    return features.astype(np.float32)

# Test with archive samples
print("\n" + "=" * 70)
print("Testing on KNOWN archive samples:")
print("=" * 70)

closed_sample = list(Path('./archive/closed_eye').glob('*'))[0]
open_sample = list(Path('./archive/open_eye').glob('*'))[0]

for sample_type, sample_path in [("CLOSED EYE", closed_sample), ("OPEN EYE", open_sample)]:
    print(f"\n[TEST] {sample_type}: {sample_path.name}")

    img = cv2.imread(str(sample_path))
    img = preprocess_eye_image(img, (86, 86))
    features = extract_eye_features(img)
    features_scaled = scaler.transform([features])

    probs = model.predict_proba(features_scaled)[0]
    pred_class = np.argmax(probs)
    confidence = np.max(probs)

    # Extract probabilities like the GUI does
    open_class_index = metadata['open_class_index']
    closed_class_index = metadata['closed_class_index']

    open_prob = float(probs[open_class_index])
    closed_prob = float(probs[closed_class_index])

    print(f"  Raw probs: {probs}")
    print(f"  Closed prob (index {closed_class_index}): {closed_prob:.4f}")
    print(f"  Open prob (index {open_class_index}): {open_prob:.4f}")
    print(f"  Predicted class: {metadata['class_names'][pred_class]}")
    print(f"  Confidence: {confidence:.4f}")

    # Check thresholds
    FRAME_OPEN_THRESHOLD = 0.50
    EYE_OPEN_THRESHOLD = 0.50

    if open_prob >= EYE_OPEN_THRESHOLD:
        print(f"  Result: OPEN (open_prob {open_prob:.4f} >= {EYE_OPEN_THRESHOLD})")
    else:
        print(f"  Result: CLOSED (open_prob {open_prob:.4f} < {EYE_OPEN_THRESHOLD})")

print("\n" + "=" * 70)
print("THRESHOLDS TO TEST:")
print("=" * 70)
print("Current settings:")
print("  FRAME_OPEN_THRESHOLD = 0.50")
print("  EYE_OPEN_THRESHOLD = 0.50")
print("  PREDICTION_MARGIN = 0.10")
print("\nIf still showing wrong results, try even lower:")
print("  FRAME_OPEN_THRESHOLD = 0.40")
print("  EYE_OPEN_THRESHOLD = 0.40")
print("\nOr investigate if probabilities are inverted.")
print("=" * 70)
