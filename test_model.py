#!/usr/bin/env python
# Quick test script to verify the trained model works

import pickle
import json
import cv2
import numpy as np
from pathlib import Path

print("=" * 60)
print("Testing Drowsiness Detection Model")
print("=" * 60)

# Load metadata
print("\n[*] Loading metadata...")
with open('eye_class_names.json', 'r') as f:
    metadata = json.load(f)
print(f"    Class names: {metadata['class_names']}")
print(f"    Open index: {metadata['open_class_index']}")
print(f"    Closed index: {metadata['closed_class_index']}")

# Load model and scaler
print("\n[*] Loading model and scaler...")
try:
    with open('eye_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("    [+] Model loaded successfully")
except Exception as e:
    print(f"    [-] Error loading model: {e}")
    exit(1)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("    [+] Scaler loaded successfully")
except Exception as e:
    print(f"    [-] Error loading scaler: {e}")
    exit(1)

# Test with sample image from archive
print("\n[*] Testing with sample images from archive...")
closed_files = list(Path('./archive/closed_eye').glob('*'))
open_files = list(Path('./archive/open_eye').glob('*'))
test_samples = {
    'closed_eye': closed_files[0] if closed_files else None,
    'open_eye': open_files[0] if open_files else None,
}

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
    """Extract features from eye patch"""
    gray = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2GRAY)

    hist_b = cv2.calcHist([eye_patch], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([eye_patch], [1], None, [16], [0, 256])
    hist_r = cv2.calcHist([eye_patch], [2], None, [16], [0, 256])
    hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_features = np.array([
        np.mean(edge_magnitude),
        np.std(edge_magnitude),
        np.sum(edge_magnitude),
    ])

    stats = np.array([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
    ])

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.array([
        np.mean(np.abs(laplacian)),
        np.std(laplacian),
    ])

    features = np.concatenate([hist, edge_features, stats, texture])
    return features.astype(np.float32)

for eye_type, sample_path in test_samples.items():
    if sample_path is None:
        print(f"    [!] No sample images found for {eye_type}")
        continue

    img = cv2.imread(str(sample_path))
    if img is None:
        print(f"    [-] Failed to load image: {sample_path}")
        continue

    # Preprocess image
    img = preprocess_eye_image(img, (86, 86))
    features = extract_eye_features(img)
    features_scaled = scaler.transform([features])

    # Predict
    probs = model.predict_proba(features_scaled)[0]
    predicted_class = np.argmax(probs)
    confidence = np.max(probs)

    class_name = metadata['class_names'][predicted_class]
    print(f"\n    {eye_type.upper()}:")
    print(f"      Predicted: {class_name} ({confidence:.2%})")
    print(f"      Closed prob: {probs[0]:.4f}, Open prob: {probs[1]:.4f}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("You can now run: streamlit run Eye_patch_extractor_&_GUI.py")
print("=" * 60)
