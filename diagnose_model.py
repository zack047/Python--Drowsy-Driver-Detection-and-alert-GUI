#!/usr/bin/env python
# Diagnostic script to check model performance on archive data

import pickle
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 70)
print("Diagnosing Model Performance on Archive Data")
print("=" * 70)

# Load metadata
with open('eye_class_names.json', 'r') as f:
    metadata = json.load(f)

# Load model and scaler
with open('eye_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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

# Test on archive samples
print("\n[*] Testing on archive samples...")
predictions_all = []
true_labels_all = []
probabilities_all = []

test_classes = {
    'closed_eye': 0,
    'open_eye': 1,
}

for class_name, class_idx in test_classes.items():
    class_path = Path(f'./archive/{class_name}')
    files = list(class_path.glob('*'))[:100]  # Test on 100 samples per class

    print(f"\n  {class_name.upper()} (label={class_idx}):")
    predictions = []
    probs = []

    for file_path in files:
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            img = preprocess_eye_image(img, (86, 86))
            features = extract_eye_features(img)
            features_scaled = scaler.transform([features])

            prob = model.predict_proba(features_scaled)[0]
            pred = np.argmax(prob)

            predictions.append(pred)
            probs.append(prob)
            predictions_all.append(pred)
            true_labels_all.append(class_idx)
            probabilities_all.append(prob)
        except Exception as e:
            continue

    predictions = np.array(predictions)
    probs = np.array(probs)

    if len(predictions) > 0:
        accuracy = np.mean(predictions == class_idx)
        print(f"    Samples tested: {len(predictions)}")
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    Mean closed_prob: {probs[:, 0].mean():.4f}")
        print(f"    Mean open_prob: {probs[:, 1].mean():.4f}")
        print(f"    Correctly classified: {np.sum(predictions == class_idx)}/{len(predictions)}")

# Overall statistics
if len(predictions_all) > 0:
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    overall_accuracy = accuracy_score(true_labels_all, predictions_all)
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels_all, predictions_all)
    print(f"  Predicted  | Closed | Open")
    print(f"  -----------|--------|------")
    print(f"  Closed (0) | {cm[0,0]:4d}   | {cm[0,1]:4d}")
    print(f"  Open (1)   | {cm[1,0]:4d}   | {cm[1,1]:4d}")

    print("\nPer-class Report:")
    print(classification_report(true_labels_all, predictions_all,
                                target_names=['closed_eye', 'open_eye']))

print("\n" + "=" * 70)
