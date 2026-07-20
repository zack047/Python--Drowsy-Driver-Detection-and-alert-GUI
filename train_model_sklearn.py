# Training script using scikit-learn (simpler alternative)
import numpy as np
import os
import json
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def preprocess_eye_image(img, image_size=(86, 86)):
    """Preprocess eye image with CLAHE enhancement"""
    img = cv2.resize(img, image_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_features(img):
    """Extract advanced features: color histogram, edges, textures, and statistics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Color histogram features (per channel)
    hist_b = cv2.calcHist([img], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [16], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [16], [0, 256])
    hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])

    # 2. Edge features using Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_features = np.array([
        np.mean(edge_magnitude),
        np.std(edge_magnitude),
        np.sum(edge_magnitude),
    ])

    # 3. Statistical features
    stats = np.array([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
    ])

    # 4. Texture features using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.array([
        np.mean(np.abs(laplacian)),
        np.std(laplacian),
    ])

    # Combine all features
    features = np.concatenate([hist, edge_features, stats, texture])
    return features.astype(np.float32)

def load_images(folder, image_size=(86, 86)):
    """Load images and extract features"""
    features_list = []
    labels = []
    class_names = []

    folder = os.path.abspath(folder)
    class_dirs = [name for name in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, name))]

    for class_idx, foldername in enumerate(class_dirs):
        loc = os.path.join(folder, foldername)
        class_names.append(foldername)
        loaded_count = 0

        for filename in os.listdir(loc):
            img_path = os.path.join(loc, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = preprocess_eye_image(img, image_size)
                if img.shape == (image_size[1], image_size[0], 3):
                    features = extract_features(img)
                    features_list.append(features)
                    labels.append(class_idx)
                    loaded_count += 1
                    if loaded_count % 500 == 0:
                        print(f"  Loaded {loaded_count} images from {foldername}...", end='\r')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        print(f"Loaded class: {foldername} ({loaded_count} images)")

    return np.array(features_list, dtype=np.float32), np.array(labels), class_names

# Load data from archive folder
print("Loading images from archive folder...")
folder = os.path.abspath("./archive")
if not os.path.isdir(folder):
    raise FileNotFoundError(f"Archive folder not found at {folder}")

X, y, class_names = load_images(folder)
print(f"\nClasses: {class_names}")
print(f"Dataset shape: {X.shape}, labels: {y.shape}")

if len(X) == 0:
    raise ValueError("No images loaded! Make sure archive/open_eye and archive/closed_eye folders exist with images.")

# Shuffle data before splitting (CRITICAL for balanced train/test)
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

split_idx = int(len(X) * 0.8)
X_train, y_train = X_shuffled[:split_idx], y_shuffled[:split_idx]
X_test, y_test = X_shuffled[split_idx:], y_shuffled[split_idx:]

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"  Training - Closed: {np.sum(y_train==0)}, Open: {np.sum(y_train==1)}")
print(f"  Test - Closed: {np.sum(y_test==0)}, Open: {np.sum(y_test==1)}")

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting classifier (better generalization)
print("Training Gradient Boosting classifier...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    verbose=1
)
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"\nTrain Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Get feature importance
feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-10:]
print(f"Top 10 important features: {top_features}")

# Save model and metadata
print("\nSaving model and metadata...")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save model
with open("eye_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save class metadata
with open("eye_class_names.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "class_names": class_names,
            "open_class_index": next(
                (i for i, name in enumerate(class_names) if "open" in name.lower()),
                None,
            ),
            "closed_class_index": next(
                (i for i, name in enumerate(class_names) if "close" in name.lower()),
                None,
            ),
            "model_type": "sklearn_rf",
        },
        f,
        indent=2,
    )

print("[+] Model saved as 'eye_model.pkl'")
print("[+] Scaler saved as 'scaler.pkl'")
print("[+] Metadata saved as 'eye_class_names.json'")
print("\nTraining complete!")
