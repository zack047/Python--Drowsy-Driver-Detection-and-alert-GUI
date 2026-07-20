# Importing Project Dependencies
import numpy as np
import os
import json
import argparse
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Setting up config for GPU training
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as exc:
        print(f"Could not enable GPU memory growth: {exc}")

# Loading in all the images and assigning target classes
def preprocess_eye_image(img, image_size=(86, 86)):
    img = cv2.resize(img, image_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_dataset_dir():
    parser = argparse.ArgumentParser(description="Train the drowsiness eye open/closed classifier.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DROWSINESS_DATASET_DIR", "./archive"),
        help="Path to the dataset root folder that contains the class subfolders.",
    )
    args, _ = parser.parse_known_args()
    dataset_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Make sure the archive folder exists with open_eye and closed_eye subdirectories."
        )
    return dataset_dir


def load_images(folder, image_size=(86, 86)):
    imgs, labels, class_names = [], [], []
    folder = os.path.abspath(folder)
    class_dirs = [name for name in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, name))]
    for class_idx, foldername in enumerate(class_dirs):
        loc = os.path.join(folder, foldername)

        class_names.append(foldername)
        for filename in os.listdir(loc):
            img_path = os.path.join(loc, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = preprocess_eye_image(img, image_size)
            if img.shape == (image_size[1], image_size[0], 3):
                imgs.append(img)
                labels.append(class_idx)
        print(f"Loaded class: {foldername}")

    return np.asarray(imgs), np.asarray(labels), class_names


folder = get_dataset_dir()
X, y, class_names = load_images(folder)
print(f"Classes: {class_names}")
print(f"Dataset shape: {X.shape}, labels: {y.shape}")

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
        },
        f,
        indent=2,
    )
print("Saved class mapping to eye_class_names.json")

# Splitting the data into 2 separate training and testing sets
def train_test_split(X, y, testing_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split_idx = int(len(X) * (1.0 - testing_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


X_train, y_train, X_test, y_test = train_test_split(X, y, testing_size=0.2)
if len(X_train) > 0:
    print(X_train[0].shape)

# Model building using sequential API
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ],
    name="eye_augmentation",
)

model = keras.Sequential(
        [
            keras.Input(shape=(86, 86, 3)),
            layers.Rescaling(1.0 / 255.0),
            data_augmentation,
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.35),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax'),
        ]
    )
print(model.summary())

# Model compilation with keeping track of accuracy while training & evaluation process
model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

callbacks = [
    ModelCheckpoint("my_model (1).h5", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
]

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.15,
    callbacks=callbacks,
    shuffle=True,
)

model.evaluate(X_test, y_test, batch_size=32)

# Saving the model
model.save('my_model (1).h5')
