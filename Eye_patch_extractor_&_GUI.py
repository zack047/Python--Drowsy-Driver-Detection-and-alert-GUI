# Importing Project Dependencies
import numpy as np
import cv2
import pandas as pd
import json
from pathlib import Path
import pickle
import time
import winsound
import streamlit as st

# Using Haar-cascade classifier from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
fallback_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Loading the trained model for prediction purpose
model = None
scaler = None
model_type = "sklearn"

try:
    with open('eye_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Loaded sklearn model successfully")
except FileNotFoundError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model('my_model (1).h5')
        model_type = "tensorflow"
        print("Loaded TensorFlow model successfully")
    except:
        print("ERROR: No trained model found! Please run train_model_sklearn.py first.")
        model = None
class_metadata_path = Path("eye_class_names.json")
class_metadata = {}
if class_metadata_path.exists():
    try:
        class_metadata = json.loads(class_metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read class metadata: {exc}")

class_names = class_metadata.get("class_names", [])
open_class_index = class_metadata.get("open_class_index")
closed_class_index = class_metadata.get("closed_class_index")

if open_class_index is None or closed_class_index is None:
    if class_names:
        open_class_index = next((i for i, name in enumerate(class_names) if "open" in name.lower()), None)
        closed_class_index = next((i for i, name in enumerate(class_names) if "close" in name.lower()), None)

    if open_class_index is None:
        open_class_index = 1
    if closed_class_index is None:
        closed_class_index = 0

print(f"Class mapping: {class_names}, open={open_class_index}, closed={closed_class_index}")

MODEL_INPUT_SIZE = (86, 86)
EYE_MIN_REL_AREA = 0.03
EYE_OPEN_THRESHOLD = 0.45     # More sensitive threshold for open eye detection
FRAME_OPEN_THRESHOLD = 0.45   # Lower threshold to detect open eyes
PREDICTION_MARGIN = 0.05      # Lower margin for better sensitivity


def get_columns(count):
    if hasattr(st, "columns"):
        return st.columns(count)
    return st.beta_columns(count)


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def preprocess_eye_patch(eye_patch):
    eye_patch = cv2.resize(eye_patch, MODEL_INPUT_SIZE)
    if not class_names:
        return eye_patch
    # The training dataset (archive/) is grayscale-only (R=G=B in every sample,
    # captured on an IR/monochrome camera). A color webcam frame carries chrominance
    # the model never saw, which corrupts the color-histogram features below.
    # Desaturating here keeps the live feed in the same domain as training.
    gray = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2GRAY)
    eye_patch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extract_eye_features(eye_patch):
    """Extract features from eye patch for sklearn model"""
    gray = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2GRAY)

    # Color histogram features (per channel)
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


def pad_box(x, y, w, h, pad_ratio, frame_w, frame_h):
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = clamp(x - pad_x, 0, frame_w - 1)
    y1 = clamp(y - pad_y, 0, frame_h - 1)
    x2 = clamp(x + w + pad_x, 0, frame_w)
    y2 = clamp(y + h + pad_y, 0, frame_h)
    return x1, y1, x2, y2


def select_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def detect_eye_patches(face_roi):
    """Return left and right eye patches, using eye detection first and geometry fallback."""
    face_h, face_w = face_roi.shape[:2]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    candidates = []
    for cascade in (eye_cascade, fallback_eye_cascade):
        if cascade.empty():
            continue
        detected = cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(max(18, face_w // 10), max(18, face_h // 10)),
        )
        for (ex, ey, ew, eh) in detected:
            area_ratio = (ew * eh) / float(face_w * face_h)
            if area_ratio >= EYE_MIN_REL_AREA and ey < int(face_h * 0.65):
                candidates.append((ex, ey, ew, eh))
        if len(candidates) >= 2:
            break

    if len(candidates) >= 2:
        candidates = sorted(candidates, key=lambda box: box[0])[:2]
        eye_patches = []
        eye_boxes = []
        for (ex, ey, ew, eh) in candidates:
            x1, y1, x2, y2 = pad_box(ex, ey, ew, eh, 0.30, face_w, face_h)
            patch = face_roi[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            eye_patches.append(preprocess_eye_patch(patch))
            eye_boxes.append((x1, y1, x2, y2))
        if len(eye_patches) == 2:
            return eye_patches[0], eye_patches[1], eye_boxes

    # Fallback: approximate eye regions from the upper face geometry.
    top = int(face_h * 0.18)
    bottom = int(face_h * 0.62)
    left_eye_box = (
        int(face_w * 0.10),
        top,
        int(face_w * 0.48),
        bottom,
    )
    right_eye_box = (
        int(face_w * 0.52),
        top,
        int(face_w * 0.90),
        bottom,
    )

    def crop_box(box):
        x1, y1, x2, y2 = box
        x1 = clamp(x1, 0, face_w - 1)
        y1 = clamp(y1, 0, face_h - 1)
        x2 = clamp(x2, x1 + 1, face_w)
        y2 = clamp(y2, y1 + 1, face_h)
        patch = face_roi[y1:y2, x1:x2]
        return preprocess_eye_patch(patch), (x1, y1, x2, y2)

    left_patch, left_box = crop_box(left_eye_box)
    right_patch, right_box = crop_box(right_eye_box)
    return left_patch, right_patch, [left_box, right_box]


def predict_eye_open_probability(eye_patch):
    if model is None:
        return 0.5, 0.5, 0, 0.5

    if model_type == "sklearn":
        # Extract features and scale
        features = extract_eye_features(eye_patch)
        features_scaled = scaler.transform([features])

        # Get probability predictions
        probs = model.predict_proba(features_scaled)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Map probabilities to open/closed classes
        if len(probs) >= 2:
            # probs[0] = closed_eye, probs[1] = open_eye (based on alphabetical sorting)
            closed_prob = float(probs[closed_class_index]) if closed_class_index < len(probs) else float(probs[0])
            open_prob = float(probs[open_class_index]) if open_class_index < len(probs) else float(probs[1])
        else:
            open_prob = 0.5
            closed_prob = 0.5

        return open_prob, closed_prob, predicted_class, confidence
    else:
        # TensorFlow model
        eye_batch = np.expand_dims(eye_patch, axis=0)
        preds = model.predict(eye_batch, verbose=0)[0]
        open_prob = float(preds[open_class_index])
        closed_prob = float(preds[closed_class_index])
        predicted_class = int(np.argmax(preds))
        confidence = float(np.max(preds))
        return open_prob, closed_prob, predicted_class, confidence


def classify_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
    face = select_largest_face(faces)
    if face is None:
        return None

    x, y, w, h = face
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = pad_box(x, y, w, h, 0.06, frame_w, frame_h)
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None

    left_eye, right_eye, eye_boxes = detect_eye_patches(face_roi)
    left_open_prob, left_closed_prob, left_label, left_conf = predict_eye_open_probability(left_eye)
    right_open_prob, right_closed_prob, right_label, right_conf = predict_eye_open_probability(right_eye)
    left_margin = left_open_prob - left_closed_prob
    right_margin = right_open_prob - right_closed_prob
    mean_open_prob = (left_open_prob + right_open_prob) / 2.0

    # Simplified logic: Eyes are open if mean probability is above threshold
    frame_open = mean_open_prob >= FRAME_OPEN_THRESHOLD

    # Debug output
    print(f"DEBUG: left_open={left_open_prob:.3f}, left_closed={left_closed_prob:.3f}")
    print(f"DEBUG: right_open={right_open_prob:.3f}, right_closed={right_closed_prob:.3f}")
    print(f"DEBUG: mean_open_prob={mean_open_prob:.3f}, threshold={FRAME_OPEN_THRESHOLD}")
    print(f"DEBUG: frame_open={frame_open}")

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for (ex1, ey1, ex2, ey2) in eye_boxes:
        cv2.rectangle(display_frame, (x1 + ex1, y1 + ey1), (x1 + ex2, y1 + ey2), (0, 255, 0), 2)

    return {
        "frame": display_frame,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_open_prob": left_open_prob,
        "right_open_prob": right_open_prob,
        "left_closed_prob": left_closed_prob,
        "right_closed_prob": right_closed_prob,
        "left_label": left_label,
        "right_label": right_label,
        "left_conf": left_conf,
        "right_conf": right_conf,
        "frame_open": frame_open,
        "mean_open_prob": mean_open_prob,
    }

# Title for GUI
st.title('Drowsiness Detection')
img = []

# Navigation Bar
nav_choice = st.sidebar.radio('Navigation', ('Home', 'Sleep Detection', 'Help Us Improve'), index=0)

# Home page
if nav_choice == 'Home':
    st.header('Prevents sleep deprivation road accidents, by alerting drowsy drivers.')
    st.image('test.jpg')
    st.markdown('<b>In accordance with the survey taken by the Times Of India, about 40 % of road </b>'
                '<b>accidents are caused</b> '
                '<b>due to sleep deprivation & fatigued drivers. In order to address this issue, this app will </b>'
                '<b>alert such drivers with the help of deep learning models and computer vision.</b>'
                '', unsafe_allow_html=True)
    st.image('sleep.jfif', width=300)
    st.markdown('<h1>How to use?<br></h1>'
                '<b>1. Go to Sleep Detection page from the Navigation Side-Bar.</b><br>'
                '<b>2. Make sure that, you have sufficient amount of light, in your room.</b><br>'
                '<b>3. Align yourself such that, you are clearly visible in the web-cam and '
                'stay closer to the web-cam. </b><br>'
                '<b>4. Web-cam will take 3 pictures of you, so keep your eyes in the same state'
                ' (open or closed) for about 5 seconds.</b><br>'
                '<b>5. If your eyes are closed, the model will make a beep sound to alert you.</b><br>'
                '<b>6. Otherwise, the model will continue taking your pictures at regular intervals of time.</b><br>'
                '<font color="red"><br><b>For the purpose of the training process of the model, '
                'dataset used is available <a href="https://www.kaggle.com/kutaykutlu/drowsiness-detection", '
                'target="_blank">here</a></font></b>'
                , unsafe_allow_html=True)
    
# Sleep Detection page
elif nav_choice == 'Sleep Detection':
    st.header('Image Prediction')
    cap = 0
    st.success('Please look at your web-cam, while following all the instructions given on the Home page.')
    st.warning(
        'Keeping the eyes in the same state is important but you can obviously blink your eyes, if they are open!!!')
    b = st.progress(0)
    for i in range(100):
        time.sleep(0.0001)
        b.progress(i + 1)

    start = st.radio('Options', ('Start', 'Stop'), key='Start_pred', index=1)

    if start == 'Start':
        open_votes = 0
        closed_votes = 0
        valid_predictions = 0
        frame_samples = []
        latest_result = None
        st.markdown('<font face="Comic sans MS"><b>Detected Facial Region of Interest(ROI)&emsp;&emsp;&emsp;&emsp;&emsp;Extractd'
                    ' Eye Features from the ROI</b></font>', unsafe_allow_html=True)
        
        cap = cv2.VideoCapture(0)
        try:
            # Collect a few frames and vote on the result to reduce blink noise.
            for _ in range(3):
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.2)
                    continue

                result = classify_frame(frame)
                if result is None:
                    frame_samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    continue

                latest_result = result
                frame_samples.append(cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB))
                open_votes += int(result["frame_open"])
                closed_votes += int(not result["frame_open"])
                valid_predictions += 1
                print(
                    f"left_open={result['left_open_prob']:.3f} right_open={result['right_open_prob']:.3f} "
                    f"open={result['frame_open']}"
                )

            if valid_predictions == 0:
                st.warning('No face detected. Move closer to the camera and try again.')
                st.stop()

            # Display the latest processed frame and the two eye patches.
            img_container = get_columns(4)
            img_container[0].image(frame_samples[-1], width=250)
            if latest_result is not None:
                img_container[2].image(latest_result["left_eye"], width=150)
                img_container[3].image(latest_result["right_eye"], width=150)

        finally:
            cap.release()

        # If the majority of frames show closed eyes, then alert the driver.
        if closed_votes >= open_votes:
            st.error('Eye(s) are closed')
            for _ in range(10):  # 20 beeps for 10 seconds (500ms each)
                winsound.Beep(4000, 500)  # Higher frequency (3000 Hz) and 500ms duration
        else:
            st.success('Eyes are Opened')

        # Warning message for retry
        st.warning('Please select "Stop" and then "Start" to try again')

# Help Us Improve page
else:
    st.header('Help Us Improve')
    st.success('We would appreciate your Help!!!')
    st.markdown(
        '<font face="Comic sans MS">To make this app better, we would appreciate your small amount of time.</font>'
        '<font face="Comic sans MS">Let me take you through, some of the basic statistical analysis of this </font>'
        '<font face="Comic sans MS">model. <br><b>Accuracy with naked eyes = 99.5%<br>Accuracy with spectacles = 96.8%</b><br></font> '
        '<font face="Comic sans MS">As we can see here, accuracy with spectacles is not at all spectacular, and hence to make this app </font>'
        '<font face="Comic sans MS">better, and to use it in real-time situations, we require as much data as we can gather.</font> '
        , unsafe_allow_html=True)
    st.warning('NOTE: Your identity will be kept anonymous, and only your eye-patch will be extracted!!!')
    # Image upload
    img_upload = st.file_uploader('Upload Image Here', ['png', 'jpg', 'jpeg'])
    if img_upload is not None:
        prog = st.progress(0)
        to_add = cv2.imread(str(img_upload.read()), 0)
        to_add = pd.DataFrame(to_add)
        
        # Save it in the database
        to_add.to_csv('Data_from_users.csv', mode='a', header=False, index=False, sep=';')
        for i in range(100):
            time.sleep(0.001)
            prog.progress(i + 1)
        st.success('Uploaded Successfully!!! Thank you for contributing.')
