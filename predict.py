import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from scipy.spatial.distance import mahalanobis

# -------------------------------
# 1. Load Models
# -------------------------------
feature_extractor = tf.keras.models.load_model("efficientnet_feature_extractor_1280.h5")
cal_rf = joblib.load("calibrated_rf.pkl")
pca = joblib.load("pca.pkl")
class_thresholds = joblib.load("class_thresholds.pkl")
class_mahal_stats = joblib.load("class_mahal_stats.pkl")

CLASSES = [
    "contamination_bacterialblotch",
    "contamination_cobweb",
    "contamination_greenmold",
    "healthy_bag",
    "healthy_mushroom",
    "not_mushroom"
]
IMG_SIZE = (160, 160)

# -------------------------------
# 2. Preprocess image
# -------------------------------
def load_and_preprocess_image(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = efficientnet_preprocess(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

# -------------------------------
# 3. Safe Prediction
# -------------------------------
def predict_image_safe_v2(img_bgr):
    if img_bgr is None:
        return "not_mushroom", 0.0, 0.0, [0.0]*len(CLASSES)

    x = load_and_preprocess_image(img_bgr)
    feat = feature_extractor.predict(x, verbose=0)  # shape (1,1280)

    # RandomForest prediction
    probs = cal_rf.predict_proba(feat)[0]
    probs = probs * 0.98 + 0.02/len(CLASSES)

    maxp = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]

    feat_pca = pca.transform(feat)
    stats = class_mahal_stats[pred_class]
    m_dist = float(mahalanobis(feat_pca[0], stats["mu"], stats["invcov"]))

    threshold = class_thresholds.get(pred_class, 0.12)
    if maxp < threshold and m_dist > stats.get("mahal_thresh", 4.0):
        return "not_mushroom", maxp, m_dist, probs.tolist()

    return pred_class, maxp, m_dist, probs.tolist()

def predict_from_path(path):
    img = cv2.imread(path)
    return predict_image_safe_v2(img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict mushroom contamination from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    pred_class, maxp, m_dist, probs = predict_from_path(args.image)
    print("Prediction:", pred_class)
    print("Confidence:", round(maxp*100, 2), "%")
    print("Mahalanobis Distance:", round(m_dist, 4))
    print("Class Probabilities:", {cls: round(p,4) for cls,p in zip(CLASSES, probs)})
