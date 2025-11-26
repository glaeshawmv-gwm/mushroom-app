import os
import numpy as np
import pandas as pd
import cv2
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.spatial.distance import mahalanobis
from numpy.linalg import pinv
import joblib

# reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

BASE_DIR = r"C:\Users\Glaesha\Documents\Thesis\mushroom_dataset"
CLASSES = [
    "contamination_bacterialblotch",
    "contamination_cobweb",
    "contamination_greenmold",
    "healthy_bag",
    "healthy_mushroom",
    "not_mushroom"
]

IMG_SIDE = 160
IMG_SIZE = (IMG_SIDE, IMG_SIDE)
BATCH_SIZE = 16
DO_FINETUNE = True
EPOCHS_FT = 20
TOP_UNFREEZE = 10
LR = 1e-4

PCA_COMPONENTS = 40
MAHAL_GLOBAL_FALLBACK = 4.0
PROB_CONFIDENT = 0.70
TTA_STEPS = 12

os.makedirs("misclassified_examples", exist_ok=True)

# -------------------------------
# 1. Prepare dataset
# -------------------------------
data = []
for cls in CLASSES:
    folder = os.path.join(BASE_DIR, cls)
    if not os.path.isdir(folder):
        print("WARNING: missing folder:", folder)
        continue
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            data.append([os.path.join(folder, fname), cls])

df = pd.DataFrame(data, columns=["path", "label"]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE)

# -------------------------------
# 2. Image generators
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.18,
    brightness_range=(0.75,1.25),
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='reflect'
)
val_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
test_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="path", y_col="label", classes=CLASSES,
    target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_STATE
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="path", y_col="label", classes=CLASSES,
    target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col="path", y_col="label", classes=CLASSES,
    target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False
)

# -------------------------------
# 3. Build and fine-tune EfficientNet
# -------------------------------
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIDE, IMG_SIDE, 3))
if DO_FINETUNE:
    base_model.trainable = True
    for layer in base_model.layers[:-TOP_UNFREEZE]:
        layer.trainable = False
else:
    base_model.trainable = False

inp = tf.keras.layers.Input(shape=(IMG_SIDE, IMG_SIDE, 3))
x = base_model(inp, training=DO_FINETUNE)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.45)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.35)(x)
out = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
ft_model = tf.keras.Model(inputs=inp, outputs=out)

ft_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
    metrics=['accuracy']
)

# -------------------------------
# 4. Train model
# -------------------------------
y_train_idx = train_df['label'].map({c:i for i,c in enumerate(CLASSES)}).values
cw = compute_class_weight('balanced', classes=np.unique(y_train_idx), y=y_train_idx)
class_weights = {i: w for i, w in enumerate(cw)}

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

ft_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FT,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# -------------------------------
# 5. Feature extraction
# -------------------------------
feature_extractor = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D(name="gap")])

def extract_features_from_generator(generator, extractor):
    steps = int(np.ceil(generator.n / generator.batch_size))
    all_feats, all_labels = [], []
    generator.reset()
    for _ in range(steps):
        x, y = next(generator)
        f = extractor.predict(x, verbose=0)
        all_feats.append(f)
        all_labels.append(np.argmax(y, axis=1))
    return np.vstack(all_feats), np.hstack(all_labels)

X_train, y_train = extract_features_from_generator(train_gen, feature_extractor)
X_val, y_val = extract_features_from_generator(val_gen, feature_extractor)
X_test, y_test = extract_features_from_generator(test_gen, feature_extractor)

# -------------------------------
# 6. RandomForest + Calibration
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=1000, max_depth=8, min_samples_split=8,
    min_samples_leaf=5, max_features='sqrt',
    class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
cal_rf.fit(X_val, y_val)

# -------------------------------
# 7. PCA + Mahalanobis stats
# -------------------------------
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

class_mahal_stats = {}
for i, cls in enumerate(CLASSES):
    mask_train = (y_train == i)
    Xc_train = X_train_pca[mask_train]
    mu = np.mean(Xc_train, axis=0) if Xc_train.shape[0] > 0 else np.mean(X_train_pca, axis=0)
    cov = np.cov(Xc_train.T) if Xc_train.shape[0] > 1 else np.cov(X_train_pca.T)
    invcov = pinv(cov + np.eye(cov.shape[0])*1e-6)
    mask_val = (y_val == i)
    threshold = float(np.percentile([mahalanobis(x, mu, invcov) for x in X_val_pca[mask_val]] if mask_val.sum() > 0 else [MAHAL_GLOBAL_FALLBACK], 95))
    threshold = max(threshold, 0.6)
    class_mahal_stats[cls] = {"mu": mu, "invcov": invcov, "mahal_thresh": threshold}

# -------------------------------
# 8. Class thresholds
# -------------------------------
val_probs_cal = cal_rf.predict_proba(X_val)
class_thresholds = {}
for i, cls in enumerate(CLASSES):
    mask = (y_val == i)
    if mask.sum() == 0:
        class_thresholds[cls] = 0.12
    else:
        maxp_for_class = np.max(val_probs_cal[mask], axis=1)
        thr = max(maxp_for_class.mean() - 1.2*maxp_for_class.std(), 0.10)
        class_thresholds[cls] = float(thr)

# -------------------------------
# 9. Save models
# -------------------------------
feature_extractor.save("efficientnet_feature_extractor_1280.h5")
joblib.dump(cal_rf, "calibrated_rf.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(class_thresholds, "class_thresholds.pkl")
joblib.dump(class_mahal_stats, "class_mahal_stats.pkl")

print("Training complete and all model components saved!")
