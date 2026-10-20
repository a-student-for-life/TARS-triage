import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIGURATION ---
# 1. Point to your temp folder (matches your screenshot)
# UPDATE THIS LINE TO THE FULL PATH:
# TODO: Update SEARCH_DIR to your local temp_images directory before running.
SEARCH_DIR = os.getenv("TARS_TEMP_IMAGE_DIR", "temp_images")

# 2. Your Weights File (matches your code)
WEIGHTS_PATH = "optisar_b4_retrained_weights.weights.h5" 
IMG_SIZE = 380 
THRESHOLD = 0.75

# --- 1. MODEL BUILDER (From your code) ---
def build_model_architecture():
    print("🏗️ Rebuilding EfficientNetB4...", flush=True)
    base_model = applications.EfficientNetB4(
        weights=None, # Loaded manually later
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4), 
        layers.Dense(512, activation='relu'), 
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. PREPROCESSING FUNCTIONS ---

def preprocess_raw(img_path):
    """
    BASELINE (BAD): Simulates passing the browser preview directly to AI.
    No domain adaptation. Just resize and scale.
    """
    img = cv2.imread(img_path) # Load as RGB
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img

def preprocess_tars(img_path):
    """
    TARS V3 (GOOD): Your Domain Adaptation Logic.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # A. Speckle
    img_float = img.astype(np.float32)
    noise = np.random.normal(0, 0.5, img.shape)
    img_noise = img_float + (img_float * noise)
    img = np.clip(img_noise, 0, 255).astype('uint8')

    # B. Blur & CLAHE
    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # C. Histogram Shift
    img = np.clip(img.astype(np.int16) + 70, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_rgb = preprocess_input(img_rgb)
    return img_rgb

# --- 3. MAIN EXECUTION ---

# Load Model
model = build_model_architecture()
try:
    model.load_weights(WEIGHTS_PATH)
    print("✅ Weights Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit()

print("\n" + "="*80)
print(f"{'FILENAME':<40} | {'RAW PREDICTION':<15} | {'TARS PREDICTION':<15} | {'RESULT'}")
print("="*80)

# Walk through all folders
total_images = 0
false_positives_prevented = 0

for root, dirs, files in os.walk(SEARCH_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            total_images += 1

            # 1. Run Raw
            raw_input = preprocess_raw(img_path)
            raw_pred = model.predict(np.expand_dims(raw_input, axis=0), verbose=0)[0][0]
            
            # 2. Run TARS
            tars_input = preprocess_tars(img_path)
            tars_pred = model.predict(np.expand_dims(tars_input, axis=0), verbose=0)[0][0]

            # Analysis
            # Assuming these are Clean Sea images (since Oil is rare)
            # If Raw > Threshold (Oil) AND TARS < Threshold (Clean), TARS saved us!
            raw_is_oil = raw_pred > THRESHOLD
            tars_is_oil = tars_pred > THRESHOLD
            
            outcome = ""
            if raw_is_oil and not tars_is_oil:
                outcome = "✅ FP PREVENTED" # Raw failed, TARS succeeded
                false_positives_prevented += 1
            elif raw_is_oil and tars_is_oil:
                outcome = "⚠️ BOTH OIL"
            elif not raw_is_oil and not tars_is_oil:
                outcome = "🟦 BOTH CLEAN"

            print(f"{file[:35]:<40} | {raw_pred:.4f}          | {tars_pred:.4f}          | {outcome}")

print("="*80)
print(f"Total Images Scanned: {total_images}")
print(f"False Positives Prevented: {false_positives_prevented}")
print("="*80)