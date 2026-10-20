import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ==========================================
# PART 1: DEFINITIONS (Your V2 Function)
# ==========================================
def preprocess_style_v2(img_path, target_size=(380, 380)):
    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        return None
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, target_size)

    # 1. STRONGER SPECKLE (Widen the hill)
    noise_sigma = 30  
    gauss_noise = np.random.normal(0, noise_sigma, img.shape).astype('int16')
    img_noise = img.astype('int16') + gauss_noise
    img = np.clip(img_noise, 0, 255).astype('uint8')

    # 2. GENTLE BLUR
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 3. CONTRAST
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4. BRIGHTNESS ALIGNMENT (+95)
    # This guarantees minimum pixel value is near 95
    img = np.clip(img.astype(np.int16) + 95, 0, 255).astype(np.uint8)

    return img # Return the matrix directly, don't just save it

# ==========================================
# PART 2: EXECUTION (Run & Plot Instantly)
# ==========================================

# 1. SETUP
DATASET_DIR = "/kaggle/working/clean_dataset"
# 👇 REPLACE THIS with the actual path to your raw downloaded browser image
RAW_INPUT_PATH = "browser_download_image.jpg" 

# Check if raw file exists (Update this path if needed!)
if not os.path.exists(RAW_INPUT_PATH):
    # Try to find a fallback if you haven't uploaded a browser image yet
    possible = [f for f in os.listdir(".") if f.endswith(".jpg") and "debug" not in f]
    if possible: RAW_INPUT_PATH = possible[0]

print(f"🔄 Processing raw image: {RAW_INPUT_PATH}")

# 2. GENERATE NEW IMAGE
processed_img = preprocess_style_v2(RAW_INPUT_PATH)

if processed_img is not None:
    # Calculate Histogram of the NEW image immediately
    hist_input = cv2.calcHist([processed_img], [0], None, [256], [0, 256])
    hist_input = hist_input / hist_input.sum() # Normalize

    # 3. GET DATASET STATS (For comparison)
    print("📊 Fetching Dataset Stats...")
    
    # Helper to get stats
    def get_stats(folder):
        if not os.path.exists(folder): return None, None
        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        # Filter out the "White Error Images" dynamically so they don't mess up the graph
        valid_hists = []
        for f in random.sample(files, min(len(files), 50)):
            im = cv2.imread(os.path.join(folder, f), 0)
            if im is None or np.mean(im > 250) > 0.5: continue # Skip bad white images
            h = cv2.calcHist([im], [0], None, [256], [0, 256])
            valid_hists.append(h / h.sum())
        if not valid_hists: return None, None
        arr = np.array(valid_hists).squeeze()
        return np.mean(arr, axis=0), np.std(arr, axis=0)

    clean_mean, clean_std = get_stats(os.path.join(DATASET_DIR, "0"))
    oil_mean, oil_std = get_stats(os.path.join(DATASET_DIR, "1"))

    # 4. PLOT EVERYTHING
    plt.figure(figsize=(12, 6))
    x = np.arange(256)
    
    # Oil (Black)
    if oil_mean is not None:
        plt.plot(x, oil_mean, 'k--', label="Oil Avg")
        plt.fill_between(x, np.maximum(0, oil_mean-oil_std), oil_mean+oil_std, color='k', alpha=0.1)
    
    # Clean (Blue)
    if clean_mean is not None:
        plt.plot(x, clean_mean, 'b--', label="Clean Avg")
        plt.fill_between(x, np.maximum(0, clean_mean-clean_std), clean_mean+clean_std, color='b', alpha=0.1)

    # Your Input (Red)
    plt.plot(x, hist_input, 'r-', linewidth=3, label="Your V2 Input (Live)")
    
    plt.title("Live Debug: V2 Input vs Dataset")
    plt.legend()
    plt.show()
    
    # Save for the AI model to use later
    cv2.imwrite("final_corrected_input.jpg", processed_img)
    print("✅ Saved 'final_corrected_input.jpg'. Use THIS file for prediction.")

else:
    print("❌ Failed to process image. Check RAW_INPUT_PATH.")