import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_normalized_hist(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not find: {image_path}")
        return None
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist / hist.sum()

# File names must match exactly what you moved into the folder
SAMPLES = {
    "Target: Oil": "../data/sample/sample_oil.jpg",
    "Target: Clean": "../data/sample/sample_not_oil.jpg",
    "AI Input (NASA)": "../outputs/debug_ai_input.jpg"
}

plt.figure(figsize=(12, 6))

colors = {"Target: Oil": "black", "Target: Clean": "blue", "AI Input (NASA)": "red"}

for label, path in SAMPLES.items():
    hist = get_normalized_hist(path)
    if hist is not None:
        plt.plot(hist, color=colors[label], label=label, linewidth=2 if "AI" in label else 1)
        if "AI" in label:
            plt.fill_between(range(256), hist.flatten(), alpha=0.2, color=colors[label])

plt.title("Histogram Signature Comparison")
plt.xlabel("Pixel Intensity (0=Dark/Oil, 255=Bright/Land)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()