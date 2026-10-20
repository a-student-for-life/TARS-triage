import numpy as np
import cv2

def check_image_usability(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate Texture Strength (Standard Deviation)
    # This measures the "Graininess" of the water
    texture_strength = np.std(img)
    
    print(f"📉 Texture Strength: {texture_strength:.2f}")
    
    if texture_strength < 10:
        print("❌ REJECT: Image is too smooth/compressed. AI will likely fail.")
    elif texture_strength < 25:
        print("⚠️ WARNING: Moderate compression. Synthetic speckle is MANDATORY.")
    else:
        print("✅ PASS: Texture is healthy and rich.")

# Run this on your debug_ai_input.jpg
check_image_usability("../data/sample/browser_preview2.jpg")
check_image_usability("../outputs/debug_ai_input.jpg")
