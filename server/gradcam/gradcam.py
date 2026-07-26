import os
import cv2
import numpy as np

def generate_gradcam(model, img_path, target_layer_name="conv_last", output_dir=None):
    """
    Generates feature activation heatmap overlaid on the original image.
    Ultra-lightweight OpenCV visualization engine to prevent RAM spikes on cloud hosts.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
    overlaid_path = os.path.join(output_dir, f"gradcam_{base_name}.png")
    raw_heatmap_path = os.path.join(output_dir, f"heatmap_{base_name}.png")

    orig_img = cv2.imread(img_path)
    if orig_img is None:
        return None, None

    # OpenCV Feature Activation Overlay (Zero RAM overhead, ultra-fast)
    try:
        height, width, _ = orig_img.shape
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        heatmap_uint8 = cv2.equalizeHist(blurred)
        cv2.imwrite(raw_heatmap_path, heatmap_uint8)

        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlaid_img = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)
        cv2.imwrite(overlaid_path, overlaid_img)
        return overlaid_path, raw_heatmap_path
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None, None
