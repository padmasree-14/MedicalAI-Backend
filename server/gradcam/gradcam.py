import os
import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(model, img_path, target_layer_name="conv_last", output_dir=None):
    """
    Generates a Grad-CAM heatmap overlaid on the original image.
    Works seamlessly across TensorFlow 2.x and Keras 3 Sequential / Functional models.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and preprocess image
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise ValueError(f"Failed to read image for Grad-CAM at {img_path}")
    
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_gray, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = tf.convert_to_tensor(np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)) # Shape: (1, 224, 224, 1)

    # Call model once to ensure build status
    _ = model(img_input)

    # Locate target layer index
    target_idx = -1
    for idx, layer in enumerate(model.layers):
        if layer.name == target_layer_name:
            target_idx = idx
            break
            
    if target_idx == -1:
        # Fallback to last Conv2D layer in model
        for idx in range(len(model.layers) - 1, -1, -1):
            if isinstance(model.layers[idx], tf.keras.layers.Conv2D):
                target_idx = idx
                break

    if target_idx == -1:
        raise ValueError("No Conv2D layer found for Grad-CAM generation.")

    # Forward pass with gradient tracking
    with tf.GradientTape() as tape:
        x = img_input
        conv_output = None
        for idx, layer in enumerate(model.layers):
            x = layer(x)
            if idx == target_idx:
                conv_output = x
                tape.watch(conv_output)
        
        predictions = x
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        pooled_grads = tf.reduce_mean(conv_output, axis=(0, 1, 2))
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_val = conv_output[0]
    heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image size
    height, width, _ = orig_img.shape
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Scale to 0-255 range
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Save raw heatmap
    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
    raw_heatmap_path = os.path.join(output_dir, f"heatmap_{base_name}.png")
    cv2.imwrite(raw_heatmap_path, heatmap_uint8)

    # Apply Jet color map
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose/Overlay heatmap on original image
    overlaid_img = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)
    
    # Save overlaid image
    overlaid_path = os.path.join(output_dir, f"gradcam_{base_name}.png")
    cv2.imwrite(overlaid_path, overlaid_img)

    return overlaid_path, raw_heatmap_path
