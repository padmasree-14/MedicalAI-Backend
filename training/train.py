import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(BASE_DIR, "server")
MODEL_DIR = os.path.join(SERVER_DIR, "model")
METRICS_DIR = os.path.join(SERVER_DIR, "static", "metrics")
DATA_DIR = os.path.join(BASE_DIR, "training", "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Generate Synthetic Chest X-Ray Images
def generate_synthetic_xray(label_name, img_id, size=224):
    """
    Generates a synthetic chest X-Ray.
    Healthy: Clear dark lung shapes with light ribs.
    Pneumonia: Lung shapes containing cloudy/opaque white patches.
    """
    # Background chest cavity (light gray)
    img = np.ones((size, size), dtype=np.uint8) * 180
    
    # Draw dark lung shapes (left and right lobes)
    # Left lung
    cv2.ellipse(img, (90, 110), (35, 80), 10, 0, 360, 40, -1)
    # Right lung
    cv2.ellipse(img, (134, 110), (35, 80), -10, 0, 360, 40, -1)
    
    # Blur background lungs to make them look soft
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Draw spine and collar bones (white)
    cv2.line(img, (112, 10), (112, 210), 200, 6) # Spine
    cv2.line(img, (50, 40), (112, 60), 210, 5) # Collar left
    cv2.line(img, (174, 40), (112, 60), 210, 5) # Collar right
    
    # Draw rib cages (faint white curved horizontal lines)
    for y in range(60, 190, 20):
        # Left ribs
        cv2.ellipse(img, (60, y), (40, 15), 30, 0, 90, 160, 2)
        # Right ribs
        cv2.ellipse(img, (164, y), (40, 15), -30, 180, 90, 160, 2)
        
    # Introduce pneumonia anomalies (cloudy white opacities inside lungs)
    if label_name == "PNEUMONIA":
        # Left lung opacity
        cv2.circle(img, (80, 120), 20, 140, -1)
        cv2.circle(img, (90, 140), 15, 120, -1)
        # Right lung opacity
        cv2.circle(img, (140, 100), 25, 130, -1)
        cv2.circle(img, (130, 130), 18, 140, -1)
        
    # Apply global Gaussian blur to make it resemble a fuzzy radiography
    img = cv2.GaussianBlur(img, (11, 11), 0)
    
    # Add random noise to simulate image capture noise
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return img

def create_dataset():
    """
    Creates folder structure for training/validation and saves synthetic images.
    """
    print("Generating synthetic Chest X-Ray dataset...")
    splits = {
        "train": {"NORMAL": 200, "PNEUMONIA": 200},
        "val": {"NORMAL": 50, "PNEUMONIA": 50}
    }
    
    for split, label_counts in splits.items():
        for label, count in label_counts.items():
            split_dir = os.path.join(DATA_DIR, split, label)
            os.makedirs(split_dir, exist_ok=True)
            for i in range(count):
                img = generate_synthetic_xray(label, i)
                filepath = os.path.join(split_dir, f"{split}_{label}_{i}.png")
                cv2.imwrite(filepath, img)
    print("Dataset generated successfully.")

# 2. Build Custom Convolutional Neural Network (CNN) Model
def build_model(input_shape=(224, 224, 1)):
    """
    Builds a custom 3-conv-layer CNN.
    Includes explicit name 'conv_last' on the final convolution layer for Grad-CAM.
    """
    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3 (Target layer for Grad-CAM)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_last'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate():
    # Make sure dataset exists
    create_dataset()
    
    # Data generator (Normalization to [0,1])
    # Mild augmentation for chest radiographies (minor shifts/rotations, no zoom/vertical flip)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Log class mapping
    class_indices = train_generator.class_indices
    print("Class indices mapping:", class_indices)
    
    # Instantiate and summarize model
    model = build_model()
    model.summary()
    
    # Train
    print("Training CNN model...")
    history = model.fit(
        train_generator,
        epochs=8,
        validation_data=val_generator
    )
    
    # Save Model
    model_path = os.path.join(MODEL_DIR, "model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save class mappings
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(class_indices, f)
    
    # 3. Model Evaluation
    # Reset validation generator
    val_generator.reset()
    y_true = val_generator.classes
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # ROC Curve
    # Index 1 is PNEUMONIA
    fpr, tpr_vals, _ = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = float(auc(fpr, tpr_vals))
    
    # Save curves
    # Loss Curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss', color='#2563EB', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#0EA5E9', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold', color='#1E3A8A')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    
    # Accuracy Curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc', color='#2563EB', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='#0EA5E9', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold', color='#1E3A8A')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()
    
    # ROC Plot
    plt.figure()
    plt.plot(fpr, tpr_vals, color='#2563EB', label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold', color='#1E3A8A')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    
    # Save metrics JSON
    metrics_summary = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": roc_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "epochs": len(history.history['loss']),
        "history": {
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']],
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Metrics saved successfully to {metrics_path}")
    print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
