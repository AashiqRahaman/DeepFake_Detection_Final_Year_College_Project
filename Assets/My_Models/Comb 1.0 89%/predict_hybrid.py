import os
import argparse
import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score  # <-- FIXED: Added accuracy_score
)
from model_attention import ModifiedBranch, MainBranch, Attention
from utils_hybrid import load_and_prep_image

# --- Constants ---
MODEL_PATH = "/kaggle/working/models/"
IMG_DIM = (299, 299)
LABELS_MAP = {0: 'real', 1: 'fake'}
LABELS_MAP_INV = {'real': 0, 'fake': 1}

# --- Global Models (Load once) ---
EXTRACTOR_A = None
EXTRACTOR_B = None
EXTRACTOR_C = None
KNN_MODEL = None
SCALER = None
SELECTED_INDICES = None

def load_all_models():
    global EXTRACTOR_A, EXTRACTOR_B, EXTRACTOR_C, KNN_MODEL, SCALER, SELECTED_INDICES
    
    print("Loading all models for prediction...")
    
    # 1. Load Attn-Xception
    custom_objects = {"ModifiedBranch": ModifiedBranch, "MainBranch": MainBranch, "Attention": Attention}
    model_attn = tf.keras.models.load_model(os.path.join(MODEL_PATH, "best_model_attention.keras"), custom_objects=custom_objects)
    attn_output = model_attn.get_layer('attention_output').output
    flattened_attn = tf.keras.layers.Flatten()(attn_output)
    EXTRACTOR_C = tf.keras.Model(inputs=model_attn.input, outputs=flattened_attn)

    # 2. Load DenseNet121
    model_dense = tf.keras.models.load_model(os.path.join(MODEL_PATH, "best_model_densenet121.keras"))
    EXTRACTOR_A = tf.keras.Model(inputs=model_dense.input, outputs=model_dense.layers[-4].output)

    # 3. Load EfficientNetB0
    model_effnet = tf.keras.models.load_model(os.path.join(MODEL_PATH, "best_model_efficientnetb0.keras"))
    EXTRACTOR_B = tf.keras.Model(inputs=model_effnet.input, outputs=model_effnet.layers[-4].output)
    
    print("DL extractors loaded.")

    # 4. Load KNN, Scaler, and Features
    KNN_MODEL = joblib.load(os.path.join(MODEL_PATH, "knn_model.joblib"))
    SCALER = joblib.load(os.path.join(MODEL_PATH, "scaler.joblib"))
    SELECTED_INDICES = np.load(os.path.join(MODEL_PATH, "selected_features.npy"))
    
    print("KNN, Scaler, and Feature Indices loaded. Ready to predict.")

def predict_single_image(image_path):
    # Load and prep images (one for each preprocess type)
    img_inet = load_and_prep_image(image_path, IMG_DIM, 'imagenet')
    img_xcept = load_and_prep_image(image_path, IMG_DIM, 'xception')
    
    if img_inet is None or img_xcept is None:
        print(f"Error: Could not process {image_path}")
        return "Error", 0.0
        
    # Add batch dimension
    batch_x_imagenet = np.expand_dims(img_inet, axis=0)
    batch_x_xception = np.expand_dims(img_xcept, axis=0)
    
    # 1. Extract features from all 3 models
    features_A = EXTRACTOR_A.predict(batch_x_imagenet, verbose=0)
    features_B = EXTRACTOR_B.predict(batch_x_imagenet, verbose=0)
    features_C = EXTRACTOR_C.predict(batch_x_xception, verbose=0)
    
    # 2. Create "super-vector"
    features_stacked = np.concatenate([features_A, features_B, features_C], axis=1) # (1, 2665)
    
    # 3. Filter with selected features
    features_selected = features_stacked[:, SELECTED_INDICES] # (1, num_selected)
    
    # 4. Scale
    features_scaled = SCALER.transform(features_selected)
    
    # 5. Predict with KNN
    prediction_proba = KNN_MODEL.predict_proba(features_scaled)
    
    predicted_index = np.argmax(prediction_proba[0])
    predicted_label = LABELS_MAP[predicted_index]
    confidence = prediction_proba[0][predicted_index] * 100
    
    return predicted_label, confidence

def evaluate_folder(folder_path):
    print(f"Scanning folder: {folder_path} (as requested in data structure)")
    
    fake_paths = glob.glob(os.path.join(folder_path, 'fake', '*.png'))
    real_paths = glob.glob(os.path.join(folder_path, 'real', '*.png'))
    
    all_paths = fake_paths + real_paths
    true_labels_str = ['fake'] * len(fake_paths) + ['real'] * len(real_paths)
    true_labels_int = [LABELS_MAP_INV[l] for l in true_labels_str]
    
    if not all_paths:
        print("No .png images found in 'fake' or 'real' subdirectories.")
        return
        
    all_preds_int = []
    
    for i, path in enumerate(tqdm(all_paths, desc="Evaluating image folder")):
        pred_label, _ = predict_single_image(path)
        all_preds_int.append(LABELS_MAP_INV[pred_label])
            
    accuracy = accuracy_score(true_labels_int, all_preds_int)
    
    print("\n--- Evaluation Summary ---")
    print(f"Total Images: {len(all_paths)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("--------------------------")
    
    print("\nClassification Report:")
    print(classification_report(true_labels_int, all_preds_int, target_names=['real (0)', 'fake (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels_int, all_preds_int)
    print(cm)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
    plt.title('Confusion Matrix - Hybrid Model (Raw Images)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig("/kaggle/working/hybrid_confusion_matrix_raw_images.png")
    print("\nConfusion matrix saved to /kaggle/working/hybrid_confusion_matrix_raw_images.png")

def main():
    parser = argparse.ArgumentParser(description='Predict if an image is real or fake.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file OR a test folder (with fake/real subdirs).')
    args = parser.parse_args()
    
    # Load all models into memory
    load_all_models()
    
    if os.path.isfile(args.input_path):
        pred_label, confidence = predict_single_image(args.input_path)
        print("\n--- Prediction Result ---")
        print(f"       File: {os.path.basename(args.input_path)}")
        print(f"Prediction is: {pred_label.upper()}")
        print(f"  Confidence: {confidence:.2f}%")
        print("-------------------------")
        
    elif os.path.isdir(args.input_path):
        evaluate_folder(args.input_path)
    else:
        print(f"Error: Input path is not a valid file or directory: {args.input_path}")

if __name__ == "__main__":
    main()
