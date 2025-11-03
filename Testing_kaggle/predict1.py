
import tensorflow as tf
import numpy as np
import cv2
import argparse
import os

# --- Import your custom layers and preprocessing function ---
from model import ModifiedBranch, MainBranch, Attention
from tensorflow.keras.applications.xception import preprocess_input

# Define the labels list globally
LABELS = ['fake', 'real']

def load_and_prep_image(image_path, target_size=(299, 299)):
    """
    Loads, resizes, and preprocesses a single image.
    Returns None if the image cannot be read.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None
            
        img = cv2.resize(img, target_size)
        img_preprocessed = preprocess_input(img)
        return np.expand_dims(img_preprocessed, axis=0)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_single_image(model, image_path):
    """
    Loads a single image, predicts it, and prints the result.
    """
    image_batch = load_and_prep_image(image_path)
    if image_batch is None: return

    print("Predicting...")
    prediction = model.predict(image_batch)
    
    predicted_index = np.argmax(prediction[0])
    predicted_label = LABELS[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    print("\n--- Prediction Result ---")
    print(f"       File: {os.path.basename(image_path)}")
    print(f"Prediction is: {predicted_label.upper()}")
    print(f"  Confidence: {confidence:.2f}%")
    print("-------------------------")

def evaluate_folder(model, folder_path):
    """
    Recursively evaluates all images in a given folder.
    """
    print(f"Scanning folder: {folder_path}\nThis may take a while...")
    total_files = 0
    correct_predictions = 0
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            true_label = os.path.basename(root).lower()
            if true_label not in LABELS: continue

            image_path = os.path.join(root, filename)
            image_batch = load_and_prep_image(image_path)
            if image_batch is None: continue
            
            total_files += 1
            prediction = model.predict(image_batch, verbose=0)
            predicted_index = np.argmax(prediction[0])
            predicted_label = LABELS[predicted_index]

            # --- MODIFICATION IS HERE ---
            confidence = prediction[0][predicted_index] * 100

            if predicted_label == true_label:
                correct_predictions += 1
                result = "CORRECT"
            else:
                result = "WRONG"
            
            # --- AND HERE ---
            print(f"  > File: {filename} | True: {true_label} | Predicted: {predicted_label} ({confidence:.2f}%)  [{result}]")

    if total_files > 0:
        accuracy = (correct_predictions / total_files) * 100
        print("\n--- Evaluation Summary ---")
        print(f"Total Images: {total_files}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"OVERALL ACCURACY: {accuracy:.2f}%")
        print("--------------------------")
    else:
        print("\nNo valid image files found in 'fake' or 'real' subdirectories.")

def main():
    parser = argparse.ArgumentParser(description='Predict if an image is real or fake.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file OR a folder.')
    parser.add_argument('--model_path', type=str, default='models/best_model.keras', help='Path to the saved model file.')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    custom_objects = {"ModifiedBranch": ModifiedBranch, "MainBranch": MainBranch, "Attention": Attention}
    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    print("Model loaded.")

    if os.path.isfile(args.input_path):
        predict_single_image(model, args.input_path)
    elif os.path.isdir(args.input_path):
        evaluate_folder(model, args.input_path)
    else:
        print(f"Error: Input path is not a valid file or directory: {args.input_path}")

if __name__ == "__main__":
    main()
    
    
    
    
# python predict1.py --input_path "../test/"