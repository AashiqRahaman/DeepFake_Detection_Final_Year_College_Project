
import tensorflow as tf
import numpy as np
import cv2
import argparse
import os  # <-- Added for file/folder operations

# --- CRITICAL: MUST IMPORT YOUR CUSTOM LAYERS ---
from model import ModifiedBranch, MainBranch, Attention

# --- CRITICAL: MUST IMPORT THE SAME PREPROCESSING FUNCTION ---
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
    
    if image_batch is None:
        return

    # --- Predict ---
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
    print(f"(Raw Probabilities: Fake={prediction[0][0]:.4f}, Real={prediction[0][1]:.4f})")

def evaluate_folder(model, folder_path):
    """
    Recursively evaluates all images in a given folder.
    Assumes subfolders are named 'fake' and 'real' (like your data structure).
    """
    print(f"Scanning folder: {folder_path}\nThis may take a while...")
    
    total_files = 0
    correct_predictions = 0
    
    # Recursively walk the directory
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files

            # Get the true label from the parent folder name
            true_label = os.path.basename(root).lower()

            # Skip files that are not in a 'fake' or 'real' subfolder
            if true_label not in LABELS:
                continue

            image_path = os.path.join(root, filename)
            
            # Load and prep the image
            image_batch = load_and_prep_image(image_path)
            if image_batch is None:
                continue # Skip images that failed to load
            
            total_files += 1

            # Predict (use verbose=0 to silence the "1/1" print for every file)
            prediction = model.predict(image_batch, verbose=0)
            predicted_index = np.argmax(prediction[0])
            predicted_label = LABELS[predicted_index]

            # Compare prediction to the true label
            if predicted_label == true_label:
                correct_predictions += 1
                result = "CORRECT"
            else:
                result = "WRONG"
            
            print(f"  > File: {filename} | True Label: {true_label} | Prediction: {predicted_label}  [{result}]")

    # After checking all files, print the final summary
    if total_files > 0:
        overall_accuracy = (correct_predictions / total_files) * 100
        print("\n--- Evaluation Summary ---")
        print(f"Total Images Processed: {total_files}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"    Wrong Predictions: {total_files - correct_predictions}")
        print(f"   OVERALL ACCURACY: {overall_accuracy:.2f}%")
        print("--------------------------")
    else:
        print("\nNo valid image files found in 'fake' or 'real' subdirectories.")

def main():
    parser = argparse.ArgumentParser(description='Predict if an image is real or fake. Works on a single file or a whole directory.')
    
    # Updated argument to accept either a file or folder
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image file OR a folder (like .../test/)')
    
    # Changed default path to the one we confirmed works
    parser.add_argument('--model_path', type=str, default='models/best_model.keras', help='Path to the saved .keras model file.')
    
    args = parser.parse_args()

    # --- Step 1: Load the Model with Custom Objects ---
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    custom_objects = {
        "ModifiedBranch": ModifiedBranch,
        "MainBranch": MainBranch,
        "Attention": Attention
    }
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")

    # --- Step 2: Check if input path is a file or directory ---
    input_path = args.input_path
    
    if os.path.isfile(input_path):
        # It's a single file. Just predict it.
        predict_single_image(model, input_path)
        
    elif os.path.isdir(input_path):
        # It's a folder. Evaluate everything inside.
        evaluate_folder(model, input_path)
        
    else:
        print(f"Error: Input path is not a valid file or directory: {input_path}")


if __name__ == "__main__":
    main()


# for kaggle
# ls /kaggle/input/1000-videos-split/1000_videos/train/fake | head -n 5
# python predict.py --input_path "/kaggle/input/1000-videos-split/1000_videos/train/"


# for local
# python predict.py --input_path "../test/"
# python predict.py --input_path "../Assets/1000_videos/test/fake/"
# python predict.py --input_path "../Assets/1000_videos/validation/"



























# for random necessary

# ls ../../Assets/1000_videos/train/fake | head -n 5   
# python predict.py --image "../../Assets/1000_videos/test/fake/067_025_1.png"
# python predict.py --image "../../Assets/1000_videos/test/fake/067_025_1.png"



# python predict.py --image "../../Assets/1000_videos/test/real/067_16.png"
# python predict.py --image "../../Assets/1000_videos/test/real/128_22.png"
# python predict.py --image "Assets/1000_videos/test/real/129_1.png"


# python predict.py --image "Assets/1000_videos/train/real/129_2.png"
# python predict.py --image "Assets/1000_videos/train/real/129_7.png"
 
# python predict.py --image "../../1000_videos/validation/fake/your_test_image_name.jpg"





# import tensorflow as tf
# import numpy as np
# import cv2
# import argparse

# # --- CRITICAL: MUST IMPORT YOUR CUSTOM LAYERS ---
# # We need to import these so Keras knows what they are when loading the model
# from model import ModifiedBranch, MainBranch, Attention

# # --- CRITICAL: MUST IMPORT THE SAME PREPROCESSING FUNCTION ---
# from tensorflow.keras.applications.xception import preprocess_input

# def load_and_prep_image(image_path, target_size=(299, 299)):
#     """
#     Loads, resizes, and preprocesses a single image.
#     """
#     # 1. Load the image using OpenCV
#     img = cv2.imread(image_path)
    
#     # 2. Resize to the model's expected input size
#     img = cv2.resize(img, target_size)
    
#     # 3. Apply the Xception-specific preprocessing (scales pixels to [-1, 1])
#     img_preprocessed = preprocess_input(img)
    
#     # 4. Expand dimensions to create a "batch" of 1
#     # Model expects shape (batch_size, height, width, channels)
#     # So we change (299, 299, 3) to (1, 299, 299, 3)
#     return np.expand_dims(img_preprocessed, axis=0)

# def main():
#     parser = argparse.ArgumentParser(description='Predict if an image is real or fake.')
#     parser.add_argument('--image', type=str, required=True, help='Path to the input image file.')
#     parser.add_argument('--model_path', type=str, default='best_model.keras', help='Path to the saved .keras model file.')
#     args = parser.parse_args()

#     # Define the labels exactly as they were in utils.py
#     labels = ['fake', 'real']

#     # --- Step 1: Load the Model with Custom Objects ---
#     print(f"Loading model from {args.model_path}...")
#     custom_objects = {
#         "ModifiedBranch": ModifiedBranch,
#         "MainBranch": MainBranch,
#         "Attention": Attention
#     }
#     model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
#     print("Model loaded successfully.")

#     # --- Step 2 & 3: Load and Preprocess the Image ---
#     print(f"Loading and preprocessing image: {args.image}...")
#     image_batch = load_and_prep_image(args.image)

#     # --- Step 4: Predict ---
#     print("Predicting...")
#     prediction = model.predict(image_batch)
    
#     # The output 'prediction' is a 2D array, e.g., [[0.98, 0.02]]
#     # Get the index of the highest probability
#     predicted_index = np.argmax(prediction[0])
    
#     # Get the human-readable label and the confidence score
#     predicted_label = labels[predicted_index]
#     confidence = prediction[0][predicted_index] * 100

#     print("\n--- Prediction Result ---")
#     print(f"This image is: {predicted_label.upper()}")
#     print(f"Confidence: {confidence:.2f}%")
#     print("-------------------------")
#     print(f"(Raw probabilities: Fake={prediction[0][0]:.4f}, Real={prediction[0][1]:.4f})")


# if __name__ == "__main__":
#     main()
    
    
