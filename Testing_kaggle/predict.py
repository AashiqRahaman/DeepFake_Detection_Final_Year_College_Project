
import tensorflow as tf
import numpy as np
import cv2
import argparse

# --- CRITICAL: MUST IMPORT YOUR CUSTOM LAYERS ---
# We need to import these so Keras knows what they are when loading the model
from model import ModifiedBranch, MainBranch, Attention

# --- CRITICAL: MUST IMPORT THE SAME PREPROCESSING FUNCTION ---
from tensorflow.keras.applications.xception import preprocess_input

def load_and_prep_image(image_path, target_size=(299, 299)):
    """
    Loads, resizes, and preprocesses a single image.
    """
    # 1. Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # 2. Resize to the model's expected input size
    img = cv2.resize(img, target_size)
    
    # 3. Apply the Xception-specific preprocessing (scales pixels to [-1, 1])
    img_preprocessed = preprocess_input(img)
    
    # 4. Expand dimensions to create a "batch" of 1
    # Model expects shape (batch_size, height, width, channels)
    # So we change (299, 299, 3) to (1, 299, 299, 3)
    return np.expand_dims(img_preprocessed, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Predict if an image is real or fake.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_path', type=str, default='best_model.keras', help='Path to the saved .keras model file.')
    args = parser.parse_args()

    # Define the labels exactly as they were in utils.py
    labels = ['fake', 'real']

    # --- Step 1: Load the Model with Custom Objects ---
    print(f"Loading model from {args.model_path}...")
    custom_objects = {
        "ModifiedBranch": ModifiedBranch,
        "MainBranch": MainBranch,
        "Attention": Attention
    }
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")

    # --- Step 2 & 3: Load and Preprocess the Image ---
    print(f"Loading and preprocessing image: {args.image}...")
    image_batch = load_and_prep_image(args.image)

    # --- Step 4: Predict ---
    print("Predicting...")
    prediction = model.predict(image_batch)
    
    # The output 'prediction' is a 2D array, e.g., [[0.98, 0.02]]
    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])
    
    # Get the human-readable label and the confidence score
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    print("\n--- Prediction Result ---")
    print(f"This image is: {predicted_label.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("-------------------------")
    print(f"(Raw probabilities: Fake={prediction[0][0]:.4f}, Real={prediction[0][1]:.4f})")


if __name__ == "__main__":
    main()
    
    
# ls ../../Assets/1000_videos/train/fake | head -n 5   
# python predict.py --image "../../Assets/1000_videos/test/fake/067_025_1.png"
# python predict.py --image "../../Assets/1000_videos/test/fake/067_025_1.png"



# python predict.py --image "../../Assets/1000_videos/test/real/067_16.png"
# python predict.py --image "../../Assets/1000_videos/test/real/128_22.png"
# python predict.py --image "Assets/1000_videos/test/real/129_1.png"


# python predict.py --image "Assets/1000_videos/train/real/129_2.png"
# python predict.py --image "Assets/1000_videos/train/real/129_7.png"
 
# python predict.py --image "../../1000_videos/validation/fake/your_test_image_name.jpg"