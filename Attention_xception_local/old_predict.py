import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
# Import the function needed for normalization, just like in your utils.py
from tensorflow.keras.applications.xception import preprocess_input

# --- Constants ---
# These labels MUST match the order from your utils.py file
LABELS = ['fake', 'real']
# The image dimensions MUST match the 'dim' used during training
IMG_SIZE = (299, 299)

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for prediction,
    matching the steps used in the training generator (utils.py).
    """
    # 1. Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # 2. Resize to the same dimensions used in training
    img = cv2.resize(img, IMG_SIZE)
    
    # 3. CRITICAL: Normalize the image just like the training data
    img_normalized = preprocess_input(img)
    
    # 4. Add a "batch" dimension
    # The model expects a batch (shape: [1, 299, 299, 3])
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='DeepFake Prediction Script')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image for prediction.')
    parser.add_argument('--model_path', type=str, default='models', help='Path to the saved model directory.')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path not found at {args.image}")
        return

    if not os.path.isdir(args.model_path):
        print(f"Error: Model directory not found at {args.model_path}. Did you train the model first?")
        return

    # --- 1. Load Model ---
    print(f"Loading trained model from '{args.model_path}'...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # --- 2. Preprocess Image ---
    print(f"Processing image: {args.image}...")
    processed_image = preprocess_image(args.image)

    # --- 3. Run Prediction ---
    print("Running prediction...")
    prediction = model.predict(processed_image)
    
    # --- 4. Interpret Results ---
    # Prediction will be like [[prob_fake, prob_real]]
    predicted_index = np.argmax(prediction[0])
    predicted_label = LABELS[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    print("\n--- Prediction Result ---")
    print(f"This image is: {predicted_label.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Raw probabilities: Fake={prediction[0][0]:.4f}, Real={prediction[0][1]:.4f})")

if __name__ == "__main__":
    main()
    
    
# python predict.py --image "Assets/1000_videos/test/fake/067_025_1.png"
# python predict.py --image "Assets/1000_videos/test/fake/067_025_1.png"



# python predict.py --image "Assets/1000_videos/test/real/067_16.png"
# python predict.py --image "Assets/1000_videos/test/real/128_22.png"
# python predict.py --image "Assets/1000_videos/test/real/129_1.png"


# python predict.py --image "Assets/1000_videos/train/real/129_2.png"
# python predict.py --image "Assets/1000_videos/train/real/129_7.png"
