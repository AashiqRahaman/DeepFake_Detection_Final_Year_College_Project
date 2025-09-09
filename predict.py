import os
import argparse
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.xception import preprocess_input

# Custom layers need to be registered for the model to load correctly
from model import Attention, MainBranch, ModifiedBranch

def get_input(path, target_size=(299, 299)):
    """
    Loads an image from the given path, resizes it, converts it to a numpy array,
    and preprocesses it for the model.
    """
    try:
        img = load_img(path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

def predict_single_image(model, image_path, labels):
    """
    Predicts the class of a single input image.
    
    Args:
        model (keras.Model): The loaded deep fake detection model.
        image_path (str): The path to the image file.
        labels (list): A list of class labels (e.g., ['fake', 'real']).

    Returns:
        tuple: A tuple containing the prediction label and confidence.
    """
    image_data = get_input(image_path)
    if image_data is None:
        return "Error", 0.0
    
    prediction = model.predict(image_data)
    confidence = np.max(prediction)
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]
    
    return predicted_label, confidence

def predict_folder(model, folder_path, labels):
    """
    Predicts the class for all images in a folder and calculates accuracy.
    
    Args:
        model (keras.Model): The loaded deep fake detection model.
        folder_path (str): The path to the folder containing images.
        labels (list): A list of class labels (e.g., ['fake', 'real']).
    """
    all_files = []
    for ext in ['png', 'jpg', 'jpeg']:
        all_files.extend(glob.glob(os.path.join(folder_path, f'**/*.{ext}'), recursive=True))

    if not all_files:
        print(f"No images found in the specified folder: {folder_path}")
        return

    correct_predictions = 0
    total_images = len(all_files)
    
    print(f"\nProcessing {total_images} images from {folder_path}...")
    for i, file_path in enumerate(all_files):
        # Extract true label from the parent directory name
        true_label = os.path.basename(os.path.dirname(file_path))
        
        predicted_label, confidence = predict_single_image(model, file_path, labels)
        
        is_correct = "Correct" if predicted_label.lower() == true_label.lower() else "Incorrect"
        if is_correct == "Correct":
            correct_predictions += 1
            
        print(f"[{i+1}/{total_images}] Image: {os.path.basename(file_path)} -> Prediction: {predicted_label} (Confidence: {confidence:.2f}) | Ground Truth: {true_label} | Result: {is_correct}")

    # Calculate and print overall accuracy
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print("\n-----------------------------------------------------")
        print(f"Prediction complete. Total Images: {total_images}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print("-----------------------------------------------------")
    else:
        print("No images were processed to calculate accuracy.")


def main():
    parser = argparse.ArgumentParser(description='Deep Fake Detection Prediction Script')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.h5)')
    parser.add_argument('--image', type=str, help='Path to a single image for prediction')
    parser.add_argument('--folder', type=str, help='Path to a folder of images for batch prediction and accuracy calculation')

    args = parser.parse_args()
    
    # Check if a model path is provided
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # Load the model with custom objects
    # This is crucial because your model uses custom Keras Layers (Attention, etc.)
    try:
        print("Loading the model...")
        model = load_model(args.model, custom_objects={'Attention': Attention, 'MainBranch': MainBranch, 'ModifiedBranch': ModifiedBranch})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define your class labels
    labels = ['fake', 'real']

    # Perform prediction based on the provided arguments
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
        predicted_label, confidence = predict_single_image(model, args.image, labels)
        print("\n-----------------------------------------------------")
        print(f"Prediction for {os.path.basename(args.image)}:")
        print(f"Predicted Class: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")
        print("-----------------------------------------------------")
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found at {args.folder}")
            return
        predict_folder(model, args.folder, labels)
    else:
        print("Please provide either --image or --folder argument.")
        parser.print_help()

if __name__ == '__main__':
    # Make sure to set run_functions_eagerly to True if your original model was built with it
    # This might be needed for the custom layers to work correctly.
    tf.config.run_functions_eagerly(True)
    main()



# python predict.py --model models/best_model.h5 --image Assets/1000_videos/test/fake/067_025_1.png