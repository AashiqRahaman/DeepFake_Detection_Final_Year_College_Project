import os
import glob
import numpy as np
import cv2
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# --- Data Loading ---
def get_files_from_structure(data_path):
    """
    Gets file paths and labels from the train/val/test structure.
    Label: 0 = real, 1 = fake
    """
    fake_paths = glob.glob(os.path.join(data_path, 'fake', '*.png'))
    real_paths = glob.glob(os.path.join(data_path, 'real', '*.png'))
    
    file_paths = fake_paths + real_paths
    labels = [1] * len(fake_paths) + [0] * len(real_paths)
    
    file_paths, labels = shuffle(file_paths, labels, random_state=42)
    
    print(f"Found {len(file_paths)} images in {data_path}: {len(fake_paths)} fake, {len(real_paths)} real.")
    return file_paths, labels

# --- Image Preprocessing ---
def load_and_prep_image(path, target_size, preprocess_type='xception'):
    """
    Loads and preprocesses a single image.
    preprocess_type can be 'xception' or 'imagenet'
    """
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            return None
        
        img_resized = cv2.resize(img, target_size)
        
        if preprocess_type == 'xception':
            img_preprocessed = xception_preprocess(img_resized)
        else: # 'imagenet'
            img_preprocessed = imagenet_preprocess(img_resized)
            
        return img_preprocessed
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

# --- Keras Data Generators ---
def image_generator(file_paths, labels, batch_size, target_size=(299, 299), preprocess_type='xception'):
    """
    Keras generator for training models.
    """
    num_samples = len(file_paths)
    while True:
        file_paths, labels = shuffle(file_paths, labels)
        
        for offset in range(0, num_samples, batch_size):
            batch_paths = file_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            
            batch_x = []
            batch_y = []
            
            for i, input_path in enumerate(batch_paths):
                img = load_and_prep_image(input_path, target_size, preprocess_type)
                if img is not None:
                    batch_x.append(img)
                    batch_y.append(batch_labels[i])
            
            if batch_x:
                batch_x = np.array(batch_x)
                batch_y = to_categorical(np.array(batch_y), num_classes=2)
                yield batch_x, batch_y
