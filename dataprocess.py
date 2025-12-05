import os
import shutil
import random
from tqdm import tqdm


# --- CONFIGURATION FOR SPLITTING ---
SOURCE_DATA_PATH = "FFPP/"  # <--- UPDATE THIS PATH this is the original preprocessed data set   https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames
DEST_DATA_PATH = "FFPP_processed_Aashiq/" 

def split_dataset():
    if os.path.exists(DEST_DATA_PATH):
        print("Dataset already processed. Skipping split.")
        return

    print("Starting data split...")
    classes = ['fake', 'real']
    
    for class_name in classes:
        # Handle 'Deepfakes' subfolder if it exists (common in your structure)
        class_dir = os.path.join(SOURCE_DATA_PATH, class_name)
        if class_name == 'fake' and os.path.exists(os.path.join(class_dir, 'Deepfakes')):
            class_dir = os.path.join(class_dir, 'Deepfakes')
            
        if not os.path.exists(class_dir):
            print(f"Error: {class_dir} not found.")
            continue

        # Get list of VIDEO folders
        video_folders = [f for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]
        random.shuffle(video_folders)
        
        # 80% Train, 10% Val, 10% Test
        train_count = int(len(video_folders) * 0.8)
        val_count = int(len(video_folders) * 0.1)
        
        splits = {
            'train': video_folders[:train_count],
            'val': video_folders[train_count:train_count + val_count],
            'test': video_folders[train_count + val_count:]
        }
        
        print(f"Processing {class_name}: {len(video_folders)} videos found.")

        for split_type, folders in splits.items():
            save_dir = os.path.join(DEST_DATA_PATH, split_type, class_name)
            os.makedirs(save_dir, exist_ok=True)
            
            for folder in tqdm(folders, desc=f"Copying to {split_type}/{class_name}"):
                src = os.path.join(class_dir, folder)
                dst = os.path.join(save_dir, folder)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)

    print(f"\n✅ Data ready at: {DEST_DATA_PATH}")

split_dataset()