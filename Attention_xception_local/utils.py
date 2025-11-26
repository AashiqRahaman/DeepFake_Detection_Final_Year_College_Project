import os
import cv2
import glob
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input

def get_input(path):
    im = cv2.imread(path)
    return(im)

def get_files(path, ext):
    files = []
    label_files= []
    for x in os.walk(path):
        for y in glob.glob(os.path.join(x[0], '*.{}'.format(ext))):
            files.append(y)
    label_files = ['fake', 'real']
    return files, label_files

# def get_output(path, label_file):
#     img_id = path.split('/')[-1].split('_')[0]
#     laba = []
#     for label in label_file:
#       if label == img_id:
#         laba.append(1)
#       else:
#         laba.append(0)
#     return laba
def get_output(path, label_file):
    # This function now gets the label from the parent directory name.
    # e.g., for path ".../train/fake/file.png", img_id will be "fake"
    img_id = path.split('/')[-2] 
    laba = []
    for label in label_file:
      if label == img_id:
        laba.append(1)
      else:
        laba.append(0)
    return laba

def image_generator(files, label_files, batch_size, resize=None):
    while True:
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_x = []
          batch_y = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              output = get_output(input_path, label_files)
              if resize is not None:
                input = cv2.resize(input, resize)
              input = preprocess_input(input) # <-- THIS LINE IS ADDED TO BETTERMENT
              batch_x.append(input)
              batch_y.append(output)

          batch_x = np.array(batch_x)
          batch_y = np.array(batch_y)
          yield batch_x, batch_y