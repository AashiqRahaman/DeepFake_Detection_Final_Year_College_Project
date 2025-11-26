# for install conda open linux terminal or use wsl then run these---
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
# for upgrade conda 
```bash
bash Miniconda3-latest-Linux-x86_64.sh -u
```


# Deepfake-Video-Forgery-Detection 
## Getting Started
### Prerequisites
You need Python3.X and conda package manager to run this tool

## Installation
The follwing steps can be used to install the required packages :
1. Clone the repository 
```bash
git clone https://github.com/AashiqRahaman/DeepFake_Detection_Final_Year_College_Project
```
2. Inialize a conda environment with neccessary packages 
```bash
conda env create -f environment.yml
```
3. Activate conda enviroment 
```bash
conda activate DeepFakeDetection
```
Once the conda enviroment is activated, we can procees to training the model.

## Training
For training the model, the following command can be used 
```bash
python main.py --train {training path} --val {validation path} --epochs {final epoch} --batch {batch size} --steps {steps} 
```
### Sample Example for my case
```bash
# python main.py --train "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\train" `
#                --val "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\validation" `
#                --epochs 5 `
#                --batch 10 `
#                --steps 100
```

```bash
python main.py \
  --train "Assets/1000_videos/train" \
  --val "Assets/1000_videos/validation" \
  --epochs 15 \
  --batch 10 \
  --steps 100

```
``` bash
# python main.py --train "Assets\1000_videos\train" --val "Assets\1000_videos\validation" --epochs 15 --batch 10 --steps 100
```

Batch size = how many samples you train on before updating the model once.

Epochs = how many times the model goes through the whole dataset.




{training path} : Path to the training image set  
{validation path} : Path to the validation image set  
{final epochs} : Number of epochs used while training final fused model  
{batch size} : Batch Size used while training and validating  
{Steps} : Number of steps per epochs while training  




# Single image
```bash
python predict.py --model models/best_model.h5 --image Assets/test_img.png
````
# Folder of images
```bash
python predict.py --model models/best_model.h5 --folder Assets/test_images/
```




assets file str
```bash
1000_videos/
├── test/
│   ├── fake/
│   │   ├── 067_025_1.png
│   │   ├── 067_025_2.png
│   │   └── ...
│   └── real/
│       ├── 067_16.png
│       ├── 067_17.png
│       └── ...
├── train/
│   ├── fake/
│   │   ├── 128_896_3.png
│   │   ├── 128_896_4.png
│   │   └── ...
│   └── real/
│       ├── 129_2.png
│       ├── 129_3.png
│       └── ...
└── validation/
    ├── fake/
    │   ├── 000_003_0.png
    │   ├── 000_003_1.png
    │   └── ...
    └── real/
        ├── 000_0.png
        ├── 000_1.png
        └── ...
```

(DeepFakeDetection) ar_wsl@Aashiq:/mnt/c/Users/aashiq/Desktop/Git/MY_Git_Hub_AashiqRahaman/DeepFake_Detection_Final_Year_College_Project$ python main.py \
  --train "Assets/1000_videos/train" \
  --val "Assets/1000_videos/validation" \
  --epochs 15 \
  --batch 10 \
  --steps 100
2025-09-09 10:26:52.189984: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/cv2/../../lib64:
2025-09-09 10:26:52.190043: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Configuration
----------------------------------------------------------------------
Training Path : Assets/1000_videos/train
Validation Path : Assets/1000_videos/validation
Epochs while training the model : 15
Batch Size : 10
Steps per epochs : 100
----------------------------------------------------------------------
************TRAINING SOFT ATTENTION BASED DEEP FAKE DETECTION MODEL************
2025-09-09 10:26:53.093560: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/cv2/../../lib64:
2025-09-09 10:26:53.093607: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2025-09-09 10:26:53.093641: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (Aashiq): /proc/driver/nvidia/version does not exist
2025-09-09 10:26:53.093805: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
/home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/tensorflow/python/data/ops/dataset_ops.py:3703: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable.debug_mode()`.
  warnings.warn(
2025-09-09 10:26:54.181368: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2025-09-09 10:26:54.181774: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2295625000 Hz
Epoch 1/15
100/100 [==============================] - 442s 4s/step - loss: 1.1598 - accuracy: 0.4850 - val_loss: 0.7004 - val_accuracy: 0.4267
2025-09-09 10:34:21.860809: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:absl:Found untraced functions such as dense_layer_call_and_return_conditional_losses, dense_layer_call_fn, reshape_layer_call_and_return_conditional_losses, reshape_layer_call_fn, dropout_layer_call_and_return_conditional_losses while saving (showing 5 of 55). These functions will not be directly callable after loading.
/home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/tensorflow/python/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '
Epoch 2/15
100/100 [==============================] - 440s 4s/step - loss: 0.7461 - accuracy: 0.5260 - val_loss: 0.6797 - val_accuracy: 0.4733
WARNING:absl:Found untraced functions such as dense_layer_call_and_return_conditional_losses, dense_layer_call_fn, reshape_layer_call_and_return_conditional_losses, reshape_layer_call_fn, dropout_layer_call_and_return_conditional_losses while saving (showing 5 of 55). These functions will not be directly callable after loading.
/home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/tensorflow/python/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '
Epoch 3/15
100/100 [==============================] - 474s 5s/step - loss: 0.6537 - accuracy: 0.6210 - val_loss: 0.7280 - val_accuracy: 0.5200
WARNING:absl:Found untraced functions such as dense_layer_call_and_return_conditional_losses, dense_layer_call_fn, reshape_layer_call_and_return_conditional_losses, reshape_layer_call_fn, dropout_layer_call_and_return_conditional_losses while saving (showing 5 of 55). These functions will not be directly callable after loading.
/home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/tensorflow/python/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_conflayer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '
Epoch 6/15
100/100 [==============================] - 493s 5s/step - loss: 0.5500 - accuracy: 0.7230 - val_loss: 0.6132 - val_accuracy: 0.6933
Epoch 7/15
100/100 [==============================] - 457s 5s/step - loss: 0.4141 - accuracy: 0.8290 - val_loss: 0.6037 - val_accuracy: 0.8133
WARNING:absl:Found untraced functions such as dense_layer_call_and_return_conditional_losses, dense_layer_call_fn, reshape_layer_call_and_return_conditional_losses, reshape_layer_call_fn, dropout_layer_call_and_return_conditional_losses while saving (showing 5 of 55). These functions will not be directly callable after loading.
/home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/tensorflow/python/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '
Epoch 8/15
100/100 [==============================] - 425s 4s/step - loss: 0.4433 - accuracy: 0.7960 - val_loss: 0.6415 - val_accuracy: 0.6067
Epoch 9/15
100/100 [==============================] - 428s 4s/step - loss: 0.3051 - accuracy: 0.8820 - val_loss: 0.4577 - val_accuracy: 0.8467
WARNING:absl:Found untraced functions such as dense_layer_call_and_return_conditional_losses, dense_layer_call_fn, reshape_layer_call_and_return_conditional_losses, reshape_layer_call_fn, dropout_layer_call_and_return_conditional_losses while saving (showing 5 of 55). These functions will not be directly callable after loading.
/home/ar_wsl/miniconda3/envs/DeepFakeDetection/lib/python3.8/site-packages/tensorflow/python/keras/utils/generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '
Epoch 10/15
100/100 [==============================] - 417s 4s/step - loss: 0.3752 - accuracy: 0.8600 - val_loss: 0.5905 - val_accuracy: 0.8133
Epoch 11/15
100/100 [==============================] - 5843s 59s/step - loss: 0.2695 - accuracy: 0.9060 - val_loss: 0.5453 - val_accuracy: 0.8000
Epoch 12/15
100/100 [==============================] - 424s 4s/step - loss: 0.2579 - accuracy: 0.8900 - val_loss: 0.5833 - val_accuracy: 0.6133
Epoch 13/15
100/100 [==============================] - 422s 4s/step - loss: 0.2555 - accuracy: 0.8980 - val_loss: 0.5067 - val_accuracy: 0.6400
Epoch 14/15
100/100 [==============================] - 419s 4s/step - loss: 0.2651 - accuracy: 0.8900 - val_loss: 0.5556 - val_accuracy: 0.8333
Epoch 15/15
100/100 [==============================] - 445s 4s/step - loss: 0.2552 - accuracy: 0.9060 - val_loss: 0.5148 - val_accuracy: 0.5800