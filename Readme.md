This is all about my final year project in my Btech cource its just starting ... I hope I will contribute in it much.

python -m venv df
df\Scripts\activate


conda env create -f environment_win.yml


conda activate DeepFakeDetection
pip install opencv-python
pip install opencv-contrib-python
pip install tensorflow



python main.py --train "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\train" `
               --val "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\validation" `
               --epochs 20 `
               --batch 8 `
               --steps 100






df) PS C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project> python main.py --train "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\train" `
>>                --val "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\validation" `
>>                --epochs 20 `
>>                --batch 32 `
>>                --steps 100
2025-08-31 20:28:41.265771: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-31 20:28:45.369120: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Configuration
----------------------------------------------------------------------
Training Path : C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\train
Validation Path : C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\validation
Epochs while training the model : 20
Batch Size : 32
Steps per epochs : 100
----------------------------------------------------------------------
************TRAINING SOFT ATTENTION BASED DEEP FAKE DETECTION MODEL************
2025-08-31 20:28:46.284506: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5
91884032/91884032 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step
WARNING:tensorflow:From C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\df\Lib\site-packages\keras\src\backend\tensorflow\core.py:232: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\df\Lib\site-packages\tensorflow\python\data\ops\structured_function.py:258: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`.
  warnings.warn(
Epoch 1/20
  3/100 ━━━━━━━━━━━━━━━━━━━━ 2:54:32 108s/step - accuracy: 0.4514 - loss: 0.0000e+00Traceback (most recent call last):