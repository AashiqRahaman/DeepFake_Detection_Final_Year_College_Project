## How to Use
---

run (from project root)

test whole folder (expects fake/ and real/ inside the folder):
```bash
python Testing/test.py --model Models/my_deepfake_detector.keras --test Assets/1000_videos/test
```

Example output:
```bash
img1.png → Predicted: fake, Prob: {'fake': 0.92, 'real': 0.08}
img2.png → Predicted: real, Prob: {'fake': 0.12, 'real': 0.88}

Overall Test Accuracy: 85.71% (12/14)
```
2025-09-09 02:07:45.826239: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1757383665.847112      93 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1757383665.854565      93 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Configuration
----------------------------------------------------------------------
Training Path : /kaggle/input/1000-videos-split/1000_videos/train
Validation Path : /kaggle/input/1000-videos-split/1000_videos/validation
Epochs while training the model : 30
Batch Size : 10
Steps per epochs : 100
----------------------------------------------------------------------
************TRAINING SOFT ATTENTION BASED DEEP FAKE DETECTION MODEL************
I0000 00:00:1757383705.682928      93 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13942 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1757383705.683759      93 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13942 MB memory:  -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5
91884032/91884032 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py:258: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`.
  warnings.warn(
Epoch 1/30
I0000 00:00:1757383709.975231      93 cuda_dnn.cc:529] Loaded cuDNN version 90300
100/100 ━━━━━━━━━━━━━━━━━━━━ 121s 1s/step - accuracy: 0.5567 - loss: 1.2042 - val_accuracy: 0.4800 - val_loss: 0.6933
Epoch 2/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 116s 1s/step - accuracy: 0.5340 - loss: 0.7468 - val_accuracy: 0.5600 - val_loss: 0.6916
Epoch 3/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 115s 1s/step - accuracy: 0.5887 - loss: 0.6898 - val_accuracy: 0.6333 - val_loss: 0.6604
Epoch 4/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 115s 1s/step - accuracy: 0.6298 - loss: 0.6474 - val_accuracy: 0.7000 - val_loss: 0.6618
Epoch 5/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.7106 - loss: 0.5926 - val_accuracy: 0.5800 - val_loss: 0.6445
Epoch 6/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.7203 - loss: 0.5433 - val_accuracy: 0.6333 - val_loss: 0.6288
Epoch 7/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.7590 - loss: 0.5178 - val_accuracy: 0.5467 - val_loss: 0.6445
Epoch 8/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.8263 - loss: 0.3918 - val_accuracy: 0.5667 - val_loss: 0.6016
Epoch 9/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.8287 - loss: 0.4161 - val_accuracy: 0.6067 - val_loss: 0.6427
Epoch 10/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.8464 - loss: 0.3790 - val_accuracy: 0.5467 - val_loss: 0.6822
Epoch 11/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.8478 - loss: 0.3413 - val_accuracy: 0.7867 - val_loss: 0.6205
Epoch 12/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 112s 1s/step - accuracy: 0.8532 - loss: 0.3234 - val_accuracy: 0.5867 - val_loss: 0.6121
Epoch 13/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.9006 - loss: 0.2896 - val_accuracy: 0.7133 - val_loss: 0.6040
Epoch 14/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 115s 1s/step - accuracy: 0.8981 - loss: 0.2439 - val_accuracy: 0.8267 - val_loss: 0.5127
Epoch 15/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 112s 1s/step - accuracy: 0.9151 - loss: 0.2312 - val_accuracy: 0.4667 - val_loss: 0.6339
Epoch 16/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 112s 1s/step - accuracy: 0.9099 - loss: 0.2543 - val_accuracy: 0.7533 - val_loss: 0.5831
Epoch 17/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9276 - loss: 0.2123 - val_accuracy: 0.8400 - val_loss: 0.4950
Epoch 18/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.8823 - loss: 0.3117 - val_accuracy: 0.7667 - val_loss: 0.5627
Epoch 19/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.9236 - loss: 0.2210 - val_accuracy: 0.7600 - val_loss: 0.6179
Epoch 20/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9227 - loss: 0.2132 - val_accuracy: 0.7400 - val_loss: 0.5811
Epoch 21/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.9023 - loss: 0.2547 - val_accuracy: 0.8067 - val_loss: 0.5508
Epoch 22/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9064 - loss: 0.2569 - val_accuracy: 0.6267 - val_loss: 0.5865
Epoch 23/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 115s 1s/step - accuracy: 0.9089 - loss: 0.2267 - val_accuracy: 0.8667 - val_loss: 0.3538
Epoch 24/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.9373 - loss: 0.2351 - val_accuracy: 0.7467 - val_loss: 0.5288
Epoch 25/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9314 - loss: 0.1594 - val_accuracy: 0.7533 - val_loss: 0.5586
Epoch 26/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9306 - loss: 0.2023 - val_accuracy: 0.8267 - val_loss: 0.4859
Epoch 27/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.9474 - loss: 0.1589 - val_accuracy: 0.5733 - val_loss: 0.5715
Epoch 28/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9120 - loss: 0.2045 - val_accuracy: 0.6133 - val_loss: 0.4886
Epoch 29/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 114s 1s/step - accuracy: 0.9297 - loss: 0.1468 - val_accuracy: 0.9333 - val_loss: 0.4097
Epoch 30/30
100/100 ━━━━━━━━━━━━━━━━━━━━ 113s 1s/step - accuracy: 0.9380 - loss: 0.2045 - val_accuracy: 0.7533 - val_loss: 0.5503