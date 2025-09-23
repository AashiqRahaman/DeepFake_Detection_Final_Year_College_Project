## How to Use
---

run (from project root)
```bash
pip install tensorflow
pip install opencv-python
pip install numpy
```
` cd Testing_kaggle `
`

!python main.py \
    --train "/kaggle/input/1000-videos-split/1000_videos/train" \
    --val "/kaggle/input/1000-videos-split/1000_videos/validation" \
    --epochs 30 \
    --batch 16 \
    --steps 727


11,630 (training images) / 16 (new batch size) = 726.8

We'll round this up to 727 steps.







python predict.py --image "../Assets/1000_videos/test/fake/067_025_1.png"

python predict.py --image "../Assets/1000_videos/test/fake/067_025_1.png" --model_path "models/best_model.keras"

python predict.py --image "../Assets/1000_videos/test/real/067_16.png" --model_path "models/best_model.keras"

python predict.py --image "../Assets/1000_videos/train/fake/128_896_3.png" --model_path "models/best_model.keras"
python predict.py --image "../Assets/1000_videos/train/fake/128_896_3.png" --model_path "models/best_model.keras"
python predict.py --image "../Assets/1000_videos/train/fake/128_896_3.png" --model_path "models/best_model.keras"


python predict.py --image "../Assets/1000_videos/train/real/129_3.png" --model_path "models/best_model.keras"
python predict.py --image "../Assets/1000_videos/train/real/132_4.png" --model_path "models/best_model.keras"
python predict.py --image "../Assets/1000_videos/train/real/133_5.png" --model_path "models/best_model.keras"


```bash
pip uninstall tensorflow
pip uninstall opencv-python
pip uninstall numpy
```










# Example output of Traning:

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





# python predict.py --input_path "../test/"
# python predict.py --input_path "../Assets/1000_videos/test/fake/"
# python predict.py --input_path "../Assets/1000_videos/test/real/128_22.png"


# Example output of Traning 2:

2025-09-20 16:57:53.645041: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1758387473.665655     191 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1758387473.671900     191 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Configuration
----------------------------------------------------------------------
Training Path : /kaggle/input/1000-videos-split/1000_videos/train
Validation Path : /kaggle/input/1000-videos-split/1000_videos/validation
Epochs while training the model : 3
Batch Size : 16
Steps per epochs : 727
----------------------------------------------------------------------
************TRAINING SOFT ATTENTION BASED DEEP FAKE DETECTION MODEL************
I0000 00:00:1758387483.552336     191 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13942 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1758387483.552998     191 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13942 MB memory:  -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5
/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py:258: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`.
  warnings.warn(
Epoch 1/3
I0000 00:00:1758387485.854022     191 cuda_dnn.cc:529] Loaded cuDNN version 90300
727/727 ━━━━━━━━━━━━━━━━━━━━ 893s 1s/step - accuracy: 0.6642 - loss: 0.6937 - val_accuracy: 0.8167 - val_loss: 0.6108
Epoch 2/3
727/727 ━━━━━━━━━━━━━━━━━━━━ 888s 1s/step - accuracy: 0.9304 - loss: 0.1895 - val_accuracy: 0.7625 - val_loss: 0.5472
Epoch 3/3
727/727 ━━━━━━━━━━━━━━━━━━━━ 886s 1s/step - accuracy: 0.9526 - loss: 0.1294 - val_accuracy: 0.8375 - val_loss: 0.6167








# Example output of Traning 3:
2025-09-20 18:54:25.607951: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1758394465.630530      92 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1758394465.637201      92 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Configuration
----------------------------------------------------------------------
Training Path : /kaggle/input/1000-videos-split/1000_videos/train
Validation Path : /kaggle/input/1000-videos-split/1000_videos/validation
Epochs while training the model : 60
Batch Size : 16
Steps per epochs : 727
----------------------------------------------------------------------
************TRAINING SOFT ATTENTION BASED DEEP FAKE DETECTION MODEL************
I0000 00:00:1758394483.441563      92 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13942 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
I0000 00:00:1758394483.442417      92 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13942 MB memory:  -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5
91884032/91884032 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py:258: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`.
  warnings.warn(
Epoch 1/60
I0000 00:00:1758394487.760853      92 cuda_dnn.cc:529] Loaded cuDNN version 90300
727/727 ━━━━━━━━━━━━━━━━━━━━ 971s 1s/step - accuracy: 0.5722 - loss: 0.9721 - val_accuracy: 0.7583 - val_loss: 0.6124
Epoch 2/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 965s 1s/step - accuracy: 0.8843 - loss: 0.2975 - val_accuracy: 0.8708 - val_loss: 0.5412
Epoch 3/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 970s 1s/step - accuracy: 0.9501 - loss: 0.1472 - val_accuracy: 0.8958 - val_loss: 0.2998
Epoch 4/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 965s 1s/step - accuracy: 0.9643 - loss: 0.1064 - val_accuracy: 0.8000 - val_loss: 0.4330
Epoch 5/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 959s 1s/step - accuracy: 0.9598 - loss: 0.1199 - val_accuracy: 0.8958 - val_loss: 0.3779
Epoch 6/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 963s 1s/step - accuracy: 0.9573 - loss: 0.1128 - val_accuracy: 0.6750 - val_loss: 0.4942
Epoch 7/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 961s 1s/step - accuracy: 0.9403 - loss: 0.1647 - val_accuracy: 0.5625 - val_loss: 0.5946
Epoch 8/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 960s 1s/step - accuracy: 0.8759 - loss: 0.3110 - val_accuracy: 0.8583 - val_loss: 0.4743
Epoch 9/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 960s 1s/step - accuracy: 0.9754 - loss: 0.0651 - val_accuracy: 0.8333 - val_loss: 0.4871
Epoch 10/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 949s 1s/step - accuracy: 0.9792 - loss: 0.0548 - val_accuracy: 0.8250 - val_loss: 0.4925
Epoch 11/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 957s 1s/step - accuracy: 0.9752 - loss: 0.0661 - val_accuracy: 0.8583 - val_loss: 0.3834
Epoch 12/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 937s 1s/step - accuracy: 0.9745 - loss: 0.0657 - val_accuracy: 0.5750 - val_loss: 0.7800
Epoch 13/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 899s 1s/step - accuracy: 0.9675 - loss: 0.0910 - val_accuracy: 0.8167 - val_loss: 0.5169
Epoch 14/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 893s 1s/step - accuracy: 0.9809 - loss: 0.0554 - val_accuracy: 0.6292 - val_loss: 0.4650
Epoch 15/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 892s 1s/step - accuracy: 0.9761 - loss: 0.0684 - val_accuracy: 0.8417 - val_loss: 0.4292
Epoch 16/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 891s 1s/step - accuracy: 0.9859 - loss: 0.0390 - val_accuracy: 0.6875 - val_loss: 0.5390
Epoch 17/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 889s 1s/step - accuracy: 0.9760 - loss: 0.0656 - val_accuracy: 0.8833 - val_loss: 0.4911
Epoch 18/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 888s 1s/step - accuracy: 0.9824 - loss: 0.0387 - val_accuracy: 0.8667 - val_loss: 0.3536
Epoch 19/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 889s 1s/step - accuracy: 0.9817 - loss: 0.0557 - val_accuracy: 0.9083 - val_loss: 0.3846
Epoch 20/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 888s 1s/step - accuracy: 0.9784 - loss: 0.0560 - val_accuracy: 0.8917 - val_loss: 0.2831
Epoch 21/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 888s 1s/step - accuracy: 0.9709 - loss: 0.0724 - val_accuracy: 0.6583 - val_loss: 0.4887
Epoch 22/60
727/727 ━━━━━━━━━━━━━━━━━━━━ 889s 1s/step - accuracy: 0.9840 - loss: 0.0352 - val_accuracy: 0.5333 - val_loss: 0.5702
Epoch 23/60
409/727 ━━━━━━━━━━━━━━━━━━━━ 6:27 1s/step - accuracy: 0.9835 - loss: 0.0504