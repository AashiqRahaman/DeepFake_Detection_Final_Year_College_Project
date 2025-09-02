# 🎭 DeepFake Detection using Soft Attention & XceptionNet

## 📌 Overview

This project is my **final year B.Tech project**, focusing on **DeepFake detection** using **Soft Attention Mechanism** on top of **XceptionNet**.
The goal is to build a deep learning model that can classify whether a given video frame is **real** or **fake** with high accuracy.

---

## 🛠 Tech Stack

* **Language**: Python
* **Frameworks**: TensorFlow, Keras
* **Computer Vision**: OpenCV
* **Deep Learning**: CNN, Attention Mechanism, Transfer Learning (XceptionNet)

---

## 📂 Dataset Structure

The dataset should be organized as:

```
1000_videos/
├── train/
│   ├── fake/
│   └── real/
├── validation/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

Each folder (`real/`, `fake/`) contains image frames (`.png`) extracted from videos.

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DeepFake_Detection_College_Project.git
cd DeepFake_Detection_College_Project
```

### 🔹 2. Create Virtual Environment (Windows)

```bash
python -m venv df
df\Scripts\activate
```

OR using Conda:

```bash
conda env create -f environment_win.yml
conda activate DeepFakeDetection
```

### 🔹 3. Install Dependencies

```bash
pip install -r requirements.txt
```

(If `requirements.txt` not available, manually install:)

```bash
pip install opencv-python opencv-contrib-python tensorflow numpy matplotlib
```

---

## 🚀 Training the Model

Run the following command:

```bash
python main.py --train "path_to_dataset/train" \
               --val "path_to_dataset/validation" \
               --epochs 20 \
               --batch 32 \
               --steps 100
```

🔹 Example in my case (Windows PowerShell): you can change it as per yours

```bash
python main.py --train "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\train" `
               --val "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\validation" `
               --epochs 20 `
               --batch 32 `
               --steps 100
```
## Train will be looks like that...
```
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
```
---

## 📊 Model Architecture

* **Base Model**: XceptionNet (pretrained on ImageNet)
* **Attention Mechanism**: Soft Attention Layer to enhance discriminative features
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Metrics**: Accuracy

---

## 📈 Results (WIP 🚧)

Currently training and testing on the **1000\_videos dataset**.
Results & performance metrics will be updated soon.

---

## 🛤 Future Work

* Add **real-time DeepFake detection** from video streams
* Train on larger datasets like **FaceForensics++**
* Deploy as a **web app** for demonstration

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome! Please feel free to fork the repo and raise a PR.

---

## 📜 License

This project is for **academic purposes only**. Not intended for malicious use.

---

✨ Author: **Aashiq Rahaman**

---

👉 Do you want me to also **add an "Inference Section"** (how to test a single image/video after training) so that your README covers both **training + testing** flow?
