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
python main.py --train "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\train" `
               --val "C:\Users\aashiq\Desktop\Git\MY_Git_Hub_AashiqRahaman\DeepFake_Detection_College_Project\Assets\1000_videos\validation" `
               --epochs 5 `
               --batch 10 `
               --steps 100
```

```bash
python main.py \
  --train "Assets/1000_videos/train" \
  --val "Assets/1000_videos/validation" \
  --epochs 3 \
  --batch 10 \
  --steps 100

```
``` bash
python main.py --train "Assets\1000_videos\train" --val "Assets\1000_videos\validation" --epochs 2 --batch 10 --steps 100
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