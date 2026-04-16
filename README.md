# Ground2Aerial — Generation of Aerial Images from Ground-Level Views for Cross-View Matching

## Abstract

This project replicates the architecture proposed in *"Bridging the Domain Gap for Ground-to-Aerial Image Matching"* by Krishna Regmi and Mubarak Shah. The system combines a **GAN (X-Fork architecture)** with a **Joint Feature Learning** module to generate synthetic aerial images from ground-level (street view) photographs, and subsequently use them to improve cross-view image matching.

The interest in this work lies in its applicability to real-world tasks such as localization, navigation, and geographic mapping.

---

## Repository Structure

```
ImageGen-for-CrossView/
│
├── CVUSA_subset/
│   ├── bingmap/            # Aerial (satellite) images
│   └── streetview/         # Ground-level (street view) images
│
├── Code/
│   ├── FeatureExtractor/
│   │   └── VGG.py          # VGG-based feature extractor
│   │
│   ├── JoinFeatureLearning/
│   │   ├── JFL.py          # Joint Feature Learning loss and network
│   │   └── main.py         # Training script for JFL
│   │
│   ├── XFork/
│   │   ├── generator.py    # U-Net-like generator (image + segmentation)
│   │   ├── discriminator.py
│   │   └── main.py         # GAN training script
│   │
│   ├── blocks.py           # Shared EncoderBlock / DecoderBlock modules
│   ├── dataset.py          # Dataset class with Canny edge concatenation
│   ├── edge_Concatenate.py # Canny edge detection and 4-channel concatenation
│   ├── eval.py             # Evaluation and visualization script
│   ├── meansOfImages.py    # Dataset mean/std computation utilities
│   └── requirements.txt
│
├── Presentation/
│   └── Computer_Vision_Presentation/
│
├── Sources/
│   ├── 1904.11045v2.pdf
│   └── Regmi_Cross-View_Image_Sy...
│
├── .gitignore
└── README.md
```

---

## Architecture Overview

### 1. X-Fork GAN

The generator follows a **U-Net-like encoder-decoder** architecture:

- **Input**: 4-channel street view image (RGB + Canny edge map), shape `[B, 4, 224, 1232]`
- **Output**: synthetic aerial image `[B, 3, 512, 512]` and a segmentation map `[B, 1, 512, 512]`

The discriminator evaluates real/fake aerial images conditioned on the street view input.

### 2. Joint Feature Learning

Three **weight-sharing VGG networks** extract features from:
- the ground-level image (`f_g`)
- the real aerial image (`f_a_pos`)
- the synthetic aerial image (`f_a_gen`)

A **triplet loss** is used to train the network so that matching pairs are pulled closer in feature space than non-matching pairs.

---

## Dataset

Due to limited computational resources, a reduced version of the [CVUSA dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/) was used.

- **Full dataset**: available upon request at the link above
- **Reduced version**: available at [SemanticAlignNet](https://pro1944191.github.io/SemanticAlignNet/)

Each sample consists of a street view image (panoramic) and its corresponding aerial (Bing Maps) image. A random negative aerial image is also sampled per item for triplet training.

---

## How to Run

### Step 1 — Train the GAN

```bash
python Code/XFork/main.py
```

This trains the generator and discriminator and saves the best model weights to:
```
Code/models/generator.pth
Code/models/discriminator.pth
```

### Step 2 — Train the Joint Feature Learning module

```bash
python Code/JoinFeatureLearning/main.py
```

This loads the pre-trained generator and trains the feature extraction network using triplet loss.

### Step 3 — Evaluate results

```bash
python Code/eval.py
```

Displays a side-by-side comparison of the original aerial image and the generated one.

> **Note**: Training was performed on [Kaggle](https://www.kaggle.com/). Pre-trained notebooks:
> - [GAN Training](https://www.kaggle.com/code/alessandro1103/primo-train-gan)
> - [Joint Feature Learning Training](https://www.kaggle.com/code/alessandro1103/notebookdc51fdc810)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Alessandro1103/ImageGen-for-CrossView.git
cd ImageGen-for-CrossView
```

2. Install the required dependencies:
```bash
pip install -r Code/requirements.txt
```

> A CUDA-compatible GPU is strongly recommended for training.

---

## Key Dependencies

| Library | Version |
|---|---|
| PyTorch | 2.4.1 |
| torchvision | 0.19.1 |
| OpenCV | 4.11.0 |
| NumPy | 1.24.4 |
| Matplotlib | 3.7.5 |
| tqdm | 4.67.1 |

---

## Author

**Alessandro De Luca**
