# DENSECBAM: AN ADVANCED CNN-BASED MODEL FOR PEDIATRIC PNEUMONIA CLASSIFICATION IN CXR IMAGING


Introduction
------------

Implementation of DenseCBAM: An efficient deep learning model integrating DenseNet169 and CBAM for enhanced feature extraction and attention in pediatric pneumonia detection using chest X-ray images. Achieves state-of-the-art accuracy while maintaining computational efficiency for clinical applications.
This is the source code for our submitted paper [DENSECBAM: AN ADVANCED CNN-BASED MODEL FOR PEDIATRIC PNEUMONIA CLASSIFICATION IN CXR IMAGING], which is submitted to International Journal of Imaging Systems and Technology, 2024

Architecture
------------
The architecture.png denotes the architecture of the DenseCBAM

# Installation

* Clone this repo

```
git clone https://github.com/KingsleyAmoafo13/KingsleyAmoafo13-DenseCBAM-Pneumonia-Detection
```
* Install all dependenies

# Data Preparation

Download the dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia



# Train

```
python train.py
```

The log files are stored in the log folder

# Test

```
python test.py
```
