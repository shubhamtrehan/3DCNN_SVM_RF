**Overview**

This repository implements an end-to-end video classification pipeline that combines a custom 3D Convolutional Neural Network (3D CNN) with classical machine learning classifiers (Random Forest and Support Vector Machine). The 3D CNN is used for feature extraction from video frames, and the extracted features are then fed into the classifiers for further evaluation. The project leverages a custom frame extraction utility to process videos (e.g., from the UCF101 dataset) into a format suitable for deep learning.

  
**Repository Structure**

- **3dcnn_svm_rf.py**  
  This file is the main pipeline that ties together all components. It:
  
  - Loads video data using a custom `FrameGenerator` from the **utils.py** file.
  - Creates TensorFlow datasets for training, validation, and testing.
  - Builds and trains a 3D CNN model (defined in **cnn_model.py**) on the video data.
  - Plots the training history (loss and accuracy).
  - Creates a feature extractor from an intermediate layer of the trained 3D CNN.
  - Extracts features from the datasets and trains a Random Forest classifier on them.
  - Trains and evaluates SVM classifiers with various kernels (linear, RBF, polynomial, and sigmoid).
  - Compares the performance of the classical classifiers against the original 3D CNN model.

  
- **cnn_model.py**  
  This module defines the custom 3D CNN architecture. Key components include:
  
  - **Conv2Plus1D Layer:**  
    Implements spatial and temporal convolutions sequentially using 3D convolutions.
  
  - **Residual Blocks:**  
    Builds residual connections using a custom `ResidualBlock` and a helper function `add_residual_block` that also adjusts dimensions when necessary.
  
  - **ResizeVideo Layer:**  
    Uses the `einops` library to rearrange and resize video frames.
  
  - **threedcnn Function:**  
    Assembles the complete 3D CNN model. The network downsamples the video spatially after each block and finally outputs class logits via a Dense layer.

  
- **utils.py**  
  Contains utility functions for video frame processing and dataset creation:
  
  - **format_frames:**  
    Pads and resizes individual frames to a specified output size.
  
  - **frames_from_video_file:**  
    Extracts a fixed number of frames from a video file at regular intervals, applies formatting, and converts BGR to RGB.
  
  - **FrameGenerator Class:**  
    Iterates over video files within a given directory, yielding processed frames and their corresponding class labels. This generator is used to create TensorFlow datasets for training and evaluation.

  
**Requirements**

- Python 3.x  
- TensorFlow and Keras  
- scikit-learn  
- NumPy  
- OpenCV (`cv2`)  
- einops  
- Matplotlib  
- Additional libraries: pathlib, tqdm

Install the required packages using pip:

```bash
pip install tensorflow scikit-learn numpy opencv-python einops matplotlib
