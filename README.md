# Gesture Recognition Model

This repository contains the code and resources for developing a Bisindo Gesture Recognition model. The project aims to help hearing individuals learn sign language by recognizing and interpreting gestures, thereby fostering inclusivity and bridging the communication gap between the hearing and deaf communities.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Project Structure

```
machine-learning/
├── dataset-create.ipynb
│ 
├── train-code-final-dataset/
│   ├── train-GRU.ipynb
│   ├── train-LSTM.ipynb
│   └── train-convLSTM.ipynb
│ 
├── model-final-dataset/
│   ├── 33Class_ConvLSTM_acc098_loss007_100seq.h5
│   ├── 33Class_GRU_acc098_loss01_100seq.h5
│   └── 33Class_LSTM_acc098_loss01_100seq.h5
│
├── test-code-final-dataset/
│   ├── video-test/
│   │   ├── A.mp4
│   │   ├── Bertemu.mp4
│   ├── test-batch-realTime-ConvLSTM.ipynb
│   ├── test-batch-realTime-GRU-LSTM.ipynb
│   └── test-phone-convLSTM.ipynb
│ 
├── performance-result-final-dataset/
│   ├── 33Class_ConvLSTM_acc098_loss007_100seq.png
│   ├── 33Class_GRU_acc098_loss01_100seq.png
│   └── 33Class_LSTM_acc098_loss01_100seq.png
│
├── README.md
└── environment_droplet.yml
```

## Installation
To run this project, you need to set up a Python environment with the required libraries. You can create an environment using the provided environment_droplet.yml file.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/talkee-C241-PS12/machine-learning.git
   cd machine-learning
   ```

2. **Create the Environment:**
   ```bash
   conda env create -f environment_droplet.yml
   ```

3. **Activate the Environment:**
   ```bash
   conda activate environment_droplet
   ```

## Usage

1. **Dataset Creation:**
   - Create and preprocess the dataset in the `dataset-create.ipynb` notebook using MediaPipe and OpenCV libraries.
   - We have already created our time series dataset consists of 33 classes (26 alphabets and 7 introduction words), where each class contains 100 sequences or videos. Each sequence consists of 30 frames. You can find our dataset on Kaggle [here](https://www.kaggle.com/datasets/niputukarismadewi/talkee-bisindo-sign-language-dataset).

2. **Train Models:**
   - Use the notebooks in the `train-code-final-dataset/` directory to train models with ddifferent architectures (GRU, LSTM, ConvLSTM) using the TensorFlow library.
   - We experimented with three different architectures to identify the model with the best performance.
   - After training, the generated models are saved in the `model-final-dataset/` directory in `.h5` format.

3. **Test Models:**
   - Use the notebooks in the `test-code-final-dataset/` directory to evaluate models using new test videos.
   - `test-batch-realTime-ConvLSTM.ipynb` allows you to test the model in real-time using a laptop camera with OpenCV.
   - `test-phone-convLSTM.ipynb` is designed for testing the model using videos captured from a mobile phone.

4. **Performance Evaluation:**
   - View the `performance-result-final-dataset/` directory for model performance charts.
   - After the evaluation, ConvLSTM emerged as the best-performing model and was selected as the final model for our deployment.

