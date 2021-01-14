# Facial Expression Recognition System

This project aims to detect and classify facial expressions using Convolutional Neural Networks (CNN) implemented in Python using Keras. The model identifies seven primary emotions: angry, disgust, fear, happy, neutral, sad, and surprise. This repository contains the code to build, train, evaluate, and use the CNN model for facial expression detection in real-time using a webcam.

## Table of Contents

* [Features](#features)
* [Data Preparation](#data-preparation)
* [Model Architecture](#model-architecture)
* [Requirements](#requirements)
* [Usage](#usage)
  * [Training](#training)
  * [GUI (Webcam Detection)](#gui-webcam-detection)
* [Model Evaluation](#model-evaluation)
* [Visualization](#visualization)
* [Exporting the Model](#exporting-the-model)

## Features

* Trains a CNN model using grayscale images of facial expressions dataset.
* Uses data augmentation (rotation, shear, zoom, shift, flip) to improve model generalization.
* Techniques like Batch Normalization, Dropout, and Early Stopping for enhanced training.
* Includes a GUI using OpenCV for real-time facial expression detection via webcame.
* Saves the best model during training based on validation accuracy.
* Exports the model and weights for future use.

## Data Preparation

Dataset Link: [FER-2013 (kaggle.com)](https://www.kaggle.com/datasets/msambare/fer2013)

* Training Data: Located in `fer2013/train` directory, categorized into folders representing different emotions.
* Validation Data: Located in `fer2013/test` directory.
* Image Size: Grayscale 48x48 pixels

Data augmentation techniques include rescaling, rotation, shear, zoom, width/height shifts, and flipping for training data.

## Model Architecture

The model consists of multiple convolutional blocks, each containing:

* Conv2D layers with ReLU activation and Batch Normalization.
* MaxPooling layers for dimensionality reduction.
* Dropout layers for regularization.

The final block includes:

* A Flatten layer.
* Fully connected Dense layers.
* A softmax activation function to classify into one of the seven emotion categories.

## Requirements

* Python
* Tensorflow
* Keras
* OpenCV
* Numpy
* Matplotlib

Install the required libraries:

```shell
pip install tensorflow keras opencv-python numpy matplotlib
```

## Usage

### Training the Model

1. Data Augmentation: `ImageDataGenerator` is used to apply transformations and generate batches for training and validation.
2. The model is compiled using the Adam optimizer and categorical cross-entropy loss. Early stopping and model checkpointing are used during training.

   ```python
   model.fit(
       model_train_generator,
       steps_per_epoch=nb_train_samples//batch_size,
       epochs=epochs,
       callbacks=callbacks,
       validation_data=model_validation_generator,
       validation_steps=nb_validation_samples//batch_size
   )
   ```
3. Model Evaluation: Evaluating model performance

   ```python
   model.evaluate(model_train_generator)
   model.evaluate(model_validation_generator)
   ```
4. Model Export: Saving the trained model

   ```python
   export = model.to_json() 
   with open("model.json", "w") as json_file:   
       json_file.write(export) 
       model.save_weights("weight.h5")
   ```

### GUI (Webcam Detection)

1. Pre-trained model and Haarcascade file is required for this.
2. Uses OpenCV to access the webcam and detects faces in real-time.
3. Run `FacialExpressionDetectorGUI.ipynb` for access.
4. Press `E` key to exit the GUI.

## Model Evaluation

* After training, the model's performance is evaluated using training and validation data, displaying loss and accuracy.
* The script includes real-time emotion detection using a webcam and displays the detected emotion on the screen.

## Visualization

* **Class Distribution**: Bar graphs showing the distribution of training and validation data for each emotion.
* **Training History**: Graphs for accuracy and loss comparison over epochs.

## Exporting the Model

The trained model is saved as a JSON file (`model.json`), and the weights are saved in an HDF5 file (`weight.h5`). These can be loaded later to rebuild the model without retraining.
