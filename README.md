# TensorFlow Image Classification Project

This project demonstrates how to train a deep learning model using TensorFlow for image classification. The dataset used is the CNR-EXT-150x150 dataset, which contains images of vehicles in parking lots. The goal is to classify images into two categories: occupied or free parking spaces.

## Project Overview
This project involves training a deep learning model using TensorFlow and Keras. The model is trained on the CNR-EXT-150x150 dataset, which contains images of parking spaces. The dataset is split into training, validation, and test sets. The model is trained with data augmentation to improve generalization. After training, the model is evaluated on the test set, and a quantized version of the model is tested for accuracy.

## Dataset
The dataset used in this project is the CNR-EXT-150x150 dataset. It contains images of parking spaces labeled as either occupied or free. The dataset is divided into three sets:
- **Training set**: Used to train the model.
- **Validation set**: Used to tune hyperparameters and monitor overfitting.
- **Test set**: Used to evaluate the final model performance.

The dataset is preprocessed and augmented to improve model performance. Data augmentation techniques include random flipping, brightness adjustment, and contrast adjustment.

## Requirements
To run this project, you need the following Python libraries:
- TensorFlow
- NumPy
- Matplotlib
- Keras
- OpenCV (for image processing)

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib keras opencv-python
```

## Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/tensorflow-image-classification.git
cd tensorflow-image-classification
```
2. Download the CNR-EXT-150x150 dataset and place it in the appropriate directory.

3. Install the required Python libraries as mentioned in the Requirements section.

## Usage
1. Open the Jupyter Notebook main_tensorflow_version.ipynb:

```bash
jupyter notebook main_tensorflow_version.ipynb
```
2. Follow the instructions in the notebook to load the dataset, preprocess the images, and train the model.

3. After training, evaluate the model on the test set and test the quantized model.

## Model Training
The model architecture used in this project is based on a modified version of AlexNet (mAlexNet). The model is trained using the following steps:

1. **Data Loading and Preprocessing**: Images are loaded and preprocessed using TensorFlow's data pipeline. Data augmentation is applied to the training set.

2. **Model Definition**: The model is defined using Keras, with layers including convolutional layers, max-pooling layers, and fully connected layers.

3. **Model Training**: The model is trained using the Adam optimizer and categorical cross-entropy loss. Training progress is monitored using callbacks.

## Model Evaluation
After training, the model is evaluated on the test set to measure its accuracy. The evaluation process includes:

1. **Loading the Test Dataset**: The test dataset is loaded and preprocessed.

2. **Model Inference**: The trained model is used to make predictions on the test set.

3. **Accuracy Calculation**: The accuracy of the model is calculated and displayed.

4. **Real-world applicability**: The code in DrawingRectangles.ipynb includes functionality to test the model on images from the PKLot dataset, which simulates a more real-time scenario. Patches from the images are cropped and the model is run on these patches to detect parking spaces.

## Quantized Model Testing
A quantized version of the model is tested to evaluate its performance. Quantization reduces the model size and improves inference speed, which is useful for deployment on resource-constrained devices. The quantized model is tested using TensorFlow Lite.

## Results
The results of the model training and evaluation are displayed in the notebook. The accuracy of the model on the test set is reported, along with the accuracy of the quantized model.
