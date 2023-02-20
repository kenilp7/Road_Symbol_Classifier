# Traffic Light Classifier 

## Introduction

The project aims to build a convolutional neural network (CNN) and train in TensorFlow to classify traffic signs using the German Traffic Sign Dataset.

## Dependencies

- TensorFlow with GPU support or Google Colab.
- The German Traffic Sign Dataset taken from https://bitbucket.org/jadslim/german-traffic-signs.
- You can use git clone to load the dataset in your Jupyter python file.

## Instructions

- Clone the repository.
- Launch the Jupyter notebook: "Traffic_light_classifier.ipynb".
- Execute the desired code cells, note that some cells may depend on previous cells.

## Data Set Summary and Exploration

- The training set contains 34799 images, validation set 4410 images, and test set 12630 images.
- The image shape is 32 x 32 x 3.
- There are 43 unique classes/labels in the dataset.

## Training

- The training data is unbalanced, with varying number of samples per class.
- Augmented images were generated for classes with limited samples.
- Normalization of the data was done to bring the range between 0.0 and 1.0.
- The convolutional neural network used has three convolutional layers, three fully connected layers, ReLU activation function, and Dropout for regularization.
- The model was trained with a batch size of 50 samples for 10 epochs.
- AdamOptimizer was used as the optimizer with a learning rate of 0.001.

## Results

- The training set accuracy is 96.44%, validation set accuracy is 98.3%, and test set accuracy is 97.3%.
