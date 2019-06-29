# Project notMNISTClassifier

## Introduction
   This project implements a series of different classifiers for the notMINST dataset. The different classifiers are defined and implemented using TensorFlow. 

## Getting Started
 The code included in this repository does not require a specific environment like AWS or Google Colab to run. To run this code in those environments or in a Jupyter Notebook, additional modifications may be required. 
1. To download and create the training, validation, and testing data, run notMNISTdataCollector.py
2. Run the .py file containing the classfier of your choice.

## Classifiers
* notMNISTdataCollector.py
   * Derived from Udacity's Deep Learning Course examples.
   * This program downloads the dataset and prepares three datasets for use by a classifier
   * The training, validation, and test dataset are stored in a pickle file for later use. 

* classifierLogScikit.py
   * A logistic classifier implemented using the Sci-kit learn library

* classifierCNN.py
   * A convolutional neural network implemented using TensorFlow's Graph API

* classifierLogTensor.py
   * A neural network classifier implemented using TensorFlow's Estimator API
   

