# Crypto Project

The goal of this project is to predict Bitcoin's price.

This project will be presented as the Capstone project for the Udacity Data Scientist Nanodegree program. 

For visualisation of the notebook: https://nbviewer.jupyter.org/github/sjuanandres0/crypto/blob/main/Main.ipynb

## Disclaimer

The model and predictions do not pretend to be used as a financial advice. They were created for educational purposes and with a time constraint. Do not take financial decisions based on the results.

## Description

This project is for educational purposes 
Cryptocurriencies are very volatile and the market i

In this project, I will try to predict Bitcoin's price based on its own Price and Volume on previous observations.

For this purpose, I will use a Recurrent Neural Network (RNN) model with Long Short Term Memory cells (LSTM).

Early stopping and dropout were added to prevent overfitting.

The notebook contains a first section with parameters that can be tuned, those are:
- n_past_total: number of total past observations from the original dataset to be considered.
- n_past: number of past observations to be considered for the LSTM training and prediction.
- n_future: number of future datapoints to predict (if 1, the model is Single-Step, if higher than 1 it becomes Multi-Step).
- activation: activation function used for the RNN.
- dropout: dropout for the hidden layers.
- n_layers: number of hidden layers.
- n_neurons: number of neurons in the hidden layers.
- features: inputs (if only input is Close, the model is Univariate; if more, then it's Multivariate).
- patience: number of epochs with no improvement after which training will be stopped.
- optimizer: optimization method used for training the model.

## 