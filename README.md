# Short Term BTC Price Prediction

The goal of this project is to predict Bitcoin's Price.

This project will be presented as the Capstone project for the [Udacity Data Scientist Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

For better visualisation of the notebook, please follow: https://nbviewer.jupyter.org/github/sjuanandres0/crypto/blob/main/Main.ipynb (It contains plotly graphs which are not interactive when rendered in GitHub).

## Disclaimer

The model and predictions do not pretend to be used as a financial advice. They were created for educational purposes and with a time constraint. Do not take financial decisions based on the results.

## Description

In this project, I will try to predict BTC's Price based on its own Price and Volume on previous observations.

For this purpose, I will use a Deep Learning, more precisely a Recurrent Neural Network (RNN) model with Long Short Term Memory cells (LSTM).

Additionally, Early stopping and Dropout regularization were added to prevent overfitting.

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

The main output is the BTC Price for the n_future observations. E.g.: if n_future is equal to 10, the model will return the BTC predicted price for the next 10 days from the last observation recorded.

## Run

### Requirements
1. Python 3.5 or higher
2. Python libraries (pip install): numpy, pandas, datetime, math, matplotlib, keras, sklearn and plotly.

### Clone this repo
```
git clone https://github.com/sjuanandres0/crypto.git
```

### Dataset
If you want to use the model with more recent data, you can retrieve it from [Yahoo Finance BTC-USD](https://finance.yahoo.com/quote/BTC-USD?p=BTC-USD) and overwrite the 'data/yahoo_BTC-USD.csv' file.

Else, there is already data up to 10th of June of 2021.

### Script
1. Launch Jupyter Notebook
2. Open the Main.ipynb
3. Run them cell by cell to understand what is going on. Markdown and comments will help you understand each line of code. Note that you can tweak the parameters in section 'Parameter settings'.
4. Output. After running the script, a folder will be generated containing the plots for Loss, Accuracy, Validation and Predictions, a csv with the predictions and the regressor in .h5 format in case it would be needed in the future to predict new values (with different inputs).

### Results
The results do not look very promising. 

By looking at the Loss and Accuracy plots, we would like to see the true and the validation lines to converge. However, they are not precisely converging as much as we would like. 

Trying with different parameters and RNN architecture in the given time was not possible to improve the model performance much more. This is attribuitable to the high volatility of BTC Price, being this highly dependant on external factors and rather than its own price and volume. For example, Elon Musk's tweets, countries releasing approval/restrictions, companies adopting BTC (VISA, PayPal, Tesla, etc).

## Article on Medium
Coming soon...

## Built with
* [Python](https://www.python.org/)
* [Jupyter](https://jupyter.org/)
* [Microsoft Visual Studio Code](https://code.visualstudio.com/) 
* [Anaconda](https://www.anaconda.com/products/individual-b)

## Versioning
We use [Git](https://git-scm.com/) for versioning. 

## Documentation
A wide variety of sources were used. You may find useful links in [links_useful.md](https://github.com/sjuanandres0/crypto/blob/main/links_useful.md).

I will outline here the main ones:
- Udacity Data Scientist Nanodegree course and the extracurriculars.
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron.
- [Machine learning mastery](https://machinelearningmastery.com/)
- [Kaggle](https://www.kaggle.com/)

## Further improvements
Due to time contraints, I was not able to complete the project as I'd like, but here I will outline few ideas that can be further developed:
- Include more features that influence BTC price. E.g.: some other stock exchange price, Twitter data, Whale data, altcoins data, active BTC addresses, etc.
- Include other models to compare the results, such as, ARIMA, SARIMA, Facebook Prophet, among others. We could also try GRU cells in stead of LSTM for the RNN.
- User more datapoints through reducing the time intervals. For this notebook, I used daily prices, but we could try with hourly prices for example.
- A trained model could be saved and through an API we could check the BTC price daily (and same for other features if we decide to have more) and make the BTC Price prediction for the next day. This could be automated with GitHub Actions. Since the model training may take some time (and may exceed the 2000 minutes monthly), we could train the model weekly (or in a different frecuency) in our local machines.
- Furthermore, Telegram messages could also be triggered.
