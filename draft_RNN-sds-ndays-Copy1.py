import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.float_format', lambda x: '%.3f' % x) #avoid scientific notation
import datetime
import sys
import pickle
import joblib

# reading csv
dataset = pd.read_csv('yahoo_BTC-USD.csv')
dataset

# checking if close is not equal to adj close
dataset[dataset['Close']!=dataset['Adj Close']]

# checking for nulls
dataset.info()


# use close only and fill NaN with ffil
df = dataset.set_index('Date')[['Close']].tail(2000)
df = df.set_index(pd.to_datetime(df.index))
df.fillna(method='ffill',inplace=True)


# plotting the Closing Prices
df.plot(figsize=(14,8))
plt.title('BTC Closing Prices')
plt.ylabel('Price')
plt.show()


# train test split
prediction_days = 10
df_train = df.head(-prediction_days)
df_test = df.tail(prediction_days)
training_set = df_train.values
test_set = df_test.values
print('training_set.shape = ', training_set.shape)
print('test_set.shape = ', test_set.shape)


# scale
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled


training_set_scaled.shape


len(training_set_scaled)


# creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
 
n_future = prediction_days #20  # Number of days you want to predict into the future
n_past = 60  # Number of past days you want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i-n_past:i, 0])
    y_train.append(training_set_scaled[i:i+n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)


# # another efficient and nice way
# # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# # https://github.com/marcosan93/Price-Forecaster/blob/master/BTC-Models/BTC-RNN-Deep-Learning.ipynb
# def split_sequence(seq, n_steps_in, n_steps_out):
#     """
#     Splits the univariate time sequence
#     """
#     X, y = [], []
#     
#     for i in range(len(seq)):
#         end = i + n_steps_in
#         out_end = end + n_steps_out
#         
#         if out_end > len(seq):
#             break
#         
#         seq_x, seq_y = seq[i:end], seq[end:out_end]
#         
#         X.append(seq_x)
#         y.append(seq_y)
#     
#     return np.array(X), np.array(y)
# X, y = split_sequence(training_set, 60, 7)
# X.shape, y.shape

X_train.shape, y_train.shape


range(n_past, len(training_set_scaled) - n_future + 1)


range(n_past, len(training_set_scaled)-7)


for i in range(n_past, len(training_set_scaled) - n_future + 1):
    print('i={}, [{}, {})'.format(i, i-n_past, i))


for i in range(n_past, len(training_set_scaled) - n_future + 1):
    print('i={}, [{}, {})'.format(i, i, i+n_future)) #  i,'-', i, '-', i+n_future)


training_set.shape

training_set[-1]

#training_set[1993]

# # creating a data structure with 60 timesteps and 1 output
# days_used = 60
# X_train = []
# y_train = []
# for i in range(days_used, len(training_set_scaled)):
#     X_train.append(training_set_scaled[i-days_used:i, 0])
#     y_train.append(training_set_scaled[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape, y_train.shape

# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape

# building the RNN

activation = 'softsign'

# Initialising the RNN
regressor = Sequential()

# Adding the first layer of the LSTM and some Drouput regularisation
regressor.add(LSTM(units=30, return_sequences=True, input_shape=(X_train.shape[1], 1))) #, activation='sigmoid'
# units is the number of neurons. 50 neurones is high dimensionality.
# return_sequences is because we are using stacked RNN, so another layer will come afterwards
regressor.add(Dropout(0.2))
# 20% of the neurons will be dropped/ignored during training (20% of 50, then 10 will be ignored)

# Adding the second LSTM layer and some Drouput regularisation
regressor.add(LSTM(units=12, return_sequences=True, activation=activation))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Drouput regularisation
regressor.add(LSTM(units=12, return_sequences=True, activation=activation))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Drouput regularisation
#regressor.add(LSTM(units=50, return_sequences=True, activation=activation))
#regressor.add(Dropout(0.3))
regressor.add(LSTM(units=12, return_sequences=True, activation=activation))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=12, return_sequences=True, activation=activation))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=12, return_sequences=True, activation=activation))
regressor.add(Dropout(0.2))

# Adding the fifth LSTM layer and some Drouput regularisation
regressor.add(LSTM(units=12, return_sequences=False, activation=activation))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=prediction_days))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#try also with RMSprop

# Model summary
regressor.summary()

# Fitting the RNN to the Training set
res = regressor.fit(X_train, y_train, batch_size=30, epochs=800, validation_split=0.1) #100/32 / 250 , validation_split=0.1)

list(res.history)

# Plotting Accuracy and Loss

results = res

history = results.history
plt.figure(figsize=(12,4))
plt.plot(history['val_loss'])
plt.plot(history['loss'])
plt.legend(['val_loss', 'loss'])
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(12,4))
plt.plot(history['val_accuracy'])
plt.plot(history['accuracy'])
plt.legend(['val_accuracy', 'accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# Validation (validation with the last training days -> -15-15 DAYS)

# getting the predictions
y_pred = regressor.predict(X_train[-1].reshape(1, n_past, 1)).tolist()[0]
y_pred = sc.inverse_transform(np.array(y_pred).reshape(-1,1))#.tolist()

# getting the true values
y_true = sc.inverse_transform(y_train[-1].reshape(-1,1))

print('y_pred:\n', y_pred.tolist())
print('y_true:\n', y_true.tolist())

# plotting the results
plt.figure(figsize=(16,5))
plt.plot(y_pred, label='Predicted')
plt.plot(y_true, label='True')

dates = df_train.index[-prediction_days:]
dates = [str(dates.date[i]) for i in range(len(dates))]
plt.xticks(range(prediction_days),dates)

plt.title('BTC price Predicted vs True')
plt.legend()
plt.show()

# Validation 2 (on the TEST SET -15 days)

# getting the predictions
x = df['Close'][-n_past-prediction_days:-prediction_days].values.reshape(-1,1)
x = sc.transform(x)
y_pred = regressor.predict(x.reshape(1, n_past, 1)).tolist()[0]
y_pred = sc.inverse_transform(np.array(y_pred).reshape(-1,1))#.tolist()

# getting the true values
y_true = df.tail(prediction_days).values

print('y_pred:\n', y_pred.tolist())
print('y_true:\n', y_true.tolist())

# plotting the results
plt.figure(figsize=(16,5))
plt.plot(y_pred, label='Predicted')
plt.plot(y_true, label='True')

dates = df.index[-prediction_days:]
dates = [str(dates.date[i]) for i in range(len(dates))]
plt.xticks(range(prediction_days),dates)

plt.title('BTC price Predicted vs True')
plt.legend()
plt.show()


# Forecasting/Predicting

# getting the predictions
x = df.tail(n_past).values.reshape(-1,1)
x = sc.transform(x)
y_pred = regressor.predict(x.reshape(1, n_past, 1)).tolist()[0]
y_pred = sc.inverse_transform(np.array(y_pred).reshape(-1,1))#.tolist()

# creating a DF of the predicted prices
y_pred_df = pd.DataFrame(y_pred, 
                         index=pd.date_range(start=df.index[-1]+datetime.timedelta(days=1),
                                             periods=len(y_pred), 
                                             freq="D"), 
                         columns=df.columns)

# getting the true values
y_true_df = df.tail(prediction_days)

# linking them
y_true_df = y_true_df.append(y_pred_df.head(1))

print('y_pred:\n', y_pred.tolist())
print('y_true:\n', y_true.tolist())

# plotting the results
plt.figure(figsize=(12,5))
plt.plot(y_pred_df, label='Predicted')
plt.plot(y_true_df, label='True')

#dates = df.index[-prediction_days:]
#dates = [str(dates.date[i]) for i in range(len(dates))]
#plt.xticks(range(prediction_days),dates)

plt.title('BTC price Predicted vs True')
plt.legend()
plt.show()


#database_filepath, model_filepath = sys.argv[1:]
#joblib.dump(regressor, 'regressor.pkl' , compress=6)


# **Improving the RNN**
# 
# Here are different ways to improve the RNN model:
# 
# Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
# 
# Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
# 
# Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
# 
# Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
# 
# Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.

# **Evaluating the RNN**
# 
# As seen in the practical lectures, the RNN we built was a regressor. Indeed, we were dealing with Regression because we were trying to predict a continuous outcome (the Google Stock Price). For Regression, the way to evaluate the model performance is with a metric called RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared differences between the predictions and the real values.
# 
# However for our specific Stock Price Prediction problem, evaluating the model with the RMSE does not make much sense, since we are more interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price. We want to check if our predictions follow the same directions as the real stock price and we don’t really care whether our predictions are close the real stock price. The predictions could indeed be close but often taking the opposite direction from the real stock price.
# 
# Nevertheless if you are interested in the code that computes the RMSE for our Stock Price Prediction problem, please find it just below:
# 
# import math
# from sklearn.metrics import mean_squared_error
# rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
# Then consider dividing this RMSE by the range of the Google Stock Price values of January 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is more relevant since for example if you get an RMSE of 50, then this error would be very big if the stock price values ranged around 100, but it would be very small if the stock price values ranged around 10000.
# 
# Enjoy Deep Learning!

