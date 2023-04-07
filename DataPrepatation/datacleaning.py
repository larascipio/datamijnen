# Import the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Create a function to report the limits of the Z-Score (Not really used in this project) as it is not a good method 
# for detecting outliers when the data is not normally distributed.
def print_z_score_limits(df, column_name):
    """ Print the upper and lower limits of the Z-score """
    
    # Compute the limits (3 = threshold: default)
    upper_limit = df[column_name].mean() + 3 * df[column_name].std()
    lower_limit = df[column_name].mean() - 3 * df[column_name].std()
    
    # Round and return the limits
    upper_limit = round(upper_limit, 2)
    lower_limit = round(lower_limit, 2)
    print_this = "Variable Name: " + column_name + " | Upper limit: " + str(upper_limit) + " | Lower limit: " + str(lower_limit)
    return(print_this)


# Isolation Forest Algorithm
def run_isolation_forest(model_data: pd.DataFrame, 
                         contamination = 0.005, 
                         n_estimators= 200, 
                         max_samples = 0.7) -> pd.DataFrame:
    
    IF = (IsolationForest(random_state = 0,
                          contamination = contamination,
                          n_estimators = n_estimators,
                          max_samples = max_samples))
    
    # Drop any rows with NaN values in the 'value' column and store the result in a new dataframe
    df = model_data.copy()
    df.dropna(subset=['value'], inplace=True)
    
    # Fit the model
    IF.fit(df[['value']])
    
    # Datapoints classified -1 are anomalous (outliers): made 1 here for plotting
    output = pd.Series(IF.predict(df[['value']])).apply(lambda x: 1 if x == -1 else 0)
    score = IF.decision_function(df[['value']])

    # Convert the array to a dataframe with a column name 'score'
    score_df = pd.DataFrame({'score': score})

    # Concatenate the score_df and output along axis 1
    merged_df = pd.concat([score_df, output], axis=1)
    merged_df.index = df.index

    # Add the id column to the merged_df dataframe
    merged_df = pd.concat([df, merged_df], axis=1)
    
    # Rename the columns
    merged_df.columns = ['id', 'time', 'variable', 'value', 'score', 'anomaly']

    return merged_df




# Make a function to impute missing values, based on ARIMA modeling.
# It can be used to model the time series and then forecast the missing values using the estimated model. 
# This method works well when the time series exhibits a complex pattern that cannot be captured by a simple linear trend.

# Implementing ARIMA, like so:
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Create the ARIMA model
def create_arima_model(train, test, p, d, q):
    """ Create the ARIMA model """
    
    # Create the model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit(disp=0)
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))[0]
    
    # Compute the error
    error = mean_squared_error(test, predictions)
    
    # Return the predictions and the error
    return predictions, error



# Implement Deep learning methods such as recurrent neural networks (RNNs) and Long short-term memory (LSTM) networks 
# to impute missing values in time series data. These methods work well when the time series exhibits a complex pattern 
# that cannot be captured by a simple linear trend and when there is a large amount of data available for training.

# Implement the Long short-term memory (LSTM) network, like so:
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Create the LSTM model
def create_lstm_model(train, batch_size, epochs, neurons):
    """ Create the LSTM model """
    
    # Reshape the data
    train = train.reshape(train.shape[0], 1, train.shape[1])
    
    # Create the model
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, train.shape[1], train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Fit the model
    for i in range(epochs):
        model.fit(train, train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    
    # Return the model
    return model


# Implement the Recurrent neural network (RNN) model, like so:
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Create the RNN model
def create_rnn_model(train, batch_size, epochs, neurons):
    """ Create the RNN model """
    
    # Reshape the data
    train = train.reshape(train.shape[0], 1, train.shape[1])
    
    # Create the model
    model = Sequential()
    model.add(SimpleRNN(neurons, batch_input_shape=(batch_size, train.shape[1], train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Fit the model
    for i in range(epochs):
        model.fit(train, train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    
    # Return the model
    return model



# Another approach for handling missing values: Bayesian Structural Time Series = Bayesian structural time series is 
# a Bayesian approach that models the underlying structure of a time series and the uncertainty surrounding it. The 
# approach uses a hierarchical model to estimate the parameters of the model and the posterior distribution of the 
# missing data. The posterior distribution can be used to generate imputations for the missing data, and the final 
# estimates are obtained by averaging across the imputations. Bayesian structural time series is effective in 
# handling missing data because it models the underlying structure of the time series and accounts for the 
# uncertainty in the data.

# Build a Bayesian structural time series model for impution of missing values, like so:
from bsts import BVAR, BVARMAX
from bsts.models import *

# Implement the Bayesian structural time series model to impute missing values
def create_bsts_model(train, test, p, q):
    """ Create the Bayesian structural time series model """
    
    # Create the model
    model = BVARMAX(train, p=p, q=q)
    model.fit()
    
    # Make predictions
    predictions = model.predict(steps=len(test))
    
    # Compute the error
    error = mean_squared_error(test, predictions)
    
    # Return the predictions and the error
    return predictions, error


# State-Space Models: State-space models are a class of models that can capture the underlying dynamics of a time 
# series, including trends, seasonal patterns, and other patterns that may be present in the data. These models can 
# be used to impute missing values by estimating the unobserved states of the system that generated the time series. 
# This approach can be especially useful for handling prolonged periods of missing data, as it can exploit the 
# information contained in the observed data to make more accurate predictions.

# Build a state-space model for impution of missing values, like so:
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create the state-space model
def create_state_space_model(train, test, p, d, q):
    """ Create the state-space model """
    
    # Create the model
    model = SARIMAX(train, order=(p, d, q))
    model_fit = model.fit(disp=0)
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    
    # Compute the error
    error = mean_squared_error(test, predictions)
    
    # Return the predictions and the error
    return predictions, error





    








