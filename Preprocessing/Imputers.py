


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














'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from utils.utils import normalization, renormalization, rounding
from utils.utils import xavier_init
from utils.utils import binary_sampler, uniform_sampler, sample_batch_index


def gain(data_x, gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
            
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data








