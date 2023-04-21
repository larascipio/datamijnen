"""
algorithms.py

- Contains the time based LSTM algorithm
- Contains the decision tree based LightGBM algorithm
"""
# BRON: https://colab.research.google.com/drive/1b3CUJuDOmPmNdZFH3LQDmt5F0K3FZhqD?usp=sharing#scrollTo=J-YOd8tLhpE0

import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint # for saving models, save models thatbest on validation 
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# from DataPreparation.datacleaning import clean_data

class LSTMM:
    def __init__(self, data):
        self._data = data
    
    def df_to_X_y(self, window_size=5):
        df_as_np = self._data.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [a for a in df_as_np[i:i+window_size]]
            X.append(row)
            label = df_as_np[i+window_size][0] # Check where 'mood' is!
            y.append(label)
        return np.array(X), np.array(y)
    
    def run(self):
        """
        Run the algorithm.
        """
        # # Scale the data using MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0,1))
        # data_scaled = scaler.fit_transform(self._data)
        window_size = 5 
        x, y = self.df_to_X_y(window_size)
        print(x.shape, y.shape)
        # split the data into 80% training and 20% test data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25) # 0.25 x 0.8 = 0.2 (20%)

        # print the shapes of the resulting arrays
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
        model1 = Sequential()

        # Create the model
        model1.add(InputLayer(X_train.shape[1:]))
        model1.add(LSTM(128, return_sequences=True))
        model1.add(Dropout(0.2))
        model1.add(LSTM(128))
        model1.add(Dense(1, 'linear'))
        model1.summary()

        # Train the model 
        cp1 = ModelCheckpoint('model1/', save_best_only=True)
        model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, callbacks=[cp1])
        model1.save('/Users/larascipio/Documents/Studie/datamijnen/Classification/Models')
        model1 = load_model('model1/')

        train_predictions = model1.predict(X_train).flatten()
        train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
        print(train_results)

        # Plotting
        folder_path_plots = '/Users/larascipio/Documents/Studie/datamijnen/Plots'
        
        plt.figure()
        plt.plot(train_results['Train Predictions'][:100], label='Train Predictions')
        plt.plot(train_results['Actuals'][:100], label='Actuals')
        plt.legend()
        plt.title('Comparison training data to actual results for best LSTM model')
        
        filename = 'comparison_train_results_LSTM.png'  # specify the file name
        filepath = f'{folder_path_plots}/{filename}'  # combine folder and file names
        plt.savefig(filepath)

        val_predictions = model1.predict(X_val).flatten()
        val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
        print(val_results)

        plt.figure()
        plt.plot(val_results['Val Predictions'][:100], label='val predictions')
        plt.plot(val_results['Actuals'][:100], label='actuals')
        plt.legend()
        plt.title('Comparison validation data to actual results for best LSTM model')
        
        filename = 'comparison_validations_LSTM.png'  # specify the file name
        filepath = f'{folder_path_plots}/{filename}'  # combine folder and file names
        plt.savefig(filepath)
        

        # Test the model
        test_predictions = model1.predict(X_test).flatten()
        test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
        print(test_results)

        plt.figure()
        plt.plot(test_results['Test Predictions'][:100], label = 'test predictions')
        plt.plot(test_results['Actuals'][:100], label='actuals')
        plt.legend()
        plt.title('Comparison testing data to actual results for best LSTM model')

        filename = 'comparison_test_prediction_LSTM.png'  # specify the file name
        filepath = f'{folder_path_plots}/{filename}'  # combine folder and file names
        plt.savefig(filepath)
        
        # Evaluation measures
        train_mae = mean_absolute_error(train_results['Actuals'], train_results['Train Predictions'])
        train_rmse = mean_squared_error(train_results['Actuals'], train_results['Train Predictions'], squared=False)

        val_mae = mean_absolute_error(val_results['Actuals'], val_results['Val Predictions'])
        val_rmse = mean_squared_error(val_results['Actuals'], val_results['Val Predictions'], squared=False)

        test_mae = mean_absolute_error(test_results['Actuals'], test_results['Test Predictions'])
        test_rmse = mean_squared_error(test_results['Actuals'], test_results['Test Predictions'], squared=False)

        print("Train MAE:", train_mae)
        print("Train RMSE:", train_rmse)

        print("Validation MAE:", val_mae)
        print("Validation RMSE:", val_rmse)

        print("Test MAE:", test_mae)
        print("Test RMSE:", test_rmse)

        # Keep evaluation measures
        evaluation_measures_model1 = pd.DataFrame({'MAE':[train_mae, val_mae, test_mae], 'RMSE':[train_rmse, val_rmse, test_rmse]}, 
                                           index=['Train', 'Validation', 'Test'])
        evaluation_measures_model1.to_csv('/Users/larascipio/Documents/Studie/datamijnen/Classification/Evaluations/evaluations_model1.csv')

class Tree: 
    def __init__(self, data):
        self._data = data
        self._predictors = self._data.loc[:, self._data.columns != 'mood']
        self._target = self._data['mood']

    def run(self):
        """
        Run the algorithm.
        """
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(self._predictors, self._target, test_size=0.2, random_state=42)

        # Train the LightGBM model
        model = lgb.LGBMClassifier(n_estimators=1000, max_depth=5)
        model.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(x_test)

        # Compute the accuracy score
        accuracy = (y_pred == y_test).sum() / len(y_test)
        print("Accuracy:", accuracy)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("AUC-ROC score:", roc_auc_score(y_test, y_pred))
        print("Accuracy:", accuracy)

        # metrics: RMSLE RMSE MAPE MAE
        # we use: mean squared error (MSE) and mean absolute error (MAE). 


if __name__ == '__main__':

    # # Import and clean data 
    # data1 = pd.read_csv('/Users/larascipio/Documents/Studie/datamijnen/Data/cleaned_data.csv')
    # data1.fillna(0,inplace=True)
    # data = pd.read_csv("/Users/larascipio/Documents/Studie/datamijnen/Data/df_temporal.txt")
    # data1.reset_index(drop=False,inplace=True)
    # print(data1.head())
    # data1 = data1.drop(columns=['id', 'date'])
    
    # # TIJDELIJK: drop columns

    # # drop the 'id' and 'date' columns
    # data1 = data1.drop(columns=['id', 'date'])
    # data.set_index(['ID','time','t'],drop=True,inplace=True)
    # data.reset_index(drop=False,inplace=True)
    # # print(data.head())
    # data.fillna(0,inplace=True)
    # # ids = df.ID.unique()
    # X = data[data.ID.values == 'AS14.01'].iloc[:,range(3,len(data.columns))]
    # print(X.head())
    # # print(data.shape)
    # # print(X.shape)
    # window = 5
    # df = data
    # last = int(len(df[df.ID.values == 'AS14.01'])/5)
    # X_train = X[:-last]
    # X_test = X[-last-window:]
    # in_tr = []
    # out_tr = []
    # for j in range(window,len(X_train)):
    #     in_tr.append(np.array(X_train.iloc[j-window:j,:]))
    #     out_tr.append(np.array(X_train.iloc[j,0]))
     
    # in_tr, out_tr = np.array(in_tr), np.array(out_tr)

    # in_te = []
    # out_te = []
    # for j in range(window,len(X_test)):
    #     in_te.append(np.array(X_test.iloc[j-window:j,:]))
    #     out_te.append(np.array(X_test.iloc[j,0]))
     
    # in_te, out_te = np.array(in_te), np.array(out_te)
    # print(in_tr.shape, out_tr.shape, in_te.shape, out_te.shape)

    df = pd.read_csv("/Users/larascipio/Documents/Studie/datamijnen/Data/df_temporal.txt",index_col=0)
    df.set_index(['ID','time','t'],drop=True,inplace=True)
    df.reset_index(drop=False,inplace=True)
    df.fillna(0,inplace=True)
    print(df.head())
    X = df.iloc[:, 3:]
    print(X.head())
    # Initialize
    experiment1 = LSTMM(X)
    experiment1.run()
    # experiment2 = Tree(data)
    # experiment.run()
    # qual = experiment.quality()






