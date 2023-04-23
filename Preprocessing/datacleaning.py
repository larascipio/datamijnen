# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Put the above code inside a function 
def remove_incorrect_values(input_data):
    
    # Set the valid ranges for each variable
    valid_ranges = {'mood': (1, 10),
                    'circumplex.arousal': (-2, 2),
                    'circumplex.valence': (-2, 2),
                    'activity': (0, 1),
                    'screen': (0, 1000000000),
                    'call': (0, 1),
                    'sms': (0, 1),
                    'appCat.builtin': (0, 1000000000),
                    'appCat.communication': (0, 100000000),
                    'appCat.entertainment': (0, 1000000000),
                    'appCat.finance': (0, 1000000000),
                    'appCat.game': (0, 1000000000),
                    'appCat.office': (0, 1000000000),
                    'appCat.other': (0, 1000000000),
                    'appCat.social': (0, 1000000000),
                    'appCat.travel': (0, 1000000000),
                    'appCat.unknown': (0, 1000000000),
                    'appCat.utilities': (0, 1000000000),
                    'appCat.weather': (0, 1000000000)}
    
    # Filter the dataframe to remove rows with values outside the valid ranges, however, keep the NaN values
    valid_df = input_data[input_data.apply(lambda x: valid_ranges[x.variable][0] <= x.value <= valid_ranges[x.variable][1] if not pd.isnull(x.value) else True, axis=1)]
    
    # Create a separate dataframe for the removed rows
    removed_df = input_data[~input_data.index.isin(valid_df.index)]
    
    # Return the valid and removed dataframes
    return valid_df, removed_df

def convert_to_wide(input_df):
    
    input_dff = input_df.copy()
    
    # Split the time into date and time
    input_dff['date'] = input_dff['time'].dt.date
    input_dff['time'] = input_dff['time'].dt.time
    
    # Create a new long format dataframe where each row is a unique combination of id and date and the variables are the mean of the values for that day 
    pivot_df = input_dff.pivot_table(index=['id', 'date'], 
                                    columns='variable', 
                                    values='value', 
                                    aggfunc='mean').reset_index()
    return pivot_df

# Isolation Forest to detect anomalies
class DetectAnomalies:
    def __init__(self, n_estimators=200, max_samples='auto', contamination=0.005, max_features=1.0, bootstrap=False, n_jobs=-1, random_state=6):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.anomaly_dict = {}
        self.non_anomaly_dict = {}
        self.anomaly_indexes = {}

    def fit(self, input_data, columns):
        # Load the data
        data = input_data.copy()
        clean_data = input_data.copy()

        # Loop through each column
        for col in columns:
            # Remove NaN values from the column
            column_data = data[col].dropna()

            # If all values in the column are NaN, skip the column
            if len(column_data) == 0:
                continue

            # Define the anomaly detector
            clf = IsolationForest(n_estimators=self.n_estimators,
                                  max_samples=self.max_samples,
                                  contamination=self.contamination,
                                  max_features=self.max_features,
                                  bootstrap=self.bootstrap,
                                  n_jobs=self.n_jobs,
                                  random_state=self.random_state)

            # Fit the anomaly detector to the column data
            clf.fit(column_data.values.reshape(-1, 1))

            # Predict the anomalies in the column data
            preds = clf.predict(column_data.values.reshape(-1, 1))

            # Extract the anomaly values and scores for the column
            anomaly_values = column_data[preds == -1]
            anomaly_scores = clf.decision_function(column_data.values.reshape(-1, 1))[preds == -1]
            non_anomaly_values = column_data[preds == 1]
            non_anomaly_scores = clf.decision_function(column_data.values.reshape(-1, 1))[preds == 1]

            # Store the anomaly values and scores in the dictionary
            self.anomaly_dict[col] = {'anomaly_values': anomaly_values, 'scores': anomaly_scores}
            self.non_anomaly_dict[col] = {'values': non_anomaly_values, 'scores': non_anomaly_scores}

            # Store the indexes of the anomalies for each column
            self.anomaly_indexes[col] = list(anomaly_values.index)

            # Print the number of anomalies and their indexes
            print(f"{col}: {len(self.anomaly_indexes)} anomalies, Indexes: {self.anomaly_indexes}")

    def plot(self):
        # Plot the anomaly values and scores for each column
        for col in self.anomaly_dict:
            plt.figure(figsize=(10, 5))
            plt.scatter(self.anomaly_dict[col]['anomaly_values'], self.anomaly_dict[col]['scores'], color='red')
            plt.scatter(self.non_anomaly_dict[col]['values'], self.non_anomaly_dict[col]['scores'], color='blue')
            plt.xlabel(col)
            plt.ylabel('Anomaly Score')
            plt.show()

    def detect_anomalies(self, input_data, columns):
        self.fit(input_data, columns)

        # Create a dataframe with the rows of the unique indexes that were found to be anomalies in any of the columns
        anomaly_df = input_data.iloc[list(set([a for b in self.anomaly_indexes.values() for a in b]))]

        # Create a dataframe without the rows of the unique indexes that were found to be anomalies in any of the columns
        clean_data = input_data.drop(anomaly_df.index)

        return anomaly_df, clean_data, self.anomaly_dict


# valid_df, removed_df = remove_incorrect_values(data)
# pivot_df = convert_to_long(valid_df)


# # Create an instance of the DetectAnomalies class
# anomaly_detector = DetectAnomalies(contamination=0.005)


def impute_with0(input_df, columns_to_impute_0):
    # Impute the missing values with 0
    imputed_df = input_df.copy()
    imputed_df[columns_to_impute_0] = imputed_df[columns_to_impute_0].fillna(0)
    return imputed_df


# KNN imputer to impute missing values
class ImputeKNN:
    def __init__(self, data, columns_to_impute, n_neighbors=5, weights='uniform', metric='nan_euclidean'):
        self.data = data.copy()
        self.columns_to_impute = columns_to_impute
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric)

    def impute(self):
        # Fit the KNN imputer on the selected columns of the dataframe
        return pd.DataFrame(self.imputer.fit_transform(self.data[self.columns_to_impute]), columns = self.columns_to_impute)

    def join2full(self, data):
        # Replace the Nan values with the imputed values from impute()
        imputed_data = self.impute()
        
        # Set the indexes of imputed_data to the indexes of original data
        imputed_data.index = data.index
        
        # Join the imputed data to the original data
        columns_to_keep = [col for col in data.columns if col not in self.columns_to_impute]
        data = data[columns_to_keep].join(imputed_data)
        
        return data

# Iterative imputer to impute missing values
class ImputeIterative:
    def __init__(self, data, columns_to_impute, estimator = BayesianRidge(), max_iter = 10, initial_strategy = 'mean', imputation_order='ascending', tol = 0.001):
        self.data = data.copy()
        self.columns_to_impute = columns_to_impute
        self.estimator = estimator
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.tol = tol
        self.imputer = IterativeImputer(estimator = self.estimator, max_iter = self.max_iter, initial_strategy = self.initial_strategy, imputation_order = self.imputation_order, tol = self.tol)

    def impute(self):
        # Fit the KNN imputer on the selected columns of the dataframe
        return pd.DataFrame(self.imputer.fit_transform(self.data[self.columns_to_impute]), columns = self.columns_to_impute)

    def join2full(self, data):
        # Replace the Nan values with the imputed values from impute()
        imputed_data = self.impute()
        
        # Set the indexes of imputed_data to the indexes of original data
        imputed_data.index = data.index
        
        # Join the imputed data to the original data
        columns_to_keep = [col for col in data.columns if col not in self.columns_to_impute]
        data = data[columns_to_keep].join(imputed_data)
        
        return data

# Class to scale the data
class Scaler:
    """
    Variables containing 'mood': No scaling is required for the target variable as it is already on a scale of 1-10.
    Variables containing 'circumplex.arousal' or 'circumplex.valence': These variables have a range between -2 to 2 and are centered around 0, 
    making them suitable for StandardScaler.
    Variables containing 'activity': This variable has a range between 0 to 1 and is also suitable for MinMaxScaler.
    Variables containing 'screen': This variable has a wide range and may contain outliers, making it suitable for RobustScaler.
    Variables containing 'call' and 'sms': These variables are binary and do not require scaling.
    Variables containing 'appCat.*': These variables represent the duration of usage of different types of apps and may have a wide range and outliers, 
    making them suitable for RobustScaler.
    """

    def __init__(self):
        self.scalers = {}

    def fit_transform(self, X):
        """
        Fit and transform the input data frame X based on the scaling instructions.
        """
        for col in X.columns:
            if 'mood' in col:
                continue
            elif 'circumplex.arousal' in col or 'circumplex.valence' in col:
                scaler = StandardScaler()
                X[col] = scaler.fit_transform(X[[col]])
                self.scalers[col] = scaler
            elif 'activity' in col:
                scaler = MinMaxScaler()
                X[col] = scaler.fit_transform(X[[col]])
                self.scalers[col] = scaler
            elif 'screen' in col:
                scaler = RobustScaler()
                X[col] = scaler.fit_transform(X[[col]])
                self.scalers[col] = scaler
            elif 'call' in col or 'sms' in col:
                continue
            elif 'appCat' in col:
                scaler = RobustScaler()
                X[col] = scaler.fit_transform(X[[col]])
                self.scalers[col] = scaler
            else:
                continue
        return X

    def transform(self, X):
        """
        Transform the input data frame X based on the scaling instructions.
        """
        for col in X.columns:
            if 'mood' in col:
                continue
            elif 'circumplex.arousal' in col or 'circumplex.valence' in col:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError("Scaler not fitted for column: {}".format(col))
                X[col] = scaler.transform(X[[col]])
            elif 'activity' in col:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError("Scaler not fitted for column: {}".format(col))
                X[col] = scaler.transform(X[[col]])
            elif 'screen' in col:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError("Scaler not fitted for column: {}".format(col))
                X[col] = scaler.transform(X[[col]])
            elif 'call' in col or 'sms' in col:
                continue
            elif 'appCat' in col:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError("Scaler not fitted for column: {}".format(col))
                X[col] = scaler.transform(X[[col]])
            else:
                continue
        return X

