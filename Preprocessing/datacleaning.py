# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
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

def convert_to_long(input_df):
    
    # Split the time into date and time
    input_df['date'] = input_df['time'].dt.date
    input_df['time'] = input_df['time'].dt.time

    # Create a new long format dataframe where each row is a unique combination of id and date and the variables are the mean of the values for that day 
    pivot_df = input_df.pivot_table(index=['id', 'date'], 
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
            plt.ylabel('anomaly score')
            plt.show()

    def detect_anomalies(self, input_data, columns):
        self.fit(input_data, columns)

        # Create a dataframe with the rows of the unique indexes that were found to be anomalies in any of the columns
        anomaly_df = input_data.iloc[list(set([a for b in self.anomaly_indexes.values() for a in b]))]

        # Create a dataframe without the rows of the unique indexes that were found to be anomalies in any of the columns
        clean_data = input_data.drop(anomaly_df.index)

        return anomaly_df, clean_data, self.anomaly_dict



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
        return pd.DataFrame(self.imputer.fit_transform(self.data[self.columns_to_impute]), columns=self.columns_to_impute)

    def join2full(self, data):
        # Replace the Nan values with the imputed values from impute()
        data[self.columns_to_impute] = self.impute()
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
        data[self.columns_to_impute] = self.impute()
        return data


# Class to scale the data
class Scaler:
    """
    mood: No scaling is required for the target variable as it is already on a scale of 1-10.
    circumplex.arousal and circumplex.valence: These variables have a range between -2 to 2 and are centered around 0, 
    making them suitable for StandardScaler.
    activity: This variable has a range between 0 to 1 and is also suitable for MinMaxScaler.
    screen: This variable has a wide range and may contain outliers, making it suitable for RobustScaler.
    call and sms: These variables are binary and do not require scaling.
    appCat.*: These variables represent the duration of usage of different types of apps and may have a wide range and outliers, 
    making them suitable for RobustScaler.
    """
    def __init__(self, data, columns_to_scale):
        self.data = data
        self.columns_to_scale = columns_to_scale
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }

    def transform(self, X):
        X = X.copy()
        for feature in self.columns_to_scale:
            if feature in ['circumplex.arousal', 'circumplex.valence']:
                scaler = self.scalers['StandardScaler']
            elif feature == 'activity':
                scaler = self.scalers['MinMaxScaler']
            else:
                scaler = self.scalers['RobustScaler']

            X[feature] = scaler.fit_transform(X[[feature]])
        return X

    def fit_transform(self):
        X = self.data.drop(['mood'], axis=1)
        y = self.data['mood']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
        X_train = self.transform(X_train)
        X_test = self.transform(X_test)
        return X_train, X_test, y_train, y_test





# # Imputing with KNNImputer
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import MinMaxScaler

# #Define a subset of the dataset
# df_knn = new_df.filter(['circumplex.valence','appCat.utilities','circumplex.arousal'], axis=1).copy()

# # Define KNN imputer and fill missing values
# knn_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)

# fig = plt.Figure()
# null_values = new_df['appCat.utilities'].isnull() 
# fig = df_knn_imputed.plot(x='circumplex.valence', y='appCat.utilities', kind='scatter', c=null_values, cmap='winter', title='KNN Imputation', colorbar=False)
# plt.show()



# # Imputing with MICE
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.linear_model import BayesianRidge

# df_mice = new_df.filter(['circumplex.valence','appCat.utilities','circumplex.arousal'], axis=1).copy()

# # Define MICE Imputer and fill missing values
# mice_imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
# df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice), columns=df_mice.columns)

# fig = plt.Figure()
# null_values = new_df['appCat.utilities'].isnull() 
# fig = df_mice_imputed.plot(x='circumplex.valence', y='appCat.utilities', kind='scatter', c=null_values, cmap='winter', title='KNN Imputation', colorbar=False)
# plt.show()





# # Count the number of missing values in each column per id in new_df
# missing_values_per_id = new_df.groupby('id').apply(lambda x: x.isnull().sum()).iloc[:, 2:]

# pd.DataFrame(missing_values_per_id).head(10)

# # Calculate the percentage of missing values in each column per id in new_df
# missing_values_per_idP = new_df.groupby('id').apply(lambda x: x.isnull().sum() / x.shape[0] * 100).iloc[:, 2:]
# pd.DataFrame(missing_values_per_idP)

# # for each id separately, plot the percentage of missing values in each column where each bar represents a column
# for id in missing_values_per_idP.index:
#     missing_values_per_idP.loc[id].plot(kind='bar', title=f'Percentage of missing values for id {id}')
#     plt.show()



