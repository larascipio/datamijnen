# Import the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Import as Dataframe
df = pd.read_csv('./Data/dataset_mood_smartphone.csv')
df.head()

# Drop unnecessary columns
data =  df.drop(['Unnamed: 0'], axis=1)
    
# Make sure the 'time' column is of type datetime
data['time'] = pd.to_datetime(data['time'])


# Deleting incorrect values from the dataset 

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
    valid_df = data[data.apply(lambda x: valid_ranges[x.variable][0] <= x.value <= valid_ranges[x.variable][1] if not pd.isnull(x.value) else True, axis=1)]
    
    # Create a separate dataframe for the removed rows
    removed_df = input_data[~input_data.index.isin(valid_df.index)]
    
    # Return the valid and removed dataframes
    return valid_df, removed_df

valid_df, removed_df = remove_incorrect_values(data)


# Split the time into date and time
valid_df['date'] = valid_df['time'].dt.date
valid_df['time'] = valid_df['time'].dt.time

# Create a new long format dataframe where each row is a unique combination of id and date and the variables are the mean of the values for that day 
new_df = valid_df.pivot_table(index=['id', 'date'], 
                          columns='variable', 
                          values='value', 
                          aggfunc='mean').reset_index()

# Check the percentage of missing values in the new dataframe for each column
new_df.isnull().sum() / len(new_df) * 100

# Drop columns that contain more than 50% missing values
new_df['date'].min()
new_df['date'].max()


import missingno as msno

msno.bar(new_df)
plt.show()
msno.matrix(new_df)















# Isolation Forest Algorithm
def run_isolation_forest(model_data: pd.DataFrame, 
                         column_name: str,
                         contamination = 0.005, 
                         n_estimators= 200, 
                         max_samples = 0.7) -> pd.DataFrame:
    
    IF = (IsolationForest(random_state = 0,
                          contamination = contamination,
                          n_estimators = n_estimators,
                          max_samples = max_samples))
    
    # Drop any rows with NaN values in the 'value' column and store the result in a new dataframe
    df = model_data.copy()
    df.dropna(subset=[column_name], inplace=True)
    
    # Fit the model
    IF.fit(df[[column_name]])
    
    # Datapoints classified -1 are anomalous (outliers): made 1 here for plotting
    output = pd.Series(IF.predict(df[[column_name]])).apply(lambda x: 1 if x == -1 else 0)
    score = IF.decision_function(df[[column_name]])

    # Convert the array to a dataframe with a column name 'score'
    score_df = pd.DataFrame({'score': score})

    # Concatenate the score_df and output along axis 1
    merged_df = pd.concat([score_df, output], axis=1)
    merged_df.index = df.index

    # Add the id column to the merged_df dataframe
    merged_df = pd.concat([df, merged_df], axis=1)
    
    # Rename the columns
    merged_df.columns = ['id', 'time', 'activity', 'circumplex.arousal', 'circumplex.valence', 'mood', 'score', 'anomaly']
    
    return merged_df




# Number of outliers in activity: 14
# Number of outliers in circumplex.arousal: 10
# Number of outliers in circumplex.valence: 8
# Number of outliers in mood: 27




# Impute missing values 
from sklearn.impute import KNNImputer

# KNN Imputation
def impute_KNN(model_data: pd.DataFrame,
               column_name: str,
               n_neighbors = 10,
               weights = 'uniform',
               metric = 'nan_euclidean') -> pd.DataFrame:
    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors = n_neighbors,
                         weights = weights,
                         metric = metric)
    df_copy = model_data.copy()
    df_filled = imputer.fit_transform(df_copy[[column_name]])
    
    # Convert the array to a dataframe with a column name 'value'
    df_filled = pd.DataFrame(df_filled, columns = ['value'])
    
    # Replace the 'value' column within df with the 'value' column from df_filled
    df_copy['value'] = df_filled['value']

    return df_copy

# Impute missing values of the column 'value' with KNN
knn = impute_KNN(data, 'value')









import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime

np.array(data['variable'])

def preprocess_data(input_data):
    # Label encode the 'variable' column
    encoder = LabelEncoder()
    encoded_var = encoder.fit_transform(np.array(data['variable']))
    
    # Convert the array to a dataframe with a column name 'value'
    df_enc = pd.DataFrame(encoded_var, columns = ['variable'])
    
    # Replace the 'value' column within df with the 'value' column from df_filled
    input_data['variable'] = df_enc['variable']



    encoded_var.columns = encoder.get_feature_names(['variable'])
    input_data = pd.concat([input_data, encoded_var], axis=1)
    input_data = input_data.drop('variable', axis=1)
    
    # Convert the 'time' column to Unix timestamps
    input_data['time'] = pd.to_datetime(input_data['time'])
    input_data['time'] = (input_data['time'] - datetime.datetime(1970,1,1)).dt.total_seconds()
    
    return input_data

preprocess_data(data)





#### BAYESIAN IMPUTATION
import pytensor.tensor as pt

with pm.Model() as model:
    # Priors
    mus = pm.Normal("mus", 0, 1, size=3)
    cov_flat_prior, _, _ = pm.LKJCholeskyCov("cov", n=3, eta=1.0, sd_dist=pm.Exponential.dist(1))
    # Create a vector of flat variables for the unobserved components of the MvNormal
    x_unobs = pm.Uniform("x_unobs", 0, 100, shape=(np.isnan(data.values).sum(),))

    # Create the symbolic value of x, combining observed data and unobserved variables
    x = pt.as_tensor(data.values)
    x = pm.Deterministic("x", pt.set_subtensor(x[np.isnan(data.values)], x_unobs))

    # Add a Potential with the logp of the variable conditioned on `x`
    pm.Potential("x_logp", pm.logp(rv=pm.MvNormal.dist(mus, chol=cov_flat_prior), value=x))
    idata = pm.sample_prior_predictive()
    idata = pm.sample()
    idata.extend(pm.sample(random_seed=120))
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

pm.model_to_graphviz(model)





