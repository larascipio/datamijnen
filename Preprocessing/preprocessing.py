import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Create a function to report the limits of the Z-Score
def print_z_score_limits (df, column_name):
    """ Print the upper and lower limits of the Z-score """
    
    # Compute the limits (3 = threshold: default)
    upper_limit = df[column_name].mean() + 3 * df[column_name].std()
    lower_limit = df[column_name].mean() - 3 * df[column_name].std()
    
    # Round and return the limits
    upper_limit = round(upper_limit, 2)
    lower_limit = round(lower_limit, 2)
    print_this = "Variable Name: " + column_name + " | Upper limit: " + str(upper_limit) + " | Lower limit: " + str(lower_limit)
    return(print_this)


# Isolation forest algorithm

def remove_outliers(df2Bcleaned):
    # Drop any rows with NaN values in the 'value' column
    df2Bcleaned = df2Bcleaned.copy()
    df2Bcleaned.dropna(subset=['value'], inplace=True)

    # Apply z-score method to identify outliers
    df2Bcleaned['z_score'] = (df2Bcleaned['value'] - df2Bcleaned['value'].mean()) / df2Bcleaned['value'].std()
    df2Bcleaned = df2Bcleaned[df2Bcleaned['z_score'].abs() < 3]

    # Apply isolation forest algorithm to identify outliers
    columns_to_scale = ['value']  # Add other columns here to include them in the isolation forest algorithm
    scaler = StandardScaler()
    df2Bcleaned[columns_to_scale] = scaler.fit_transform(df2Bcleaned[columns_to_scale])

    clf = IsolationForest(random_state=0, contamination=0.05)
    clf.fit(df2Bcleaned[columns_to_scale])
    df2Bcleaned['is_outlier'] = clf.predict(df2Bcleaned[columns_to_scale])

    # Remove any data points that are identified as outliers by both methods
    df2Bcleaned = df2Bcleaned[(df2Bcleaned['z_score'].abs() < 3) & (df2Bcleaned['is_outlier'] != -1)]

    # Drop the added columns and return the cleaned data
    df2Bcleaned.drop(['z_score', 'is_outlier'], axis=1, inplace=True)
    return df2Bcleaned



import plotly.express as px
from sklearn.ensemble import IsolationForest


# Drop any rows with NaN values in the 'value' column and store the result in a new dataframe
df = data.copy()
df.dropna(subset=['value'], inplace=True)





model =  IsolationForest(contamination=0.05)
model.fit(df[['value']])
df['outliers']=pd.Series(model.predict(df[['value']])).apply(lambda x: 'yes' if (x == -1) else 'no')
df.query('outliers=="yes"')


df['score'] = model.decision_function(df[['value']])
df['anomaly_value'] = model.predict(df[['value']])
df.head()


outliers = df.loc[df['anomaly_value'] == -1]
outlier_index = list(outliers.index)

#datapoints classified -1 are anomalous
df['anomaly_value'].value_counts()

plt.figure(figsize = (16, 8))
plt.plot(df['value'], marker = '.')
plt.plot(outliers['value'], 'o', color = 'red', label = 'outlier')
plt.title('Detection By Isolation Forest')

#plt.grid()
plt.xlabel('Date')
plt.ylabel('Neutral Current')
plt.legend()
plt.show()