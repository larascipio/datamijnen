from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
import pandas as pd

from Preprocessing.datacleaning import print_z_score_limits
from Preprocessing.datacleaning import remove_outliers


# https://www.machinelearningplus.com/time-series/time-series-analysis-python/#google_vignette
# https://www.python-graph-gallery.com/basic-time-series-with-matplotlib?utm_content=cmp-true
# https://unidata.github.io/python-training/workshop/Time_Series/basic-time-series-plotting/


# Import as Dataframe
df = pd.read_csv('./Data/dataset_mood_smartphone.csv')
df.head()

# Drop unnecessary columns
data =  df.drop(['Unnamed: 0'], axis=1)
    
# Make sure the 'time' column is of type datetime
data['time'] = pd.to_datetime(data['time'])


# Inspect the dataset 
print(f"Number of records: {len(data)}")
print(f"Number of attributes: {len(data.columns)}")
print(f"Attribute types:\n{data.dtypes}")
print(f"Ranges of values:\n{data['value'].describe()}")

# Check for missing values
print(f"Number of missing values:\n{data.isnull().sum()}") # 202 missing values in the 'value' column

# Generate descriptive statistics for the dataset
print(data.describe())

# Counts of values recorded per month
pd.to_datetime(data["time"]).dt.to_period('M').value_counts().sort_index()


# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(data, x = data.index, y = data.value, title='Values over time.')    


# Create line plot of mood variable over time for each participant
plt.figure(figsize=(12, 6))
sns.lineplot(x='time', y='value', data=data[data['variable'] == 'mood'], hue='id')
plt.title('Mood over time by participant')
plt.xlabel('Time')
plt.ylabel('Mood')
plt.show()


# Data not normally distributed, a lot of extreme values
# Show histograms - all variables except for the identifier variables
plt.hist(np.array(data['value'].tolist()).astype('float'), density=False, bins=30)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()




# Outlier removal based on Z-score solely (NOT GOOD BECAUSE NOT NORMALLY DISTRIBUTED)
# Print the upper and lower limits
print_z_score_limits(data, "value")
'Variable Name: value | Upper limit: 861.84 | Lower limit: -780.51'

# Filter outliers
sample_z = data[(data['value'] >= 861.84) | (data['value'] <= -780.51)]
print(sample_z.shape) # (1694, 4) --> 1694 outliers according to Z-score


# Outlier removal based on IQR
c_data = remove_outliers(data)

# Density plot of the cleaned data after outlier removal
plt.hist(np.array(c_data['value'].tolist()).astype('float'), density=True, bins=30)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()


