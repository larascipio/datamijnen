import pandas as pd
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np

# https://www.machinelearningplus.com/time-series/time-series-analysis-python/#google_vignette
# https://www.python-graph-gallery.com/basic-time-series-with-matplotlib?utm_content=cmp-true
# https://unidata.github.io/python-training/workshop/Time_Series/basic-time-series-plotting/

# Goal analysis: to predict the mood of the next day for
# the subjects in the dataset (being the average of the mood values measured during that
# day

# Import as Dataframe
data = pd.read_csv('./Data/dataset_mood_smartphone.csv')
print(data.head())

data['time'] = pd.to_datetime(data['time'])
print(data.head())

# Inspect the dataset
print(f"Number of records: {len(data)}")
print(f"Number of attributes: {len(data.columns)}")
print(f"Attribute types:\n{data.dtypes}")
print(f"Ranges of values:\n{data['value'].describe()}")
print(f"Unique variables:\n{','.join(data.variable.unique())}")

stats = data.describe(include='all')
print(stats)
print(data.hist())
exit()
# Reshape the DataFrame using pivot_table()
pivoted_df = data.pivot_table(columns='variable', values='value')
print(pivoted_df.head())
# Calculate the covariance matrix using cov()

# TODO create table 
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

