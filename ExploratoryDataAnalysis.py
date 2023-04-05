from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

# Import as Dataframe
df = pd.read_csv('/Users/gast/Desktop/VS code/Artificial Intelligence Projects/Data Mining Techniques/Data/dataset_mood_smartphone.csv')
df.head()
list(df.columns)
df = df.iloc[: , 1:]


# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x = df.index, y = df.value, title='Values over time.')    

# Counts of values recorded per month
pd.to_datetime(df["time"]).dt.to_period('M').value_counts().sort_index()


plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))

# Specify how our lines should look
ax.plot(df.time, df.values, color='tab:orange', label='Windspeed')

# Same as above
ax.set_xlabel('j')
ax.set_ylabel('joe')
ax.set_title('..')
ax.grid(True)
ax.legend(loc='upper left')
plt.show()

