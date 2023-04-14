# Import libraries
import numpy as np
import pandas as pd

# Feature Engineering
def feature_engineering(df):
    """ Feature engineering for time series data """
    
    # Make variables for the hour, day, month, year, and weekday
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.weekday
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['weekday'] = df['time'].dt.strftime('%A')
    
    # Make a variable for the day of the year
    df['day_of_year'] = df['time'].dt.dayofyear
    
    # Make a variable for the week of the year
    df['week_of_year'] = df['time'].dt.weekofyear
    
    # Make a variable for the day of the month
    df['day_of_month'] = df['time'].dt.day
    
    # Make a variable for the week of the month
    df['week_of_month'] = df['time'].dt.week
    
    # Make a variable for the quarter of the year
    df['quarter'] = df['time'].dt.quarter
    
    # Make a variable for the lag, which is the value from the previous hour
    df['lag'] = df['value'].shift(1)
    
    # Make a variable for the rolling mean, which is the average value over the last 7 days
    df['rolling_mean'] = df['value'].rolling(7, min_periods=1).mean()
    
    # Make a variable for the rolling standard deviation, which is the standard deviation of the values over the last 7 days
    df['rolling_std'] = df['value'].rolling(7, min_periods=1).std()
    
    # Make a variable for the rolling minimum, which is the minimum value over the last 7 days
    df['rolling_min'] = df['value'].rolling(7, min_periods=1).min()
    
    # Make a variable for the rolling maximum, which is the maximum value over the last 7 days
    df['rolling_max'] = df['value'].rolling(7, min_periods=1).max()
    
    # Make a variable for the rolling median, which is the median value over the last 7 days
    df['rolling_median'] = df['value'].rolling(7, min_periods=1).median()
    
    # Make a variable for the rolling sum, which is the sum of the values over the last 7 days
    df['rolling_sum'] = df['value'].rolling(7, min_periods=1).sum()
    
    # Make a variable for the rolling count, which is the number of values over the last 7 days
    df['rolling_count'] = df['value'].rolling(7, min_periods=1).count()
    
    # Make a variable for the rolling skew, which is the skew of the values over the last 7 days
    df['rolling_skew'] = df['value'].rolling(7, min_periods=1).skew()
    
    # Make a variable for the rolling kurtosis, which is the kurtosis of the values over the last 7 days
    df['rolling_kurtosis'] = df['value'].rolling(7, min_periods=1).kurt()
    
    # Make a variable for the rolling quantile, which is the quantile of the values over the last 7 days
    df['rolling_quantile'] = df['value'].rolling(7, min_periods=1).quantile(0.5)
    
    # Make a variable for the number of days since the last time the value was above 100
    df['days_since_above_100'] = (df['value'] > 100).astype(int).groupby((df['value'] > 100).astype(int).diff().ne(0).cumsum()).cumsum()
    
    # Make a variable for the number of days since the last time the value was below 100
    df['days_since_below_100'] = (df['value'] < 100).astype(int).groupby((df['value'] < 100).astype(int).diff().ne(0).cumsum()).cumsum()
    
    # Return the dataframe with the new features
    return df
    
