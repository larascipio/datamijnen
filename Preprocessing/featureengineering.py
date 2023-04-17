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
    
    # Add a column for morning or evening
    df['morning'] = np.where(df['hour'] < 12, 1, 0)

    # Add a column for weekend or weekday
    df['weekend'] = np.where(df['weekday'] >= 5, 1, 0)
    
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
    
    # Add a column for total app usage
    app_cats = ['appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
    df['app_usage'] = df[app_cats].sum(axis=1)

    # Add a column for total app usage per day
    df['app_usage_per_day'] = df.groupby('id')['app_usage'].transform('mean')

    # Add a column for total call duration per day
    df['call_duration_per_day'] = df.groupby(['id', 'date'])['call'].transform('sum')

    # Add a column for total call duration per week
    df['call_duration_per_week'] = df.groupby(['id', 'weekend'])['call'].transform('sum')

    # Add a column for total activity per day
    df['activity_per_day'] = df.groupby('id')['activity'].transform('mean')

    # Add a column for total activity per week
    df['activity_per_week'] = df.groupby(['id', 'weekend'])['activity'].transform('mean')

    # Add a column for average arousal per day
    df['arousal_per_day'] = df.groupby('id')['circumplex.arousal'].transform('mean')

    # Add a column for average valence per day
    df['valence_per_day'] = df.groupby('id')['circumplex.valence'].transform('mean')

    # Add a column for average mood per day
    df['mood_per_day'] = df.groupby('id')['mood'].transform('mean')

    # Return the dataframe with the new features
    return df
    
