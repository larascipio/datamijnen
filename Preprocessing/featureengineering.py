# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Preprocessing.datacleaning import remove_incorrect_values, convert_to_wide, impute_with0, ImputeKNN, ImputeIterative


# Feature Engineering
def feature_engineering(df):
    """ Feature engineering for time series data """
    
    # 1. Time-based features:
    num_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in ['id', 'date']]
    
    # Ensure 'date' column is a datetimelike object
    df['date'] = pd.to_datetime(df['date']) 
    
    # Make variables for the hour, day, month, year, and weekday
    df['day'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    
    # Make a variable for weekday 
    df['weekday'] = df['date'].dt.strftime('%A')
    
    # Make a variable for weekend based on the weekday
    df['weekend'] = np.where(df['weekday'].isin(['Saturday', 'Sunday']), 1, 0)
    
    # Make a variable for the day of the year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Make a variable for the week of the year
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Make a variable for the day of the month
    df['day_of_month'] = df['date'].dt.day
    
    # Make a variable for the week of the month
    df['week_of_month'] = np.ceil(df['date'].dt.day/7).astype(int)
    
    # Create seasonality features
    df['season'] = np.where(df['date'].dt.month.isin([1, 2, 12]), 'winter',
                   np.where(df['date'].dt.month.isin([3, 4, 5]), 'spring',
                   np.where(df['date'].dt.month.isin([6, 7, 8]), 'summer',
                   np.where(df['date'].dt.month.isin([9, 10, 11]), 'fall', 'unknown'))))
    
    # Name the app.Cats here, otherwise all newly created columns starting with app will be used
    app_cats = [c for c in df.columns if c.startswith('appCat')]

    # 2. Statistical features
    window_size = [4, 5, 6]
            
    # Fill a dataframe with the new columns
    new_cols = []
    for c in num_cols:
        df[f'{c}_time_since_last_activity'] = df['date'].diff().dt.days
        
        # Loop through the window sizes
        for window in window_size:
            # Make sure that if training data is running empty, and the window size is larger than the number of observations, the window size is set to the number of observations
            window = min(window, len(df))
            # Make a variable for the rolling mean
            rolling_mean = pd.Series(df.groupby('id')[c].rolling(window).mean().values).reset_index(drop=True)
            rolling_mean.name = f'{c}_rolling_mean_{window}'
            new_cols.append(rolling_mean)
            
            # Make a variable for the rolling standard deviation
            rolling_std = pd.Series(df.groupby('id')[c].rolling(window).std().values).reset_index(drop=True)
            rolling_std.name = f'{c}_rolling_std_{window}'
            new_cols.append(rolling_std)
        
            # Make a variable for the rolling minimum
            rolling_min = pd.Series(df.groupby('id')[c].rolling(window).min().values).reset_index(drop=True)
            rolling_min.name = f'{c}_rolling_min_{window}'
            new_cols.append(rolling_min)
            
            # Make a variable for the rolling maximum
            rolling_max = pd.Series(df.groupby('id')[c].rolling(window).max().values).reset_index(drop=True)
            rolling_max.name = f'{c}_rolling_max_{window}'
            new_cols.append(rolling_max)
            
            # Make a variable for the rolling median
            rolling_median = pd.Series(df.groupby('id')[c].rolling(window).median().values).reset_index(drop=True)
            rolling_median.name = f'{c}_rolling_median_{window}'
            new_cols.append(rolling_median)
            
            # Make a variable for the rolling sum
            rolling_sum = pd.Series(df.groupby('id')[c].rolling(window).sum().values).reset_index(drop=True)
            rolling_sum.name = f'{c}_rolling_sum_{window}'
            new_cols.append(rolling_sum)
            
            # Make a variable for the rolling count
            rolling_count = pd.Series(df.groupby('id')[c].rolling(window).count().values).reset_index(drop=True)
            rolling_count.name = f'{c}_rolling_count_{window}'
            new_cols.append(rolling_count)
            
            # Make a variable for the rolling skew
            rolling_skew = pd.Series(df.groupby('id')[c].rolling(window).skew().values).reset_index(drop=True)
            rolling_skew.name = f'{c}_rolling_skew_{window}'
            new_cols.append(rolling_skew)

            # Make a variable for the rolling kurtosis
            rolling_kurtosis = pd.Series(df.groupby('id')[c].rolling(window).kurt().values).reset_index(drop=True)
            rolling_kurtosis.name = f'{c}_rolling_kurtosis_{window}'
            new_cols.append(rolling_kurtosis)
            
            # Make a variable for the rolling quantile (0.5)
            rolling_quantile_50 = pd.Series(df.groupby('id')[c].rolling(window).quantile(0.5).values).reset_index(drop=True)
            rolling_quantile_50.name = f'{c}_rolling_quantile_50_{window}'
            new_cols.append(rolling_quantile_50)
            
            # Make a variable for the rolling quantile (0.25)
            rolling_quantile_25 = pd.Series(df.groupby('id')[c].rolling(window).quantile(0.25).values).reset_index(drop=True)
            rolling_quantile_25.name = f'{c}_rolling_quantile_25_{window}'
            new_cols.append(rolling_quantile_25)
            
            # Make a variable for the rolling quantile (0.75)
            rolling_quantile_75 = pd.Series(df.groupby('id')[c].rolling(window).quantile(0.75).values).reset_index(drop=True)
            rolling_quantile_75.name = f'{c}_rolling_quantile_75_{window}'
            new_cols.append(rolling_quantile_75)

            # Make a variable for the lag feature
            lag = pd.Series(df.groupby('id')[c].shift(window)).reset_index(drop=True)
            lag.name = f'{c}_lag_{window}'
            new_cols.append(lag)
    

    # Put the new_cols into a dataframe
    new_cols = pd.DataFrame(new_cols).T

    # Concatenate the new_cols to the original dataframe
    df = pd.concat([df, new_cols], axis=1)
    
    # Remove the rows where the rolling features are NaN if there are any
    if df.isnull().values.any():
        df = df.dropna()
    
    # 3. Domain-specific features:
    
    # Add a column for the most frequent app
    most_freq_app = pd.Series(df[app_cats].idxmax(axis=1)).reset_index(drop=True)
    most_freq_app.name = 'most_freq_app'
    
    # Label encode the categorical variables
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['id', 'date']]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])

    # Add a column for total app usage per day per id in time
    df['app_usage'] = df[app_cats].sum(axis=1)
    
    # Count the number of days per id and join it to the dataframe as a new column on id
    num_days = df.groupby('id')['day'].count()
    # Join the number of days to the dataframe on id
    df = df.join(num_days, on='id', rsuffix='_num_days')

    # Add a column for the total number of apps used per day per id in time based on which values are greater than 0
    df['num_apps_used'] = (df[app_cats] > 0).sum(axis=1)
    
    # Duration of social interaction
    social_apps = ['appCat.communication', 'appCat.social']
    df['social_interaction_duration'] = df[social_apps].sum(axis=1)
    
    # Duration of 'fun' apps and office apps
    # appCat.entertainment is most correlated with 0.336546500751683 from appCat.office
    EntOff_apps = ['appCat.entertainment', 'appCat.office']
    df['EntOff_interaction_duration'] = df[EntOff_apps].sum(axis=1)
    
    # Duration of 'work' apps and communication apps
    # appCat.finance is most correlated with 0.29156044051631286 from appCat.communication
    FinCom_apps = ['appCat.finance', 'appCat.communication']
    df['FinCom_interaction_duration'] = df[FinCom_apps].sum(axis=1)
    
    # Duration of 'all dopamine' apps
    # appCat.game is most correlated with 0.27441381254710484 from appCat.entertainment
    EntGam_apps = ['appCat.entertainment', 'appCat.game']
    df['EntGam_interaction_duration'] = df[EntGam_apps].sum(axis=1)
    
    # appCat.other is most correlated with 0.22075455275327413 from appCat.entertainment
    OthEnt_apps = ['appCat.other', 'appCat.entertainment']
    df['OthEnt_interaction_duration'] = df[OthEnt_apps].sum(axis=1)
    
    # appCat.travel is most correlated with 0.15521002650076768 from appCat.other
    OthTra_apps = ['appCat.other', 'appCat.travel']
    df['OthTra_interaction_duration'] = df[OthTra_apps].sum(axis=1)

    # appCat.utilities is most correlated with 0.23926868329595047 from appCat.weather
    UtWea_apps = ['appCat.utilities', 'appCat.weather']
    df['UtWea_interaction_duration'] = df[UtWea_apps].sum(axis=1)
    
    # appCat.weather is most correlated with 0.27312767718313147 from appCat.communication
    WeaCom_apps = ['appCat.weather', 'appCat.communication']
    df['WeaCom_interaction_duration'] = df[WeaCom_apps].sum(axis=1)

    # Create a new feature that represents the total time spent on social apps vs. work apps
    df['social_time'] = df[['appCat.communication', 'appCat.social','appCat.entertainment']].sum(axis=1)
    df['work_time'] = df[['appCat.office', 'appCat.finance']].sum(axis=1)

    # Create cross-product features for all numeric columns
    new_cross_cols = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            new_col_name = f"{num_cols[i]}_{num_cols[j]}"
            new_col = df[num_cols[i]] * df[num_cols[j]]
            new_col.name = new_col_name
            new_cross_cols.append(new_col)

    # Add the new cross-product features to the DataFrame
    df = pd.concat([df, *new_cross_cols], axis=1)
    
    # Return the dataframe with the new features
    return df

# activity is most correlated with 0.277739650358384 from mood
# appCat.builtin is most correlated with 0.11764174086734279 from appCat.communication
# appCat.communication is most correlated with 0.5137391505411122 from screen
# appCat.entertainment is most correlated with 0.336546500751683 from appCat.office
# appCat.finance is most correlated with 0.29156044051631286 from appCat.communication
# appCat.game is most correlated with 0.27441381254710484 from appCat.entertainment
# appCat.office is most correlated with 0.336546500751683 from appCat.entertainment
# appCat.other is most correlated with 0.22075455275327413 from appCat.entertainment
# appCat.social is most correlated with 0.3679284176282209 from screen
# appCat.travel is most correlated with 0.15521002650076768 from appCat.other
# appCat.unknown is most correlated with 0.39516287604927275 from screen
# appCat.utilities is most correlated with 0.23926868329595047 from appCat.weather
# appCat.weather is most correlated with 0.27312767718313147 from appCat.communication
# call is most correlated with nan from appCat.builtin
# circumplex.arousal is most correlated with 0.23007562860534975 from circumplex.valence
# circumplex.valence is most correlated with 0.4711588597771999 from mood
# mood is most correlated with 0.4711588597771999 from circumplex.valence
# screen is most correlated with 0.5137391505411122 from appCat.communication
# sms is most correlated with nan from appCat.builtin