# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Preprocessing.datacleaning import remove_incorrect_values, convert_to_wide, impute_with0, ImputeKNN, ImputeIterative

# Feature Engineering
def feature_engineering(df_original):
    """ Feature engineering for time series data """
    
    # Save original dataframe 
    df = df_original.copy()
    
    # Reset the index
    df = df.reset_index(drop=True)
    
    # 1. Time-based features:
    # num_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in ['id', 'date']]
    
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
    
    # Make a variable for the start of the month
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    
    # Make a variable for the start of the month
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    
    # Calculate the time difference between each consecutive row 
    df['time_diff'] = (df['date'].diff().dt.total_seconds() / (60*60)).fillna(0)
    
    # Random gaussian noise
    def random_noise(dataframe):
        return np.random.normal(scale=1.5, size=(len(dataframe),))

    # Name the app.Cats here, otherwise all newly created columns starting with app will be used
    app_cats = [c for c in df.columns if c.startswith('appCat')]
    
    # 2. Statistical features
    
    # Rolling variables for mood
    windows = [4, 5, 6]
    for window in windows:
        # Rolling mean for mood
        df['mood_roll_mean_' + str(window)] = df.groupby(["mood", "id"])['mood']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2, win_type="triang").mean()) + random_noise(df)
        # Rolling std for mood
        df['mood_roll_std_' + str(window)] = df.groupby(["mood", "id"])['mood']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2, win_type="triang").std()) + random_noise(df)
        # Rolling sum for mood
        df['mood_roll_sum_' + str(window)] = df.groupby(["mood", "id"])['mood']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2, win_type="triang").sum()) + random_noise(df)                                                                                                                                                                                                                                                          
    
    # Lag features for mood
    lags = [4, 5, 6]        
    for lag in lags:
        df['mood_lag_' + str(lag)] = df.groupby(["mood", "id"])['mood'].transform(
            lambda x: x.shift(lag)) + random_noise(df)

    # Exponentially Weighted Mean Features
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    for alpha in alphas:
        for lag in lags:
            df['mood_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                df.groupby(["mood", "id"])['mood'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    
    # 2. Categorical features: convert to one-hot encoding
    cols_to_encode = [object for object in df.select_dtypes(include='object').columns.tolist() if object != 'id']
     
    # create an instance of the OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False)

    # fit and transform the selected columns using the OneHotEncoder
    encoded_cols = ohe.fit_transform(df[cols_to_encode])

    # create a DataFrame from the encoded columns with column names
    encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(cols_to_encode))

    # concatenate the encoded columns with the original DataFrame
    df = pd.concat([df.drop(cols_to_encode, axis=1), encoded_df], axis=1)
        
    # 3. Domain-specific features:

    # Add a column for the most frequent app
    df['most_freq_app'] = df[app_cats].idxmax(axis=1)
    df['most_freq_app'] = LabelEncoder().fit_transform(df['most_freq_app'])
       
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

