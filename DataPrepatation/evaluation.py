
from datacleaning import create_arima_model, create_lstm_model, create_bsts_model, create_state_space_model


# Compare model performance for imputing missing values

# Make a function to compare the performance of different models for imputing missing values
def compare_imputation_models(train, test, p_values, d_values, q_values):
    """ Compare the performance of different models for imputing missing values """
    
    # Create a dataframe to store the results
    results = pd.DataFrame(columns=['p', 'd', 'q', 'model', 'error'])
    
    # Create a list of models
    models = [('ARIMA', create_arima_model), ('LSTM', create_lstm_model), ('BSTS', create_bsts_model), ('State Space', create_state_space_model)]
    
    # Loop through the models
    for name, model in models:
        
        # Loop through the p values
        for p in p_values:
            
            # Loop through the d values
            for d in d_values:
                
                # Loop through the q values
                for q in q_values:
                    
                    # Try to make predictions
                    try:
                        
                        # Make predictions
                        predictions, error = model(train, test, p, d, q)
                        
                        # Store the results
                        results = results.append({'p': p, 'd': d, 'q': q, 'model': name, 'error': error}, ignore_index=True)
                    
                    # If an error occurs, skip the model
                    except:
                        continue
    
    # Return the results
    return results
