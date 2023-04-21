"""
main.py

Data Mining Techniques - Advanced Assigment 1

- Runs all algorithms created for the experiment. 
- Uses command line arguments for choosing the dataset, the algorithm and
    how the algorithm should be used.
"""

# ------------------------------- Imports --------------------------------------

from DataPreparation import datacleaning

import pandas as pd
import argparse
import time

if __name__ == '__main__':
    
    # --------------------------- Load and Preprocess the Data --------------------------------
    
    data = 'Data/dataset_mood_smartphone.csv'
    cleaned_data = datacleaning(data)

    # EDA
    
    # Choose features

    # Impute the data 

    # --------------------------- Run algorithms -----------------------------

    algorithms = [
            LSTM, 
            Forest
        ]
    for Algorithm in algorithms:
        experiment = Algorithm(clean_data)
        experiment.run()
        qual = experiment.quality()
        if qual > best_qual:
            best_qual = qual
            print(best_qual)
            # TODO: keep best qual in csv
            # TODO: visualize results 

    # --------------------------- Compare results -----------------------------

    

    

    