import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    # Read the generated dirty data file
    df = pd.read_csv(input_path)

    print("Original data frame shape:", df.shape)

    # Define a condition function to immediately drop rows for the 5 columns with inserted dirty data
    cols_to_clean = ['https_token', 'length_url', 'phish_hints', 'nb_at', 'page_rank']

    def is_valid_integer(x):
        if pd.isna(x):
            return False
        # '+' and '%' are dropped
        if x in ['+', '%']:
            return False
        try:
            # Convert value to float
            num = float(x)
        except Exception:
            return False
        # Return True if integer, False otherwise
        return num.is_integer()

    # Remove invalid rows for each column
    for col in cols_to_clean:
        df = df[df[col].apply(is_valid_integer)]

    # Remove rows where domain_age and domain_registration_length are negative
    df = df[(df['domain_registration_length'] >= 0)]

    # Data value change
    # Map status column: legitimate -> 0, phishing -> 1
    df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

    # Remove url column (Unusable data)
    df = df.drop('url', axis=1)

    # Save the final cleaned data
    df.to_csv(output_path, index=False)
    print("Cleaned data frame shape:", df.shape)

if __name__ == '__main__':
    clean_data('dataset_phishing_dirty.csv', 'dataset_phishing_clean.csv')