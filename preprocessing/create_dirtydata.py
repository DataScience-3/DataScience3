import pandas as pd
import random
import numpy as np

def create_dirty_data(input_path, output_path, dirty_counts, dirty_values):
    # Read the original dataset
    df = pd.read_csv(input_path)
    idx_list = df.index.tolist()

    # Insert dirty values into specified columns
    for col, count in dirty_counts.items(): # idx_list is the index list of the dataframe
        # If the number of rows is less than count, select with replacement; otherwise, select without replacement.
        if len(idx_list) < count:
            selected_indices = random.choices(idx_list, k=count)
        else:
            selected_indices = random.sample(idx_list, count)

        # Replace the value of the selected row and column with a random dirty value
        for i in selected_indices:
            df.at[i, col] = random.choice(dirty_values)

    # Save the dirty dataset
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    dirty_counts = { 'https_token': 55, 'length_url': 34, 'phish_hints': 46, 'nb_at': 20, 'page_rank': 43 }
    dirty_values = [np.nan, '+', '%', 3.22, -1.48, -9.53, 28.7]
    create_dirty_data('dataset_phishing.csv', 'dataset_phishing_dirty.csv', dirty_counts, dirty_values)
