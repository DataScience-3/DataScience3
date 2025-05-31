import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, MinMaxScaler

def scale_features(input_path, output_path):
    # Read CSV file
    df = pd.read_csv(input_path)

    # Columns to apply log scaling (use log1p for zeros)
    log_cols = ['nb_hyperlinks', 'web_traffic', 'longest_word_path', 'length_url',
                'ratio_extRedirection', 'longest_words_raw', 'avg_word_path']
    for col in log_cols:
        df[col] = np.log1p(df[col])

    # Apply Min-Max Scaler: safe_anchor
    minmax_scaler = MinMaxScaler()
    df[['safe_anchor']] = minmax_scaler.fit_transform(df[['safe_anchor']])

    # Apply RobustScaler: nb_www and domain_age_created
    robust_scaler = RobustScaler()
    df[['nb_www', 'domain_age_created']] = robust_scaler.fit_transform(df[['nb_www', 'domain_age_created']])

    # Apply Z-score scaling (StandardScaler): page_rank
    zscore_scaler = StandardScaler()
    df[['page_rank']] = zscore_scaler.fit_transform(df[['page_rank']])

    # Apply QuantileTransformer: links_in_tags, ratio_digits_url, ratio_extHyperlinks, ratio_intHyperlinks
    quantile_transformer = QuantileTransformer(random_state=42)
    df[['links_in_tags', 'ratio_digits_url', 'ratio_extHyperlinks', 'ratio_intHyperlinks']] = quantile_transformer.fit_transform(
        df[['links_in_tags', 'ratio_digits_url', 'ratio_extHyperlinks', 'ratio_intHyperlinks']]
    )

    print(df.head())

    # Secondary scaling
    # Columns to apply Z-score standardization (if not already log scaled, etc.)
    numeric_scaled_cols = ['web_traffic', 'nb_hyperlinks', 'length_url', 'longest_words_raw', 'longest_word_path']

    # Apply StandardScaler to only those columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_scaled_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=numeric_scaled_cols, index=df.index)

    # Keep other columns as is
    other_cols = [col for col in df.columns if col not in numeric_scaled_cols]
    df_other = df[other_cols]

    # Merge scaled columns and other columns
    final_df = pd.concat([df_scaled, df_other], axis=1)

    # Save the final matrix to CSV file
    final_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    scale_features('selected_features_intersection.csv', 'final_matrix.csv')
