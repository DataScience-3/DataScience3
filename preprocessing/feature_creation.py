import pandas as pd

def create_features(input_path, output_path):
    # Read file
    df = pd.read_csv(input_path)

    # If negative: create domain_age_missing column (set to True)
    df["domain_age_missing"] = (df["domain_age"] < 0).astype(int)

    # Otherwise (value > 0): put original value in domain_age_created column
    df["domain_age_created"] = df["domain_age"].apply(lambda x: x if x >= 0 else 0)

    # Delete original domain_age column
    df = df.drop("domain_age", axis=1)

    # Show result
    print(df.head())

    # Save to file
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    create_features('dataset_phishing_clean.csv', 'dataset_phishing_feature_creation.csv')