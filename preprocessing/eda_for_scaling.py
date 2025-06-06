import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('selected_features_intersection.csv')

# Generate histogram for each column
for col in df.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col].dropna(), bins=20, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

