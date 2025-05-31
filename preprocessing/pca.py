import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_pca(input_path, output_path, n_components=10):
    # Read CSV file
    df = pd.read_csv(input_path)

    # Fit PCA with all components to get cumulative explained variance
    pca_full = PCA()
    pca_full.fit(df)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Determine the minimum number of components to reach cumulative explained variance of 0.95 or higher
    num_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components selected to reach 95% variance: {num_components}")

    # Re-run PCA with the selected number of components
    pca = PCA(n_components=num_components)
    principalComponents = pca.fit_transform(df)

    # Plot cumulative explained variance graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.axhline(y=0.95, color='red', linestyle='--', label="95% Threshold")
    plt.axvline(x=num_components, color='green', linestyle='--', label=f"{num_components} Components")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2D scatter plot: first and second principal components
    if num_components >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1], alpha=0.5, c='blue')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Scatter Plot (First 2 Components)")
        plt.grid(True)
        plt.show()
        
    # Select n principal components
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(n_components)])

    # Save PCA results to CSV file
    principalDf.to_csv(output_path, index=False)

if __name__ == '__main__':
    run_pca('final_matrix.csv', 'pca_result.csv', n_components=10)