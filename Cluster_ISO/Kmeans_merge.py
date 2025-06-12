import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def cluster_purity(y_true, y_pred):
    df_lab = pd.DataFrame({'true': y_true, 'cluster': y_pred})
    purities = []
    for c in df_lab['cluster'].unique():
        sub = df_lab[df_lab['cluster'] == c]
        purities.append(sub['true'].value_counts().max() / len(sub))
    return np.mean(purities)

def kmeans_pipeline_from_df(
    df,
    k=6,
    max_iter=300,
    target_col='status',
    show_plots=True
):
    #Target 분리
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_scaled = StandardScaler().fit_transform(X)
    X_2d     = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    #KMeans
    km       = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42)
    clusters = km.fit_predict(X_scaled)

    #Evaluate
    sil = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(y, clusters)
    pur = cluster_purity(y, clusters)

    print(f"=== KMeans (k={k}) ===")
    print(f"Silhouette Score   : {sil:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Purity             : {pur:.4f}")

    #Labeling
    mapping     = {cid: np.bincount(y[clusters==cid]).argmax()
                   for cid in range(k)}
    pred_labels = np.array([mapping[c] for c in clusters])
    acc         = accuracy_score(y, pred_labels)
    cm          = confusion_matrix(y, pred_labels)

    print(f"\nMapped Classification Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    if not show_plots:
        return {
            'silhouette': sil,
            'ari': ari,
            'purity': pur,
            'accuracy': acc,
            'confusion_matrix': cm
        }

    #visualization
    legend_true = [
        Line2D([0],[0], marker='o', color='w', label='Legit',
               markerfacecolor='blue', markersize=8),
        Line2D([0],[0], marker='o', color='w', label='Phish',
               markerfacecolor='red',  markersize=8)
    ]

    #True labels
    plt.figure(figsize=(5,4))
    colors = ['blue' if l==0 else 'red' for l in y]
    plt.scatter(X_2d[:,0], X_2d[:,1], c=colors, s=20, alpha=0.6)
    plt.legend(handles=legend_true, title='True')
    plt.title('True Labels (PCA 2D)')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.tight_layout(); plt.show()

    #Cluster assignments
    plt.figure(figsize=(5,4))
    cmap = plt.get_cmap('tab10')
    plt.scatter(X_2d[:,0], X_2d[:,1],
                c=[cmap(c) for c in clusters], s=20, alpha=0.6)
    patches = [ Line2D([0],[0], marker='o', color='w',
                       label=f'Cluster {i}',
                       markerfacecolor=cmap(i), markersize=8)
                for i in range(k) ]
    plt.legend(handles=patches, title='Clusters')
    plt.title(f'Cluster Assignment (k={k})')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.tight_layout(); plt.show()

    #Mapped predictions
    plt.figure(figsize=(5,4))
    colors_pred = ['blue' if l==0 else 'red' for l in pred_labels]
    plt.scatter(X_2d[:,0], X_2d[:,1], c=colors_pred, s=20, alpha=0.6)
    plt.legend(handles=legend_true, title='Predicted')
    plt.title(f'Mapped Predictions (k={k})')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.tight_layout(); plt.show()

    #Confusion matrix heatmap
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit','Phish'],
                yticklabels=['Legit','Phish'])
    plt.title('Confusion Matrix')
    plt.tight_layout(); plt.show()

    return {
        'silhouette': sil,
        'ari': ari,
        'purity': pur,
        'accuracy': acc,
        'confusion_matrix': cm
    }

df=pd.read_csv('final_matrix.csv')
kmeans_pipeline_from_df(df)




