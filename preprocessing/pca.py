import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CSV 파일 읽기
df = pd.read_csv("final_matrix.csv")

# PCA를 전체 component로 피팅하여 누적 설명 분산 구하기
pca_full = PCA()
pca_full.fit(df)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 누적 설명 분산 0.95 이상이 되는 최소 컴포넌트 수 결정
num_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components selected to reach 95% variance: {num_components}")

# 선택된 컴포넌트 수를 이용해 PCA 재실행
pca = PCA(n_components=num_components)
principalComponents = pca.fit_transform(df)

# 누적 설명 분산 그래프 그리기
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

# 2차원 산점도: 첫번째와 두번째 주성분
if num_components >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], alpha=0.5, c='blue')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot (First 2 Components)")
    plt.grid(True)
    plt.show()
    
# 10개의 주성분 선택
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(10)])

# PCA 결과를 CSV 파일로 저장
principalDf.to_csv("pca_result.csv", index=False)