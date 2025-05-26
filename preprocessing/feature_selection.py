import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# 파일 읽기 (feature creation 후 데이터 사용)
df = pd.read_csv("dataset_phishing_feature_creation.csv")

# 타깃(y)와 피처(X) 분리 (status 컬럼이 타깃 데이터)
X = df.drop("status", axis=1)
y = df["status"]

# -------------------------------
# 1. RandomForest 를 이용한 Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
rf_importances = rf.feature_importances_

# Series 생성 후 내림차순 정렬
rf_feat_importance = pd.Series(rf_importances, index=X.columns).sort_values(ascending=False)
rf_top20 = rf_feat_importance.head(20)

# -------------------------------
# 2. Mutual Information Gain 을 이용한 피처 스코어 산출
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_feat_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False).head(20)

# -------------------------------
# Plot 결과(두 가지 방법)를 subplot 으로 표현 (horizontal bar plot)
plt.figure(figsize=(14, 6))

# RandomForest Importance Plot (horizontal bar)
plt.subplot(1, 2, 1)
rf_top20.plot(kind="barh")
plt.title("Top 20 Feature Importances (RandomForest)")
plt.xlabel("Importance Score")

# Mutual Information Gain Score Plot (horizontal bar)
plt.subplot(1, 2, 2)
mi_feat_scores.plot(kind="barh", color="orange")
plt.title("Top 20 Feature Scores (Mutual Information Gain)")
plt.xlabel("MI Score")

plt.tight_layout()
plt.show()

# selection based on the importance score
# 두 방법에서 선정된 top 20 feature의 교집합만 남기기
selected_features = list(set(rf_top20.index) & set(mi_feat_scores.index))
X_selected = X[selected_features]
print(f"Selected features ({len(selected_features)}): {selected_features}")

# 교집합으로 선택된 feature와 target(status) 컬럼 포함하여 저장
df_selected = X_selected.copy()
df_selected["status"] = y
df_selected.to_csv("selected_features_intersection.csv", index=False)