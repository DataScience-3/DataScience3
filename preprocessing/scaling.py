import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, MinMaxScaler

# CSV 파일 읽기
df = pd.read_csv('selected_features_intersection.csv')

# Log scaling 적용할 컬럼 (0인 경우도 log1p로 처리)
log_cols = ['nb_hyperlinks', 'web_traffic', 'longest_word_path', 'length_url',
            'ratio_extRedirection', 'longest_words_raw', 'avg_word_path']
for col in log_cols:
    df[col] = np.log1p(df[col])

# Min-Max Scaler 적용: safe_anchor
minmax_scaler = MinMaxScaler()
df[['safe_anchor']] = minmax_scaler.fit_transform(df[['safe_anchor']])

# RobustScaler 적용: nb_www와 domain_age_created
robust_scaler = RobustScaler()
df[['nb_www', 'domain_age_created']] = robust_scaler.fit_transform(df[['nb_www', 'domain_age_created']])

# Z-score scaling (StandardScaler) 적용: page_rank
zscore_scaler = StandardScaler()
df[['page_rank']] = zscore_scaler.fit_transform(df[['page_rank']])

# QuantileTransformer 적용: links_in_tags, ratio_digits_url, ratio_extHyperlinks, ratio_intHyperlinks
quantile_transformer = QuantileTransformer()
df[['links_in_tags', 'ratio_digits_url', 'ratio_extHyperlinks', 'ratio_intHyperlinks']] = quantile_transformer.fit_transform(
    df[['links_in_tags', 'ratio_digits_url', 'ratio_extHyperlinks', 'ratio_intHyperlinks']]
)

print(df.head())


# Secondary scaling
# Z-score 표준화를 적용할 숫자형 열 (이미 log scaling 등을 적용한 열이 아닐 경우)
numeric_scaled_cols = ['web_traffic', 'nb_hyperlinks', 'length_url', 'longest_words_raw', 'longest_word_path']

# StandardScaler를 적용하여 해당 열만 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_scaled_cols])
df_scaled = pd.DataFrame(scaled_data, columns=numeric_scaled_cols, index=df.index)

# 나머지 열은 그대로 둠
other_cols = [col for col in df.columns if col not in numeric_scaled_cols]
df_other = df[other_cols]

# 스케일링된 열과 나머지 열을 merge
final_df = pd.concat([df_scaled, df_other], axis=1)

# 최종 행렬을 CSV 파일로 저장
final_df.to_csv("final_matrix.csv", index=False)
