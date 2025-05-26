import pandas as pd

# 파일 읽기
df = pd.read_csv("dataset_phishing_clean.csv")

# 음수인 경우: domain_age_missing 컬럼 생성 (True로 지정)
df["domain_age_missing"] = (df["domain_age"] < 0).astype(int)

# 그 외 (0보다 큰 값)인 경우: domain_age_created 컬럼에 원래의 값을 넣어줌
df["domain_age_created"] = df["domain_age"].apply(lambda x: x if x >= 0 else 0)

# 기존 domain_age 컬럼 삭제
df = df.drop("domain_age", axis=1)

# 결과 확인
print(df.head())

df.to_csv("dataset_phishing_feature_creation.csv", index=False)