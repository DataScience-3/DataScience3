import pandas as pd
import numpy as np

# 생성된 dirty data 파일 읽기
df = pd.read_csv('dataset_phishing_dirty.csv')

print("원본 데이터프레임의 shape:", df.shape)

# 더티 데이터가 삽입된 5개 컬럼에 대해 바로 drop 처리할 조건 함수 정의
cols_to_clean = ['https_token', 'length_url', 'phish_hints', 'nb_at', 'page_rank']

def is_valid_integer(x):
    if pd.isna(x):
        return False
    # '+'와 '%'는 drop 처리
    if x in ['+', '%']:
        return False
    try:
        # 값을 float로 변환
        num = float(x)
    except Exception:
        return False
    # 정수 값이면 True, 아니면 False
    return num.is_integer()

# 각 컬럼에 대해 유효하지 않은 행 제거
for col in cols_to_clean:
    df = df[df[col].apply(is_valid_integer)]

# domain_age와 domain_registration_length가 음수인 행 제거
df = df[(df['domain_registration_length'] >= 0)]

# Data value change
# status 컬럼을 mapping: legitimate -> 0, phishing -> 1
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# url 컬럼 제거 (Unusable data)
df = df.drop('url', axis=1)

# 최종 정제된 데이터 저장
df.to_csv('dataset_phishing_clean.csv', index=False)
print("정제된 데이터프레임의 shape:", df.shape)