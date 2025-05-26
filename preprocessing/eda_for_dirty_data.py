import pandas as pd
import numpy as np

# 데이터 읽기
df = pd.read_csv('dataset_phishing.csv')

# 각 컬럼별 음수 값 개수 확인
negative_counts = {}
for column in df.columns:
    # 숫자로 변환, 변환 불가 값은 NaN 처리
    col_numeric = pd.to_numeric(df[column], errors='coerce')
    negative_count = (col_numeric < 0).sum()
    if negative_count > 0:
        negative_counts[column] = negative_count

# 결과 출력
print("\n음수 값을 포함한 컬럼과 음수 값의 수:")
for column, count in negative_counts.items():
    print(f"{column}: {count}개")


