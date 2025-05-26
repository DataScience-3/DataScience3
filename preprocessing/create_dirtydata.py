import pandas as pd
import random
import numpy as np

dirty_counts = { 'https_token': 55, 'length_url': 34, 'phish_hints': 46, 'nb_at': 20, 'page_rank': 43 }

dirty_values = [np.nan, '+', '%', 3.22, -1.48, -9.53, 28.7]

df = pd.read_csv('dataset_phishing.csv')
idx_list = df.index.tolist()

for col, count in dirty_counts.items(): # 데이터프레임의 인덱스 리스트 idx_list = df.index.tolist()
    # 데이터 행의 개수가 count보다 작으면 중복 선택(복원 추출)하도록, 그렇지 않으면 중복 없이 선택합니다.
    if len(idx_list) < count:
        selected_indices = random.choices(idx_list, k=count)
    else:
        selected_indices = random.sample(idx_list, count)

    # 선택한 각 행의 해당 컬럼 값을 임의의 dirty value로 대체
    for i in selected_indices:
        df.at[i, col] = random.choice(dirty_values)

df.to_csv('dataset_phishing_dirty.csv', index=False)
