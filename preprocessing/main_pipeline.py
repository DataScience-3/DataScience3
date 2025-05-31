# 전체 데이터 전처리 및 분석 파이프라인을 순차적으로 실행하는 스크립트입니다.
# 각 단계별 함수는 각 파일에서 import하여 사용합니다.

from create_dirtydata import create_dirty_data
from data_cleaning import clean_data
from feature_creation import create_features
from feature_selection import select_features
from scaling import scale_features
from pca import run_pca
import numpy as np
import sys

def main():
    # 1. Dirty data 생성
    dirty_counts = { 'https_token': 55, 'length_url': 34, 'phish_hints': 46, 'nb_at': 20, 'page_rank': 43 }
    dirty_values = [np.nan, '+', '%', 3.22, -1.48, -9.53, 28.7]
    create_dirty_data('dataset_phishing.csv', 'dataset_phishing_dirty.csv', dirty_counts, dirty_values)

    # 2. Dirty data 정제
    clean_data('dataset_phishing_dirty.csv', 'dataset_phishing_clean.csv')

    # 3. 파생 피처 생성
    create_features('dataset_phishing_clean.csv', 'dataset_phishing_feature_creation.csv')

    # 4. 피처 선택
    select_features('dataset_phishing_feature_creation.csv', 'selected_features_intersection.csv')

    # 5. 스케일링
    scale_features('selected_features_intersection.csv', 'final_matrix.csv')

    # 6. PCA
    run_pca('final_matrix.csv', 'pca_result.csv', n_components=10)

if __name__ == '__main__':
    main() 