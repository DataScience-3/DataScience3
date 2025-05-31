import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score
)


def run_ISO(df: pd.DataFrame) -> dict:
    """
    df만 입력으로 받아서 IsolationForest 교차검증을 수행하고,
    평가 지표를 출력한 뒤, 결과를 딕셔너리로 반환합니다.
    
    내부에서는 label_col='status', n_splits=5, contamination=0.1, random_state=42로 고정합니다.
    """
    label_col = 'status'
    n_splits = 5
    contamination = 0.1
    random_state = 42

    #Target 분리
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}'Target을 찾을 수 없습니다.")

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    #스케일링
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    #PCA 2차원 결과
    X_pca2 = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

    #StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.zeros_like(y)
    scores = np.zeros_like(y, dtype=float)
    thresholds = []

    #교차 검증
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_scaled, y), start=1):
        X_tr, y_tr = X_scaled[tr_idx], y[tr_idx]
        X_te = X_scaled[te_idx]

        #정상 데이터만 학습
        X_tr_norm = X_tr[y_tr == 0]

        iso = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state
        )
        iso.fit(X_tr_norm)

        # threshold = -offset_ -> -1을 붙여서 큰 값이 이상치인 것으로 바꿈(실제는 -1이 가장 이상치)
        thr = -iso.offset_
        thresholds.append(thr)
        print(f"Fold {fold_idx} Threshold = {thr:.4f}")

        raw_pred = iso.predict(X_te)                # {1: 정상, -1: 이상치}
        pred_label = (raw_pred == -1).astype(int)    # 이상치→1, 정상→0
        score_te = -iso.score_samples(X_te)         # score_samples: 낮을수록 이상치, 부정하여 클수록 이상치 의심 ↑

        y_pred[te_idx] = pred_label
        scores[te_idx] = score_te

    #평가 지표 계산
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    f1_p = f1_score(y, y_pred, pos_label=1)
    f1_l = f1_score(y, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y, scores)
    report = classification_report(y, y_pred, target_names=['Legit', 'Phish'])

    #출력
    print("\n=== IsolationForest Evaluation ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"ROC AUC       : {roc_auc:.4f}")
    print(f"F1 (Phish)    : {f1_p:.4f}")
    print(f"F1 (Legit)    : {f1_l:.4f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)

    #결과 반환
    results = {
        'confusion_matrix': cm,
        'accuracy': acc,
        'f1_phish': f1_p,
        'f1_legit': f1_l,
        'roc_auc': roc_auc,
        'classification_report': report,
        'y_true': y.copy(),
        'y_pred': y_pred,
        'scores': scores,
        'thresholds': thresholds,
        'X_pca2': X_pca2
    }

    return results


def run_ISO_plot(df: pd.DataFrame):
    """
    df만 입력받아 내부에서 run_iforest_cv_from_df를 호출하고,
    평가 지표 출력 및 시각화까지 모두 수행합니다.
    """
    results = run_iforest_cv_from_df(df)

    cm         = results['confusion_matrix']
    y_true     = results['y_true']
    y_pred     = results['y_pred']
    scores     = results['scores']
    thresholds = results['thresholds']
    roc_auc    = results['roc_auc']
    X_pca2     = results['X_pca2']

    #Confusion matrix Hitmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Legit', 'Phish'],
        yticklabels=['Legit', 'Phish']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    #Cluster
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X_pca2[:, 0],
        X_pca2[:, 1],
        c=y_pred,
        cmap=plt.cm.Set1,
        s=20,
        alpha=0.7
    )
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ['Legit', 'Phish'], title='Predicted')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Prediction')
    plt.tight_layout()
    plt.show()

    #이상치 점수 분포 + 마지막 Fold 기준 threshold
    final_thr = thresholds[-1]
    plt.figure(figsize=(6, 4))
    sns.histplot(scores, bins=30, kde=True)
    plt.axvline(final_thr, color='red', linestyle='--', label=f'Threshold={final_thr:.2f}')
    plt.title(' Anomaly Score Distribution')
    plt.xlabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #ROC 커브
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('final_matrix.csv')
    run_ISO_plot(df)



