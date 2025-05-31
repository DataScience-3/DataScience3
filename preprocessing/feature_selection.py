import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

def select_features(input_path, output_path):
    # Read file (use data after feature creation)
    df = pd.read_csv(input_path)

    # Split target (y) and features (X) (status column is target)
    X = df.drop("status", axis=1)
    y = df["status"]

    # -------------------------------
    # 1. Feature Importance using RandomForest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    # Create Series and sort descending
    rf_feat_importance = pd.Series(rf_importances, index=X.columns).sort_values(ascending=False)
    rf_top20 = rf_feat_importance.head(20)

    # -------------------------------
    # 2. Calculate feature scores using Mutual Information Gain
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_feat_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False).head(20)

    # -------------------------------
    # Plot results (both methods) as subplot (horizontal bar plot)
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
    # Keep only the intersection of top 20 features selected by both methods
    # There are two ways to select. We have used the second one.

    # 1. For automated feature selection:
    # selected_features = list(set(rf_top20.index) & set(mi_feat_scores.index))
    # X_selected = X[selected_features]
    # print(f"Selected features ({len(selected_features)}): {selected_features}")

    # 2. Feature selection by hand because of its reproducibility
    # Selected features are:
    selected_features = [
        'google_index', 'links_in_tags', 'length_url', 'ratio_extRedirection',
        'longest_words_raw', 'domain_age_created', 'nb_hyperlinks', 'ratio_digits_url',
        'longest_word_path', 'web_traffic', 'safe_anchor', 'avg_word_path',
        'nb_www', 'ratio_extHyperlinks', 'ratio_intHyperlinks', 'page_rank'
    ]

    # Save the selected features (intersection) and target (status) column
    X_selected = X[selected_features]
    df_selected = X_selected.copy()
    df_selected["status"] = y
    df_selected.to_csv(output_path, index=False)

if __name__ == '__main__':
    select_features('dataset_phishing_feature_creation.csv', 'selected_features_intersection.csv')