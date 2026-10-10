"""
churn_prediction.py - End-to-End Customer Churn Prediction Pipeline.

This script:
1. Loads or generates a reproducible telecom customer churn dataset (7,043 customer records).
2. Performs data preprocessing with scikit-learn ColumnTransformer:
   - One-Hot Encoding for categorical demographic and service features.
   - Standard Scaling for continuous numerical features (tenure, monthly charges).
3. Splits data into stratified 80% train / 20% test sets.
4. Trains a balanced Logistic Regression classification pipeline.
5. Evaluates model performance using:
   - Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Confusion Matrix and full Classification Report.
6. Conducts feature importance analysis (model coefficients and odds ratios).
7. Generates visual plots (confusion matrix heatmap, ROC curve, feature importance chart).
8. Persists the trained model (churn_model.joblib) and all evaluation metrics.
"""

import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--data-path", type=str, default="data.csv", help="Path to customer dataset CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def get_or_create_dataset(data_path: Path, seed: int = 42) -> pd.DataFrame:
    """
    Load dataset from CSV if available, or generate a reproducible synthetic telecom dataset.
    Modeled after industry telecom churn dynamics (7,043 customer records).
    """
    if data_path.exists():
        print(f"Loading existing customer dataset from: {data_path}")
        df = pd.read_csv(data_path)
        return df

    print(f"Generating reproducible synthetic telecom churn dataset (N=7,043, seed={seed})...")
    rng = np.random.default_rng(seed)
    n = 7043

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.clip(rng.gamma(2.2, 14, n).round().astype(int), 0, 72)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    monthly = np.round(
        np.clip(rng.normal(70, 25, n) + (internet == "Fiber optic") * 12, 18, 130), 2
    )
    tech = rng.choice(["Yes", "No", "No internet service"], n, p=[0.22, 0.60, 0.18])
    online_sec = rng.choice(["Yes", "No", "No internet service"], n, p=[0.28, 0.54, 0.18])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n,
        p=[0.34, 0.22, 0.22, 0.22],
    )
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # Logit model reflecting plausible business churn drivers
    logit = (
        -1.15
        + 0.95 * (contract == "Month-to-month")
        - 0.75 * (contract == "Two year")
        + 0.018 * (monthly - 70)
        - 0.032 * tenure
        + 0.55 * (internet == "Fiber optic")
        + 0.45 * (payment == "Electronic check")
        + 0.32 * (paperless == "Yes")
        + 0.28 * senior
        - 0.30 * (partner == "Yes")
        - 0.38 * (tech == "Yes")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "InternetService": internet,
            "TechSupport": tech,
            "OnlineSecurity": online_sec,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "Contract": contract,
            "MonthlyCharges": monthly,
            "Churn": churn,
        }
    )
    df.to_csv(data_path, index=False)
    print(f"Dataset successfully created and saved to: {data_path}")
    return df


def build_pipeline(cat_cols: list, num_cols: list, seed: int = 42) -> Pipeline:
    """Build scikit-learn preprocessing and Logistic Regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    return pipeline


def extract_feature_importance(pipeline: Pipeline, cat_cols: list, num_cols: list) -> pd.DataFrame:
    """Extract model coefficients and odds ratios from fitted Logistic Regression pipeline."""
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = encoded_cat_names + num_cols

    coefficients = pipeline.named_steps["classifier"].coef_[0]
    odds_ratios = np.exp(coefficients)

    fi_df = pd.DataFrame(
        {
            "Feature": all_feature_names,
            "Coefficient": coefficients,
            "Odds_Ratio": odds_ratios,
            "Impact": [
                "Increases Churn" if c > 0 else "Decreases Churn" for c in coefficients
            ],
        }
    ).sort_values(by="Coefficient", ascending=False)

    return fi_df


def save_visualizations(
    y_true, y_pred, y_prob, fi_df: pd.DataFrame, output_dir: Path
):
    """Generate and save publication-quality evaluation plots."""
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        linewidths=0.5,
    )
    plt.title("Confusion Matrix — Customer Churn Prediction", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Logistic Regression (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Top Feature Importance (Coefficients)
    top_positive = fi_df.head(6)
    top_negative = fi_df.tail(6)
    top_features = pd.concat([top_positive, top_negative]).sort_values(by="Coefficient")

    colors = ["#16a34a" if c < 0 else "#dc2626" for c in top_features["Coefficient"]]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_features["Feature"], top_features["Coefficient"], color=colors)
    plt.axvline(0, color="black", linestyle="--", lw=0.8)
    plt.title("Key Customer Churn Drivers (Logistic Regression Coefficients)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Coefficient Value (Positive: Churn Risk | Negative: Retention Factor)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load / Prepare Dataset
    data_path = Path(args.data_path)
    df = get_or_create_dataset(data_path, seed=args.seed)

    X = df.drop(columns="Churn")
    y = df["Churn"]

    # Explicitly extract column names as Python lists to avoid dtype indexing discrepancies
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    print(f"\nFeature Summary:")
    print(f"  - Categorical Features ({len(cat_cols)}): {cat_cols}")
    print(f"  - Numerical Features   ({len(num_cols)}): {num_cols}")
    print(f"  - Total Customers:     {len(df):,}")
    print(f"  - Churn Rate:          {y.mean()*100:.2f}% ({y.sum():,} churned)")

    # 2. Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    print(f"  - Train Set:           {len(X_train):,} samples")
    print(f"  - Test Set:            {len(X_test):,} samples")

    # 3. Build and Fit Pipeline
    pipeline = build_pipeline(cat_cols, num_cols, seed=args.seed)
    print("\nTraining Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    # 4. Predict on Test Set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 5. Model Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

    # 6. Feature Importance Extraction
    fi_df = extract_feature_importance(pipeline, cat_cols, num_cols)

    # 7. Save Artifacts
    # Save Model
    model_path = output_dir / "churn_model.joblib"
    joblib.dump(pipeline, model_path)

    # Save Metrics CSV and JSON
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "classification_report.txt").write_text(cls_report)
    pd.DataFrame(
        cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
    ).to_csv(output_dir / "confusion_matrix.csv")
    fi_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save Visualization Plots
    print("Generating and saving evaluation plots...")
    save_visualizations(y_test, y_pred, y_prob, fi_df, output_dir)

    # 8. Print Results Summary
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:   {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:      {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print("====================================================")

    print("\n--- Top Factors Increasing Churn Risk ---")
    for _, row in fi_df.head(4).iterrows():
        print(f"  + {row['Feature']:<30} (Coef: {row['Coefficient']:+.4f}, Odds Ratio: {row['Odds_Ratio']:.2f}x)")

    print("\n--- Top Factors Decreasing Churn Risk (Protective) ---")
    for _, row in fi_df.tail(4).iloc[::-1].iterrows():
        print(f"  - {row['Feature']:<30} (Coef: {row['Coefficient']:+.4f}, Odds Ratio: {row['Odds_Ratio']:.2f}x)")

    print(f"\nArtifacts successfully saved in: {output_dir.resolve()}")
    print("  - Trained Model:        churn_model.joblib")
    print("  - Metrics CSV:          metrics.csv / metrics.json")
    print("  - Classification Rep:   classification_report.txt")
    print("  - Confusion Matrix:     confusion_matrix.csv & confusion_matrix.png")
    print("  - ROC Curve:            roc_curve.png")
    print("  - Feature Importance:   feature_importance.csv & feature_importance.png")


if __name__ == "__main__":
    main()
