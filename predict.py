"""
predict.py - Standalone Inference and Customer Churn Scoring Script.

This script:
1. Loads the persisted scikit-learn pipeline (`results/churn_model.joblib`).
2. Takes customer profiles (pre-configured archetypes or user-provided input).
3. Evaluates churn probability using the trained Logistic Regression model.
4. Outputs the churn prediction, confidence percentage, risk classification tier,
   and actionable customer retention recommendations.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Customer Churn Probability")
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/churn_model.joblib",
        help="Path to trained pipeline file (default: results/churn_model.joblib)",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        choices=["Month-to-month", "One year", "Two year"],
        help="Customer contract type",
    )
    parser.add_argument("--tenure", type=int, default=None, help="Customer tenure in months (0-72)")
    parser.add_argument(
        "--monthly-charges", type=float, default=None, help="Monthly charges in dollars ($18-$130)"
    )
    parser.add_argument(
        "--internet",
        type=str,
        default=None,
        choices=["DSL", "Fiber optic", "No"],
        help="Internet service type",
    )
    parser.add_argument(
        "--payment",
        type=str,
        default=None,
        choices=["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        help="Payment method",
    )
    return parser.parse_args()


def load_pipeline(model_path: Path):
    """Load the serialized scikit-learn pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python churn_prediction.py' first to train and save the pipeline."
        )
    return joblib.load(model_path)


def score_customer(pipeline, customer_df: pd.DataFrame, label: str = "Customer"):
    """Score customer profile and print diagnostic breakdown."""
    prob_churn = pipeline.predict_proba(customer_df)[0, 1]
    prediction = int(prob_churn >= 0.5)

    if prob_churn >= 0.65:
        risk_level = "HIGH RISK"
        color_tag = "[CRITICAL ACTION REQUIRED]"
        action = "Offer contract migration discount (1-2 year agreement), complimentary premium tech support, and billing review."
    elif prob_churn >= 0.40:
        risk_level = "MODERATE RISK"
        color_tag = "[MONITOR / TARGETED OFFER]"
        action = "Send proactive satisfaction survey and loyalty reward incentive before next renewal cycle."
    else:
        risk_level = "LOW RISK"
        color_tag = "[HEALTHY / LOYAL]"
        action = "Maintain regular service quality; opportunity for upsell or referral program enrollment."

    print("\n" + "=" * 60)
    print(f"PROFILE: {label}")
    print("=" * 60)
    for col in customer_df.columns:
        print(f"  {col:<20}: {customer_df[col].iloc[0]}")
    print("-" * 60)
    print(f"Predicted Outcome:       {'CHURN (1)' if prediction == 1 else 'RETAIN (0)'}")
    print(f"Churn Probability:       {prob_churn * 100:.2f}%")
    print(f"Risk Tier:               {risk_level} {color_tag}")
    print(f"Retention Strategy:      {action}")
    print("=" * 60)


def run_sample_archetypes(pipeline):
    """Run inference across 3 representative customer archetypes."""
    print("\n--- Running Churn Scoring on Representative Customer Archetypes ---")

    # Archetype 1: High Risk (New customer, month-to-month, fiber optic, electronic check)
    high_risk_customer = pd.DataFrame(
        [
            {
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "InternetService": "Fiber optic",
                "TechSupport": "No",
                "OnlineSecurity": "No",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "Contract": "Month-to-month",
                "MonthlyCharges": 89.50,
            }
        ]
    )
    score_customer(pipeline, high_risk_customer, "Archetype 1 - New Month-to-Month Fiber Customer")

    # Archetype 2: Low Risk (Long tenure, two-year contract, bundled security, automatic payment)
    low_risk_customer = pd.DataFrame(
        [
            {
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 54,
                "InternetService": "DSL",
                "TechSupport": "Yes",
                "OnlineSecurity": "Yes",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer",
                "Contract": "Two year",
                "MonthlyCharges": 55.20,
            }
        ]
    )
    score_customer(pipeline, low_risk_customer, "Archetype 2 - Long-Tenure Two-Year Contract Customer")

    # Archetype 3: Moderate Risk (Moderate tenure, one-year contract)
    moderate_risk_customer = pd.DataFrame(
        [
            {
                "SeniorCitizen": 1,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 18,
                "InternetService": "Fiber optic",
                "TechSupport": "No",
                "OnlineSecurity": "Yes",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Credit card",
                "Contract": "One year",
                "MonthlyCharges": 82.00,
            }
        ]
    )
    score_customer(pipeline, moderate_risk_customer, "Archetype 3 - Mid-Tenure One-Year Contract Senior Customer")


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    pipeline = load_pipeline(model_path)

    # If user provided custom parameters
    if args.contract is not None or args.tenure is not None:
        custom_customer = pd.DataFrame(
            [
                {
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": args.tenure if args.tenure is not None else 12,
                    "InternetService": args.internet if args.internet is not None else "Fiber optic",
                    "TechSupport": "No",
                    "OnlineSecurity": "No",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": args.payment if args.payment is not None else "Electronic check",
                    "Contract": args.contract if args.contract is not None else "Month-to-month",
                    "MonthlyCharges": args.monthly_charges if args.monthly_charges is not None else 75.0,
                }
            ]
        )
        score_customer(pipeline, custom_customer, "User-Specified Custom Customer Profile")
    else:
        run_sample_archetypes(pipeline)


if __name__ == "__main__":
    main()
