import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

from models.expense_predictor import ExpensePredictor
from models.anomaly_detector import AnomalyDetector
from models.budget_recommender import BudgetRecommender

# ---------------- Logging ---------------- #
logging.basicConfig(
    filename="logs/insight_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/finance_test.csv"
MODEL_DIR = "models/"
INSIGHTS_DIR = "insights/"

os.makedirs(INSIGHTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


class InsightGenerator:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.exp_pred = ExpensePredictor(os.path.join(MODEL_DIR, "expense_predictor.pkl"))
        self.anomaly_det = AnomalyDetector(os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
        self.budget_rec = BudgetRecommender(os.path.join(MODEL_DIR, "budget_recommender.pkl"))

    # ---------------- Responsible AI ---------------- #
    def responsible_ai_checks(self):
        issues = []

        # Fairness Check → ensure no expense category is unfairly neglected
        if "category" in self.df.columns:
            category_counts = self.df["category"].value_counts(normalize=True)
            if category_counts.min() < 0.05:
                issues.append("⚠️ Some categories underrepresented (<5%) → possible bias in models.")

        # Transparency → log model predictions
        logging.info("Generating transparency samples for predictions...")
        sample = self.df.sample(min(5, len(self.df)))
        preds = self.exp_pred.predict(sample)
        for i, val in enumerate(preds):
            logging.info(f"Transparency: Input={sample.iloc[i].to_dict()}, Prediction={val}")

        # Privacy → check for sensitive columns
        sensitive_cols = [col for col in self.df.columns if "name" in col.lower() or "account" in col.lower()]
        if sensitive_cols:
            issues.append(f"⚠️ Sensitive columns detected: {sensitive_cols}. Ensure anonymization before training.")

        return issues if issues else ["✅ All Responsible AI checks passed."]

    # ---------------- Insights ---------------- #
    def expense_trends(self):
        if "date" not in self.df.columns or "amount" not in self.df.columns:
            return None

        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        monthly = self.df.groupby(self.df["date"].dt.to_period("M"))["amount"].sum()

        plt.figure(figsize=(8, 4))
        monthly.plot(kind="line", marker="o", title="Monthly Expense Trend")
        plt.ylabel("Total Expense")
        plt.tight_layout()
        path = os.path.join(INSIGHTS_DIR, "expense_trend.png")
        plt.savefig(path)
        plt.close()
        return path

    def anomaly_insights(self):
        anomalies = self.anomaly_det.detect(self.df)
        return anomalies.head(5) if not anomalies.empty else None

    def budget_alignment(self):
        recs = self.budget_rec.recommend(self.df)
        return recs.head(5)

    def savings_forecast(self):
        if "date" not in self.df.columns or "amount" not in self.df.columns:
            return None

        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        monthly = self.df.groupby(self.df["date"].dt.to_period("M"))["amount"].sum()

        x = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly.values

        if len(x) < 2:
            return None

        coeffs = np.polyfit(x.flatten(), y, 1)
        forecast_next = coeffs[0] * (len(x) + 1) + coeffs[1]

        return {"last_month": float(y[-1]), "forecast_next_month": float(forecast_next)}

    # ---------------- Run All ---------------- #
    def generate_all(self):
        insights = {}

        # Responsible AI
        insights["responsible_ai"] = self.responsible_ai_checks()

        # Trends
        insights["expense_trend_chart"] = self.expense_trends()

        # Anomalies
        insights["anomalies"] = self.anomaly_insights()

        # Budget
        insights["budget_recommendations"] = self.budget_alignment()

        # Forecast
        insights["savings_forecast"] = self.savings_forecast()

        logging.info("✅ Insights generated successfully.")
        return insights


if __name__ == "__main__":
    gen = InsightGenerator()
    result = gen.generate_all()

    print("\n=== Responsible AI Checks ===")
    for issue in result["responsible_ai"]:
        print("-", issue)

    print("\n=== Expense Trend Chart ===")
    print("Saved at:", result["expense_trend_chart"])

    print("\n=== Detected Anomalies ===")
    print(result["anomalies"])

    print("\n=== Budget Recommendations ===")
    print(result["budget_recommendations"])

    print("\n=== Savings Forecast ===")
    print(result["savings_forecast"])
