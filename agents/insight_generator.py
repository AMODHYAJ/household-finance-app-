import pandas as pd
import os
from models.expense_predictor import ExpensePredictor
from models.anomaly_detector import AnomalyDetector
from models.category_classifier import CategoryClassifier
from models.budget_recommender import BudgetRecommender

class InsightGeneratorAgent:
    def __init__(self, data_agent=None, model_dir="models/"):
        self.data_agent = data_agent
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Load models
        self.expense_predictor = ExpensePredictor(model_path=os.path.join(model_dir, "expense_predictor.pkl"))
        self.anomaly_detector = AnomalyDetector(model_path=os.path.join(model_dir, "anomaly_detector.pkl"))
        self.category_classifier = CategoryClassifier(model_path=os.path.join(model_dir, "category_classifier.pkl"))
        self.budget_recommender = BudgetRecommender()

    def summarize(self, df):
        if df is None or df.empty:
            return "No transactions yet."
        total_income = df[df["type"] == "income"]["amount"].sum()
        total_expense = df[df["type"] == "expense"]["amount"].sum()
        return {
            "Total Income": round(total_income, 2),
            "Total Expense": round(total_expense, 2),
            "Savings": round(total_income - total_expense, 2)
        }

    def generate_insights(self, df):
        if df is None or df.empty:
            return ["No data available."]

        insights = []

        # Income vs Expense check
        income = df[df["type"] == "income"]["amount"].sum()
        expense = df[df["type"] == "expense"]["amount"].sum()
        insights.append("⚠ You spent more than you earned." if expense > income else "✅ You saved money overall.")

        # Expense prediction
        try:
            last_date = pd.to_datetime(df["date"]).max()
            next_month = (last_date.month % 12) + 1
            next_year = last_date.year + (1 if last_date.month == 12 else 0)
            pred = self.expense_predictor.predict_next_month(next_month, next_year)
            insights.append(f"📅 Predicted expenses for {next_month}/{next_year}: ${pred:.2f}")
        except Exception:
            insights.append("⚠ Prediction unavailable.")

        # Anomaly detection
        try:
            anomalies = self.anomaly_detector.detect(df)
            if not anomalies.empty:
                a = anomalies.iloc[-1]
                insights.append(f"🚨 Anomaly: {a['category']} - ${a['amount']:.2f} on {a['date']}")
        except Exception:
            insights.append("⚠ Anomaly detection unavailable.")

        # Category classifier
        try:
            if "note" in df.columns and not df["note"].dropna().empty:
                note = df["note"].dropna().iloc[-1]
                pred_cat = self.category_classifier.predict(note)
                insights.append(f"📝 Last note '{note}' classified as '{pred_cat}'")
        except Exception:
            insights.append("⚠ Classification unavailable.")

        # Budget recommendations
        try:
            insights.extend(self.budget_recommender.recommend(df))
        except Exception:
            insights.append("⚠ Recommendations unavailable.")

        return insights
