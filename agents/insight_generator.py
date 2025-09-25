# agents/insight_generator.py
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from models.anomaly_detector import AnomalyDetector
from models.budget_recommender import BudgetRecommender

# ---------------- Setup ---------------- #
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/insight_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/finance_test.csv"
MODEL_DIR = "models/"


class InsightGeneratorAgent:
    """
    Insight generator for UNIQUE value:
      - Summary stats
      - Anomalies (with explanations)
      - Trend detection
      - Forecasting
      - Goal tracking
      - Personalized alerts
      - Responsible AI checks
    """

    def __init__(self, df=None, savings_target: float = 5000):
        if df is None:
            try:
                self.df = pd.read_csv(DATA_PATH)
            except Exception:
                self.df = pd.DataFrame()
        else:
            if hasattr(df, "get_transactions_df") and callable(df.get_transactions_df):
                try:
                    self.df = df.get_transactions_df()
                except Exception:
                    self.df = pd.DataFrame()
            elif isinstance(df, pd.DataFrame):
                self.df = df.copy()
            else:
                try:
                    self.df = pd.DataFrame(df)
                except Exception:
                    self.df = pd.DataFrame()

        self.anomaly_det = AnomalyDetector(os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
        self.budget_rec = BudgetRecommender(os.path.join(MODEL_DIR, "budget_recommender.pkl"))

        self.savings_target = savings_target

    # ---------------- Responsible AI ---------------- #
    def responsible_ai_checks(self):
        issues = []

        if self.df is None or self.df.empty:
            return ["⚠️ No data available for Responsible AI checks."]

        # Category balance
        if "category" in self.df.columns:
            category_counts = self.df["category"].value_counts(normalize=True)
            if not category_counts.empty and category_counts.min() < 0.05:
                issues.append("⚠️ Some categories are underrepresented (<5%) → possible bias.")
            else:
                issues.append("✅ Category distribution looks balanced.")

        # Transparency sample
        try:
            sample = self.df.sample(min(3, len(self.df)))
            logging.info("Transparency sample logged:")
            for _, row in sample.iterrows():
                logging.info(row.to_dict())
            issues.append("✅ Transparency: sample rows logged.")
        except Exception as e:
            logging.warning(f"Transparency check failed: {e}")
            issues.append("⚠️ Transparency check failed.")

        # Sensitive data
        sensitive_cols = [c for c in self.df.columns if "name" in c.lower() or "account" in c.lower()]
        if sensitive_cols:
            issues.append(f"⚠️ Sensitive columns detected: {sensitive_cols}. Consider anonymization.")
        else:
            issues.append("✅ No sensitive columns detected.")

        return issues

    # ---------------- Basic Insights ---------------- #
    def summary_stats(self):
        if self.df is None or self.df.empty:
            return {}

        total_income = self.df[self.df["type"] == "income"]["amount"].sum()
        total_expense = self.df[self.df["type"] == "expense"]["amount"].sum()
        savings = total_income - total_expense
        savings_rate = (savings / total_income * 100) if total_income > 0 else 0

        return {
            "income": total_income,
            "expense": total_expense,
            "savings": savings,
            "savings_rate": savings_rate
        }

    def anomaly_insights(self):
        if self.df is None or self.df.empty:
            return pd.DataFrame(), []

        anomalies = pd.DataFrame()
        messages = []
        try:
            anomalies = self.anomaly_det.detect(self.df)
            if not anomalies.empty:
                for _, row in anomalies.iterrows():
                    messages.append(
                        f"⚠️ On {row['date']}, unusual spending detected: ${row['amount']:.2f} in {row['category']}."
                    )
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")

        return anomalies, messages

    # ---------------- Advanced Insights ---------------- #
    def trend_insights(self):
        if self.df is None or self.df.empty:
            return "No data available for trend analysis."

        if "date" not in self.df.columns or "amount" not in self.df.columns:
            return "Insufficient data for trend analysis."

        df_copy = self.df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        monthly = df_copy.groupby(df_copy["date"].dt.to_period("M"))["amount"].sum()

        if len(monthly) < 3:
            return "Not enough data for trend detection."

        # Simple trend detection: compare last 3 months
        last_values = monthly[-3:].values
        if all(np.diff(last_values) > 0):
            return "📈 Spending has increased for the past 3 months."
        elif all(np.diff(last_values) < 0):
            return "📉 Spending has decreased for the past 3 months."
        else:
            return "➡️ Spending shows mixed patterns over the past months."

    def savings_forecast(self):
        if self.df is None or self.df.empty:
            return None

        if "date" not in self.df.columns or "amount" not in self.df.columns:
            return None

        df_copy = self.df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        monthly = df_copy.groupby(df_copy["date"].dt.to_period("M"))["amount"].sum()

        if len(monthly) < 2:
            return None

        x = np.arange(len(monthly))
        y = monthly.values

        coeffs = np.polyfit(x, y, 1)
        forecast_next = float(coeffs[0] * (len(x)) + coeffs[1])

        return {
            "last_month": float(y[-1]),
            "forecast_next_month": forecast_next
        }

    def goal_tracking(self):
        stats = self.summary_stats()
        if not stats or stats.get("savings") is None:
            return "No savings data available."

        progress = (stats["savings"] / self.savings_target) * 100
        return f"🎯 You have reached {progress:.1f}% of your savings target (${self.savings_target:,.0f})."

    def alerts(self):
        if self.df is None or self.df.empty:
            return []

        if "date" not in self.df.columns or "amount" not in self.df.columns:
            return []

        alerts = []
        df_copy = self.df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        weekly = df_copy.groupby(df_copy["date"].dt.to_period("W"))["amount"].sum()

        if len(weekly) < 4:
            return []

        last_week = weekly.iloc[-1]
        avg_prev = weekly.iloc[:-1].mean()

        if last_week > avg_prev * 1.3:
            alerts.append("⚠️ This week’s expenses are over 30% higher than your usual average.")

        return alerts

    # ---------------- Run All ---------------- #
    def generate_all(self):
        insights = {}
        insights["responsible_ai"] = self.responsible_ai_checks()
        insights["summary_stats"] = self.summary_stats()
        anomalies, anomaly_msgs = self.anomaly_insights()
        insights["anomalies"] = anomalies
        insights["anomaly_messages"] = anomaly_msgs
        insights["trend"] = self.trend_insights()
        insights["forecast"] = self.savings_forecast()
        insights["goal"] = self.goal_tracking()
        insights["alerts"] = self.alerts()
        return insights
