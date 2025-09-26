import pandas as pd
import numpy as np
import logging
import os

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

    def __init__(self, data_agent=None, model_dir="models/"):
        self.data_agent = data_agent
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Load models
        self.anomaly_detector = AnomalyDetector(model_path=os.path.join(model_dir, "anomaly_detector.pkl"))
        self.budget_recommender = BudgetRecommender(model_path=os.path.join(model_dir, "budget_recommender.pkl"))
        self.savings_target = 5000

    def get_dataframe(self):
        """Get dataframe from data agent - FIXED version"""
        try:
            if self.data_agent is not None and hasattr(self.data_agent, 'get_transactions_df'):
                df = self.data_agent.get_transactions_df()
                # Proper way to check if DataFrame is empty
                if df is not None and not df.empty:
                    return df
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error getting dataframe: {e}")
            return pd.DataFrame()

    # ---------------- Responsible AI ---------------- #
    def responsible_ai_checks(self):
        """Responsible AI checks - FIXED version"""
        issues = []
        
        try:
            df = self.get_dataframe()
            # Proper DataFrame emptiness check
            if df is None or df.empty:
                return ["⚠️ No data available for Responsible AI checks."]

            # Category balance
            if "category" in df.columns:
                category_counts = df["category"].value_counts(normalize=True)
                if not category_counts.empty and len(category_counts) > 0 and category_counts.min() < 0.05:
                    issues.append("⚠️ Some categories are underrepresented (<5%) → possible bias.")
                else:
                    issues.append("✅ Category distribution looks balanced.")

            # Transparency sample
            try:
                sample_size = min(3, len(df))
                if sample_size > 0:
                    sample = df.sample(sample_size)
                    logging.info("Transparency sample logged:")
                    for _, row in sample.iterrows():
                        logging.info(row.to_dict())
                    issues.append("✅ Transparency: sample rows logged.")
            except Exception as e:
                logging.warning(f"Transparency check failed: {e}")
                issues.append("⚠️ Transparency check failed.")

            # Sensitive data check
            sensitive_cols = [c for c in df.columns if "name" in c.lower() or "account" in c.lower()]
            if sensitive_cols:
                issues.append(f"⚠️ Sensitive columns detected: {sensitive_cols}. Consider anonymization.")
            else:
                issues.append("✅ No sensitive columns detected.")

        except Exception as e:
            logging.error(f"Responsible AI checks failed: {e}")
            issues.append("⚠️ Responsible AI checks failed due to error.")

        return issues

    # ---------------- Basic Insights ---------------- #
    def summary_stats(self):
        df = self.get_dataframe()
        if df is None or df.empty:
            return {}

        total_income = df[df["type"] == "income"]["amount"].sum()
        total_expense = df[df["type"] == "expense"]["amount"].sum()
        savings = total_income - total_expense
        savings_rate = (savings / total_income * 100) if total_income > 0 else 0

        return {
            "income": total_income,
            "expense": total_expense,
            "savings": savings,
            "savings_rate": savings_rate
        }

    def anomaly_insights(self):
        df = self.get_dataframe()
        if df is None or df.empty:
            return pd.DataFrame(), []

        anomalies = pd.DataFrame()
        messages = []
        try:
            anomalies = self.anomaly_detector.detect(df)
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
        df = self.get_dataframe()
        if df is None or df.empty:
            return "No data available for trend analysis."

        if "date" not in df.columns or "amount" not in df.columns:
            return "Insufficient data for trend analysis."

        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        monthly = df_copy.groupby(df_copy["date"].dt.to_period("M"))["amount"].sum()

        if len(monthly) < 2:
            # Not enough for a trend, but show the only month
            if len(monthly) == 1:
                month = str(monthly.index[0])
                amount = monthly.iloc[0]
                return f"Only one month of data: {month} total spending was ${amount:.2f}."
            else:
                return "Not enough data for trend detection."

       # Simple trend detection: compare last 2 or 3 months
            last_values = monthly[-3:].values
            if len(last_values) >= 2:
                if all(np.diff(last_values) > 0):
                    return "📈 Spending has increased over the last months."
                elif all(np.diff(last_values) < 0):
                    return "📉 Spending has decreased over the last months."
                else:
                    return "➡️ Spending shows mixed patterns recently."
            else:
                return "Not enough data for trend detection."

    def savings_forecast(self):
        df = self.get_dataframe()
        if df is None or df.empty:
            return None

        if "date" not in df.columns or "amount" not in df.columns:
            return None

        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        monthly = df_copy.groupby(df_copy["date"].dt.to_period("M"))["amount"].sum()

       if len(monthly) < 1:
            return None
        elif len(monthly) == 1:
            # Only one month, so just repeat that value as a "forecast"
            last_month = float(monthly.iloc[0])
            return {
                "last_month": last_month,
                "forecast_next_month": last_month
            }
        else:
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
        df = self.get_dataframe()
        if df is None or df.empty:
            return []

        if "date" not in df.columns or "amount" not in df.columns:
            return []

        alerts = []
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
        weekly = df_copy.groupby(df_copy["date"].dt.to_period("W"))["amount"].sum()

        if len(weekly) < 2:
            # Not enough for a weekly comparison, but show last week
            if len(weekly) == 1:
                week = str(weekly.index[0])
                amount = weekly.iloc[0]
                alerts.append(f"Only one week of data: {week} total spending was ${amount:.2f}.")
            return alerts

        last_week = weekly.iloc[-1]
        avg_prev = weekly.iloc[:-1].mean()

        if last_week > avg_prev * 1.3:
            alerts.append("⚠️ This week's expenses are over 30% higher than your usual average.")
        else:
            alerts.append("✅ This week's expenses are within your usual range.")

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
        
        # Add budget recommendations
        try:
            df = self.get_dataframe()
            insights["budget_recommendations"] = self.budget_recommender.recommend(df)
        except Exception as e:
            insights["budget_recommendations"] = ["Budget recommendations unavailable"]
            
        return insights

    # Keep the original methods for backward compatibility
    def summarize(self, df=None):
        if df is None:
            df = self.get_dataframe()
        if df is None or df.empty:
            return "No transactions yet."
        total_income = df[df["type"] == "income"]["amount"].sum()
        total_expense = df[df["type"] == "expense"]["amount"].sum()
        return {
            "Total Income": round(total_income, 2),
            "Total Expense": round(total_expense, 2),
            "Savings": round(total_income - total_expense, 2)
        }

    def generate_insights(self, df=None):
        if df is None:
            df = self.get_dataframe()
        if df is None or df.empty:
            return ["No data available."]

        # Use the new generate_all method but format for backward compatibility
        all_insights = self.generate_all()
        insights = []
        
        # Format insights for the original method's expected output
        stats = all_insights.get("summary_stats", {})
        if stats:
            income = stats.get("income", 0)
            expense = stats.get("expense", 0)
            if expense > income:
                insights.append("⚠ You spent more than you earned.")
            else:
                insights.append("✅ You saved money overall.")
        
        # Add anomaly messages
        insights.extend(all_insights.get("anomaly_messages", []))
        
        # Add trend insights
        trend = all_insights.get("trend", "")
        if trend:
            insights.append(trend)
            
        # Add budget recommendations
        recs = all_insights.get("budget_recommendations", [])
        insights.extend(recs)
        
        return insights