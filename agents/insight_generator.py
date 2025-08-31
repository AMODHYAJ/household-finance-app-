from collections import defaultdict
from config.settings import OPENAI_API_KEY
from datetime import date

class InsightGeneratorAgent:
    def __init__(self, data_agent):
        self.data_agent = data_agent

    def summarize(self):
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            print("No transactions yet.")
            return
        total_income = df[df["type"] == "income"]["amount"].sum()
        total_expense = df[df["type"] == "expense"]["amount"].sum()
        savings = total_income - total_expense
        print("----- Summary -----")
        print(f"Total Income : {total_income:.2f}")
        print(f"Total Expense: {total_expense:.2f}")
        print(f"Savings      : {savings:.2f}")

    def highest_expense_category(self):
        df = self.data_agent.get_transactions_df()
        exp = df[df["type"] == "expense"]
        if exp.empty:
            print("No expenses recorded.")
            return
        grp = exp.groupby("category")["amount"].sum().sort_values(ascending=False)
        top_cat = grp.index[0]
        print(f"Highest expense category: {top_cat} ({grp.iloc[0]:.2f})")

    # (Optional for mid) simple “reasoned” text without API
    def simple_insights(self):
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            print("No transactions to analyze.")
            return
        lines = []
        # savings trend rough check
        income = df[df["type"] == "income"]["amount"].sum()
        expense = df[df["type"] == "expense"]["amount"].sum()
        if expense > income:
            lines.append("You spent more than you earned. Consider a stricter budget next month.")
        else:
            lines.append("Great! You saved money overall. Look for categories to optimize further.")
        # spike detection (very simple)
        import pandas as pd
        exp = df[df["type"] == "expense"].copy()
        if not exp.empty:
            exp["month"] = pd.to_datetime(exp["date"]).dt.to_period("M").astype(str)
            by_month = exp.groupby("month")["amount"].sum().sort_index()
            if len(by_month) >= 2:
                last, prev = by_month.iloc[-1], by_month.iloc[-2]
                delta = last - prev
                if delta > 0 and prev > 0 and delta / prev > 0.25:
                    lines.append(f"Expense spike: last month increased by {delta:.2f} (~{(delta/prev)*100:.1f}%).")
        print("----- Insights -----")
        for L in lines:
            print("-", L)
