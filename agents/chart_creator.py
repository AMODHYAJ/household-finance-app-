import os
import matplotlib.pyplot as plt
from collections import defaultdict
from config.settings import CHART_DIR

class ChartCreatorAgent:
    def __init__(self, data_agent):
        self.data_agent = data_agent
        os.makedirs(CHART_DIR, exist_ok=True)

    def pie_expenses_by_category(self, save=True, show=False):
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            print("No data to plot.")
            return
        exp = df[df["type"] == "expense"]
        if exp.empty:
            print("No expenses to plot.")
            return
        totals = exp.groupby("category")["amount"].sum()
        plt.figure()
        plt.pie(totals.values, labels=totals.index, autopct="%1.1f%%")
        plt.title("Expenses by Category")
        # Explainability caption
        caption = f"This pie chart shows the proportion of expenses by category. The largest category is '{totals.idxmax()}' with ${totals.max():.2f}."
        print("[Explainability]", caption)
        path = os.path.join(CHART_DIR, "expenses_by_category.png")
        if save: plt.savefig(path, bbox_inches="tight")
        if show: plt.show()
        plt.close()
        if save: print(f"📊 Saved: {path}")

    def bar_income_vs_expense(self, save=True, show=False):
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            print("No data to plot.")
            return
        income = df[df["type"] == "income"]["amount"].sum()
        expense = df[df["type"] == "expense"]["amount"].sum()
        plt.figure()
        plt.bar(["Income", "Expense"], [income, expense])
        plt.title("Income vs Expense")
        # Explainability caption
        if income > expense:
            caption = f"Bar chart comparing total income (${income:.2f}) and expenses (${expense:.2f}). You saved ${income-expense:.2f} overall."
        else:
            caption = f"Bar chart comparing total income (${income:.2f}) and expenses (${expense:.2f}). You spent ${expense-income:.2f} more than you earned."
        print("[Explainability]", caption)
        path = os.path.join(CHART_DIR, "income_vs_expense.png")
        if save: plt.savefig(path, bbox_inches="tight")
        if show: plt.show()
        plt.close()
        if save: print(f"📊 Saved: {path}")

    def line_monthly_expense_trend(self, save=True, show=False):
        import pandas as pd
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            print("No data to plot.")
            return
        exp = df[df["type"] == "expense"].copy()
        if exp.empty:
            print("No expenses to plot.")
            return
        exp["month"] = pd.to_datetime(exp["date"]).dt.to_period("M").astype(str)
        series = exp.groupby("month")["amount"].sum().sort_index()
        plt.figure()
        plt.plot(list(series.index), list(series.values), marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title("Monthly Expense Trend")
        plt.xlabel("Month")
        plt.ylabel("Amount")
        # Explainability caption
        if len(series) > 1:
            trend = "increased" if series.iloc[-1] > series.iloc[0] else "decreased"
            caption = f"Line chart showing monthly expense trend. Expenses have {trend} from ${series.iloc[0]:.2f} in {series.index[0]} to ${series.iloc[-1]:.2f} in {series.index[-1]}."
        else:
            caption = "Line chart showing monthly expense trend. Not enough data to determine trend."
        print("[Explainability]", caption)
        path = os.path.join(CHART_DIR, "monthly_expense_trend.png")
        if save: plt.savefig(path, bbox_inches="tight")
        if show: plt.show()
        plt.close()
        if save: print(f"📊 Saved: {path}")
