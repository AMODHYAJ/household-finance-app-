import pandas as pd

class BudgetRecommender:
    def recommend(self, df):
        if df is None or df.empty:
            return ["No data for budget recommendations."]

        expenses = df[df["type"] == "expense"]
        if expenses.empty:
            return ["No expenses found."]

        recs = []
        by_cat = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
        top_cat = by_cat.index[0]

        recs.append(f"💡 Most of your spending is in '{top_cat}'. Consider reducing it by 10%.")
        if expenses["amount"].max() > expenses["amount"].mean() * 2:
            recs.append("⚠ You have some unusually high expenses. Review them for possible savings.")
        recs.append("✅ Set aside at least 20% of your income as savings.")

        return recs
