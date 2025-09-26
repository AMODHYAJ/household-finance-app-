"""
Responsible AI & Commercialization (Template)
------------------------------------------------
* Fairness: Designs unbiased workflows, regular bias audits.
* Transparency: Documents agent roles, logs actions.
* Explainability: Rationale for design choices is clear.
* Data Protection: Shares only necessary data between agents.
* Commercialization: White-label system design, consulting, custom integrations.
"""
import os
from rich import print
from agents.data_collector import DataCollectorAgent
from agents.chart_creator import ChartCreatorAgent
from agents.insight_generator import InsightGeneratorAgent

class ArchitectAgent:
    def __init__(self):
        self.data_agent = DataCollectorAgent()
        self.chart_agent = ChartCreatorAgent(self.data_agent)
        self.insight_agent = InsightGeneratorAgent(self.data_agent)

    def menu(self):
        print("\n[bold cyan]Household Finance – Mid Evaluation Demo[/bold cyan]")
        print("1) Register")
        print("2) Login")
        print("3) Add Transaction")
        print("4) List Transactions")
        print("5) Charts (save PNGs)")
        print("6) Insights (summary + highest category)")
        print("7) Simple AI-ish insights (offline)")
        print("0) Exit")

    def start(self):
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/exports", exist_ok=True)
        os.makedirs("visualizations/static_charts", exist_ok=True)

        while True:
            self.menu()
            choice = input("Choose: ").strip()
            if choice == "1":
                self.data_agent.register_user()
            elif choice == "2":
                self.data_agent.login_user()
            elif choice == "3":
                self.data_agent.add_transaction()
            elif choice == "4":
                self.data_agent.list_transactions()
            elif choice == "5":
                self.chart_agent.pie_expenses_by_category(save=True, show=False)
                self.chart_agent.bar_income_vs_expense(save=True, show=False)
                self.chart_agent.line_monthly_expense_trend(save=True, show=False)
            elif choice == "6":
                self.insight_agent.summarize()
                self.insight_agent.highest_expense_category()
            elif choice == "7":
                self.insight_agent.simple_insights()
            elif choice == "0":
                print("bye! 👋")
                break
            else:
                print("Invalid option.")
