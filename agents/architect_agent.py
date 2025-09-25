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
        print("\n[bold cyan]Household Finance – Advanced AI Platform[/bold cyan]")
        print("1) Register")
        print("2) Login")
        print("3) Add Transaction")
        print("4) List Transactions")
        print("5) Basic Charts (save PNGs)")
        print("6) Advanced Charts & Dashboards")
        print("7) Insights (summary + highest category)")
        print("8) Natural Language Chart Queries")
        print("0) Exit")

    def start(self):
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/exports", exist_ok=True)
        os.makedirs("visualizations/static_charts", exist_ok=True)
        os.makedirs("visualizations/interactive_charts", exist_ok=True)

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
                self._generate_basic_charts()
            elif choice == "6":
                self._generate_advanced_charts()
            elif choice == "7":
                self.insight_agent.summarize()
                self.insight_agent.highest_expense_category()
            elif choice == "8":
                self._handle_natural_language_queries()
            elif choice == "0":
                print("bye! 👋")
                break
            else:
                print("Invalid option.")

    def _generate_basic_charts(self):
        """Generate basic static charts"""
        if not self._check_auth():
            return
            
        print("\n📊 Generating Basic Charts...")
        
        try:
            # Basic charts using the enhanced chart agent
            self.chart_agent.pie_expenses_by_category(save=True, show=False)
            self.chart_agent.bar_income_vs_expense(save=True, show=False)
            self.chart_agent.line_savings_over_time(save=True, show=False)
            
            print("✅ Basic charts generated successfully!")
            print("📍 Charts saved to: visualizations/static_charts/")
            
        except Exception as e:
            print(f"❌ Error generating basic charts: {e}")

    def _generate_advanced_charts(self):
        """Generate advanced interactive charts and dashboards"""
        if not self._check_auth():
            return
            
        print("\n🎯 Generating Advanced Charts & Dashboards...")
        
        try:
            # Interactive Dashboard
            print("📈 Creating Interactive Dashboard...")
            dashboard_fig = self.chart_agent.create_interactive_dashboard()
            dashboard_path = "visualizations/interactive_charts/dashboard.html"
            dashboard_fig.write_html(dashboard_path)
            print(f"✅ Dashboard saved: {dashboard_path}")
            
            # Comparative Analysis
            print("🔍 Generating Comparative Analysis...")
            comparative_fig = self.chart_agent.comparative_charts("this_month", "last_month")
            comparative_path = "visualizations/interactive_charts/comparative_analysis.html"
            comparative_fig.write_html(comparative_path)
            print(f"✅ Comparative analysis saved: {comparative_path}")
            
            # Predictive Charts
            print("🔮 Generating Predictive Charts...")
            predictive_fig = self.chart_agent.predictive_charts(periods=6)
            predictive_path = "visualizations/interactive_charts/predictive_charts.html"
            predictive_fig.write_html(predictive_path)
            print(f"✅ Predictive charts saved: {predictive_path}")
            
            print("📍 All advanced charts saved to: visualizations/interactive_charts/")
            
        except Exception as e:
            print(f"❌ Error generating advanced charts: {e}")

    def _handle_natural_language_queries(self):
        """Handle natural language chart requests"""
        if not self._check_auth():
            return
            
        print("\n🗣️ Natural Language Chart Queries")
        print("Examples:")
        print("  - 'Show me electricity expenses for the last 3 months'")
        print("  - 'Compare income and expenses this month vs last month'")
        print("  - 'Display savings trend over time'")
        print("  - 'Show expenses by category'")
        print("  - 'Predict my expenses for next 6 months'")
        print("  - Type 'back' to return to main menu")
        
        while True:
            query = input("\nEnter your chart query: ").strip()
            
            if query.lower() == 'back':
                break
                
            if not query:
                continue
                
            try:
                print(f"🔍 Processing: '{query}'")
                
                # Generate chart based on natural language query
                result = self.chart_agent.natural_language_to_chart(query)
                
                if isinstance(result, tuple):
                    fig, caption = result
                    print("✅ Chart generated successfully!")
                    
                    # Display caption information
                    if isinstance(caption, dict):
                        print("\n📋 Chart Explanation:")
                        print(f"   What's compared: {caption.get('comparison', 'N/A')}")
                        print(f"   Chart type: {caption.get('chart_type', 'N/A')}")
                        print(f"   Reasoning: {caption.get('reasoning', 'N/A')}")
                        print(f"   Insight: {caption.get('insights', 'N/A')}")
                    
                    # Save the chart
                    query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    filename = f"nl_query_{query_safe[:30]}.html"
                    filepath = f"visualizations/interactive_charts/{filename}"
                    fig.write_html(filepath)
                    print(f"💾 Chart saved: {filepath}")
                    
                else:
                    fig = result
                    print("✅ Chart generated successfully!")
                    
                    # Save the chart
                    query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    filename = f"nl_query_{query_safe[:30]}.html"
                    filepath = f"visualizations/interactive_charts/{filename}"
                    fig.write_html(filepath)
                    print(f"💾 Chart saved: {filepath}")
                    
            except Exception as e:
                print(f"❌ Error processing query: {e}")

    def _check_auth(self):
        """Check if user is authenticated"""
        if not self.data_agent.current_user:
            print("❌ Please login first.")
            return False
        return True

    # Backward compatibility methods
    def simple_insights(self):
        """Wrapper for simple insights (backward compatibility)"""
        self.insight_agent.summarize()
        self.insight_agent.highest_expense_category()
