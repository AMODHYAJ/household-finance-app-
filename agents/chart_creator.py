import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

CHART_DIR = "visualizations/static_charts"  # Temporary hardcoded value

class ChartCreatorAgent:
    def __init__(self, data_agent):
        self.data_agent = data_agent
        os.makedirs(CHART_DIR, exist_ok=True)
        self.chart_descriptions = {}

    # Basic Chart Methods
    def pie_expenses_by_category(self, save=True, show=False, time_range=None):
        """Generate pie chart for category-wise expenses with filtering"""
        df = self._get_filtered_data(time_range)
        if df is None or df.empty:
            return self._create_empty_chart("No data available for pie chart")
        
        exp = df[df["type"] == "expense"]
        if exp.empty:
            return self._create_empty_chart("No expenses data available")
        
        totals = exp.groupby("category")["amount"].sum()
        if totals.empty:
            return self._create_empty_chart("No expense categories found")
        
        fig = px.pie(values=totals.values, names=totals.index, title="Expenses by Category")
        caption = self._generate_caption(
            chart_type="pie",
            comparison="distribution of expenses across different categories",
            reasoning="Pie charts are ideal for showing proportional distribution of categorical data",
            insights=self._get_expense_insights(totals)
        )
        return self._finalize_chart(fig, "expenses_by_category", save, caption)

    def bar_income_vs_expense(self, save=True, show=False, time_range=None):
        """Generate bar chart for monthly income vs expenses"""
        df = self._get_filtered_data(time_range)
        if df is None or df.empty:
            return self._create_empty_chart("No data available for bar chart")
        
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure()
        if 'income' in monthly_data.columns:
            fig.add_trace(go.Bar(name='Income', x=monthly_data.index, y=monthly_data['income'], marker_color='green'))
        if 'expense' in monthly_data.columns:
            fig.add_trace(go.Bar(name='Expense', x=monthly_data.index, y=monthly_data['expense'], marker_color='red'))
        
        fig.update_layout(title="Monthly Income vs Expenses", barmode='group')
        caption = self._generate_caption(
            chart_type="bar",
            comparison="monthly income versus expenses over time",
            reasoning="Bar charts effectively compare categorical data across different periods",
            insights=self._get_income_expense_insights(monthly_data)
        )
        return self._finalize_chart(fig, "income_vs_expense", save, caption)

    def line_savings_over_time(self, save=True, show=False, time_range=None):
        """Generate line graph for savings over time"""
        df = self._get_filtered_data(time_range)
        if df is None or df.empty:
            return self._create_empty_chart("No data available for savings trend")
        
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_totals = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        monthly_totals['savings'] = monthly_totals.get('income', 0) - monthly_totals.get('expense', 0)
        monthly_totals['cumulative_savings'] = monthly_totals['savings'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_totals.index, y=monthly_totals['cumulative_savings'],
                               mode='lines+markers', name='Cumulative Savings', line=dict(color='blue', width=3)))
        fig.update_layout(title="Savings Over Time", xaxis_title="Month", yaxis_title="Amount")
        caption = self._generate_caption(
            chart_type="line",
            comparison="cumulative savings growth over time",
            reasoning="Line charts are perfect for showing trends and progress over time periods",
            insights=self._get_savings_insights(monthly_totals)
        )
        return self._finalize_chart(fig, "savings_over_time", save, caption)

    # Automatic Chart Selection
    def auto_chart_selection(self, x_col, y_col, save=True, show=False):
        """Automatically select the best chart type based on data characteristics"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for automatic chart selection")
        
        # Determine chart type based on data characteristics
        if self._is_time_series(x_col, df):
            return self._create_time_series_chart(df, x_col, y_col, save)
        elif self._is_categorical_numeric(x_col, y_col, df):
            return self._create_bar_chart(df, x_col, y_col, save)
        elif self._is_two_numeric(x_col, y_col, df):
            return self._create_scatter_plot(df, x_col, y_col, save)
        elif self._is_single_numeric(y_col, df):
            return self._create_histogram(df, y_col, save)
        else:
            return self._create_bar_chart(df, x_col, y_col, save)

    # Advanced Features
    def create_interactive_dashboard(self, filters=None):
        """Create an interactive dashboard with filtering capabilities"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for dashboard")
        
        # Apply filters
        if filters:
            df = self._apply_filters(df, filters)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expenses by Category', 'Monthly Trends', 
                          'Income vs Expense', 'Savings Progress'),
            specs=[[{"type": "pie"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Pie chart - Expenses by category
        expenses_df = df[df["type"] == "expense"]
        if not expenses_df.empty:
            category_totals = expenses_df.groupby("category")["amount"].sum()
            fig.add_trace(go.Pie(labels=category_totals.index, values=category_totals.values,
                               name="Expenses"), row=1, col=1)
        
        # Line chart - Monthly trends
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_trends = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        for col in monthly_trends.columns:
            fig.add_trace(go.Scatter(x=monthly_trends.index, y=monthly_trends[col],
                                   mode='lines+markers', name=col.title()),
                         row=1, col=2)
        
        # Bar chart - Income vs Expense
        if 'income' in monthly_trends.columns and 'expense' in monthly_trends.columns:
            fig.add_trace(go.Bar(name='Income', x=monthly_trends.index, 
                               y=monthly_trends['income']), row=2, col=1)
            fig.add_trace(go.Bar(name='Expense', x=monthly_trends.index, 
                               y=monthly_trends['expense']), row=2, col=1)
        
        # Savings progress
        if 'income' in monthly_trends.columns and 'expense' in monthly_trends.columns:
            savings = monthly_trends.get('income', 0) - monthly_trends.get('expense', 0)
            fig.add_trace(go.Scatter(x=monthly_trends.index, y=savings.cumsum(),
                                   mode='lines+markers', name='Cumulative Savings',
                                   line=dict(color='green')), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Financial Dashboard", showlegend=True)
        return fig

    def comparative_charts(self, current_period, previous_period):
        """Generate comparative charts (this month vs last month)"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for comparative analysis")
        
        # Filter data for both periods
        current_df = self._filter_by_period(df, current_period)
        previous_df = self._filter_by_period(df, previous_period)
        
        # Create comparative bar chart
        fig = go.Figure()
        
        # Compare expenses by category
        current_expenses = current_df[current_df['type'] == 'expense'].groupby('category')['amount'].sum()
        previous_expenses = previous_df[previous_df['type'] == 'expense'].groupby('category')['amount'].sum()
        
        categories = list(set(current_expenses.index) | set(previous_expenses.index))
        
        current_values = [current_expenses.get(cat, 0) for cat in categories]
        previous_values = [previous_expenses.get(cat, 0) for cat in categories]
        
        fig.add_trace(go.Bar(name=current_period, x=categories, y=current_values))
        fig.add_trace(go.Bar(name=previous_period, x=categories, y=previous_values))
        
        fig.update_layout(title=f"Expense Comparison: {current_period} vs {previous_period}",
                         barmode='group')
        
        return fig

    def natural_language_to_chart(self, query):
        """Convert natural language queries to charts"""
        query = query.lower()
        
        # Parse the query
        if any(word in query for word in ['electricity', 'utility', 'bill']):
            return self._create_utility_chart(query)
        elif any(word in query for word in ['month', 'last', 'recent']):
            return self._create_time_based_chart(query)
        elif any(word in query for word in ['category', 'categories']):
            return self.pie_expenses_by_category()
        elif any(word in query for word in ['income', 'expense', 'comparison']):
            return self.bar_income_vs_expense()
        elif any(word in query for word in ['savings', 'trend', 'progress']):
            return self.line_savings_over_time()
        else:
            return self.create_interactive_dashboard()

    def predictive_charts(self, periods=6):
        """Generate predictive charts for future expenses/savings"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for predictions")
        
        # Prepare data for forecasting
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        monthly_data = monthly_data.sort_index()
        
        # Create future predictions using linear regression
        future_months = self._generate_future_months(monthly_data.index[-1], periods)
        
        fig = go.Figure()
        
        # Plot historical data and predictions for each type
        for col in ['income', 'expense']:
            if col in monthly_data.columns:
                historical_values = monthly_data[col].values
                
                # Simple linear regression for prediction
                if len(historical_values) > 1:
                    X = np.arange(len(historical_values)).reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, historical_values)
                    
                    # Predict future values
                    future_X = np.arange(len(historical_values), len(historical_values) + periods).reshape(-1, 1)
                    future_values = model.predict(future_X)
                    
                    # Combine historical and future data
                    all_months = list(monthly_data.index) + future_months
                    all_values = list(historical_values) + list(future_values)
                    
                    fig.add_trace(go.Scatter(x=all_months, y=all_values, 
                                           mode='lines+markers', name=f'{col.title()}',
                                           line=dict(dash='solid' if col == 'income' else 'dash')))
        
        fig.update_layout(title=f"Financial Projections (Next {periods} Months)")
        return fig

    def _create_time_based_chart(self, query):
        """Generate a time-based chart (e.g., line chart) based on natural language query"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for time-based chart")

        # Parse time range from query
        time_range = None
        if 'last 3 months' in query or 'last quarter' in query:
            time_range = 'last_3_months'
        elif 'last 6 months' in query:
            time_range = 'last_6_months'
        elif 'this year' in query:
            time_range = 'this_year'
        elif 'last month' in query:
            time_range = 'last_month'
        else:
            time_range = 'all'

        # Filter data based on time range
        df_copy = self._get_filtered_data(time_range) if time_range != 'last_month' else self._filter_by_period(df.copy(), 'last_month')

        if df_copy.empty:
            return self._create_empty_chart("No data available for the specified time range")

        # Determine chart focus (e.g., expenses, income, or savings)
        if any(word in query for word in ['expense', 'expenses']):
            df_focus = df_copy[df_copy['type'] == 'expense'].groupby('date')['amount'].sum().reset_index()
            chart_type = "expenses"
        elif any(word in query for word in ['income']):
            df_focus = df_copy[df_copy['type'] == 'income'].groupby('date')['amount'].sum().reset_index()
            chart_type = "income"
        else:
            # Default to savings-like trend if no specific type is mentioned
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
            monthly_totals = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
            df_focus = pd.DataFrame({'date': monthly_totals.index, 'savings': monthly_totals.get('income', 0) - monthly_totals.get('expense', 0)})
            chart_type = "savings"

        # Create a line chart for the time-based data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_focus['date'], y=df_focus['amount'] if 'amount' in df_focus else df_focus['savings'],
                               mode='lines+markers', name=f"{chart_type.title()} Trend",
                               line=dict(color='blue' if chart_type == 'savings' else 'green' if chart_type == 'income' else 'red')))

        fig.update_layout(title=f"{chart_type.title()} Trend Over Time ({time_range.replace('_', ' ').title() if time_range else 'All Data'})",
                         xaxis_title="Date",
                         yaxis_title="Amount")

        # Generate caption
        caption = self._generate_caption(
            chart_type="line",
            comparison=f"{chart_type} trends over the selected time period",
            reasoning="Line charts are ideal for visualizing trends over time",
            insights=f"The {chart_type} trend shows {'growth' if (df_focus['amount'].iloc[-1] > df_focus['amount'].iloc[0] if 'amount' in df_focus else df_focus['savings'].iloc[-1] > df_focus['savings'].iloc[0]) else 'decline' if (df_focus['amount'].iloc[-1] < df_focus['amount'].iloc[0] if 'amount' in df_focus else df_focus['savings'].iloc[-1] < df_focus['savings'].iloc[0]) else 'stability'} over time."
        )

        return self._finalize_chart(fig, f"time_based_{chart_type}", save=False, caption=caption)

    def _create_utility_chart(self, query):
        """Generate a chart focused on utility-related expenses (e.g., electricity, bills)"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for utility chart")

        # Filter for utility-related categories, handling NaN without na parameter
        df_copy = df.copy()
        df_copy['category'] = df_copy['category'].fillna('')
        utility_df = df_copy[df_copy['category'].str.lower().isin(['electricity', 'utility', 'bill'])]

        if utility_df.empty:
            return self._create_empty_chart("No utility-related data available")

        # Aggregate by date for a time-based view
        utility_df['date'] = pd.to_datetime(utility_df['date'])
        utility_totals = utility_df.groupby('date')['amount'].sum().reset_index()

        # Create a line chart for utility expenses over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=utility_totals['date'], y=utility_totals['amount'],
                               mode='lines+markers', name='Utility Expenses',
                               line=dict(color='orange')))

        fig.update_layout(title="Utility Expenses Over Time",
                         xaxis_title="Date",
                         yaxis_title="Amount")

        caption = self._generate_caption(
            chart_type="line",
            comparison="utility expenses over time",
            reasoning="Line charts are suitable for tracking expense trends",
            insights="Utility costs show fluctuations that may correlate with usage patterns."
        )

        return self._finalize_chart(fig, "utility_expenses", save=False, caption=caption)

    # Helper Methods
    def _get_filtered_data(self, time_range=None):
        """Get filtered data based on time range"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return df
        
        if time_range:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            if time_range == 'last_3_months':
                cutoff_date = datetime.now() - timedelta(days=90)
                df = df_copy[df_copy['date'] >= cutoff_date]
            elif time_range == 'last_6_months':
                cutoff_date = datetime.now() - timedelta(days=180)
                df = df_copy[df_copy['date'] >= cutoff_date]
            elif time_range == 'this_year':
                current_year = datetime.now().year
                df = df_copy[df_copy['date'].dt.year == current_year]
        
        return df

    def _generate_caption(self, chart_type, comparison, reasoning, insights):
        """Generate automatic captions for charts"""
        caption = {
            'chart_type': chart_type,
            'comparison': comparison,
            'reasoning': reasoning,
            'insights': insights,
            'summary': f"This {chart_type} chart shows {comparison}. {reasoning}. Key insight: {insights}"
        }
        return caption

    def _finalize_chart(self, fig, chart_name, save, caption):
        """Finalize chart with common settings and save if needed"""
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        if save:
            path = os.path.join(CHART_DIR, f"{chart_name}.html")
            fig.write_html(path)
            print(f"📊 Saved: {path}")
        
        # Store caption for later retrieval
        self.chart_descriptions[chart_name] = caption
        
        return fig, caption

    def _create_empty_chart(self, message):
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper",
                          x=0.5, y=0.5, xanchor="center", yanchor="middle",
                          showarrow=False, font=dict(size=16))
        fig.update_layout(title=message)
        return fig, {"error": message}

    # Insight generation methods
    def _get_expense_insights(self, category_totals):
        """Generate insights from expense data"""
        if len(category_totals) == 0:
            return "No expense data available"
        
        top_category = category_totals.idxmax()
        top_amount = category_totals.max()
        total_expenses = category_totals.sum()
        percentage = (top_amount / total_expenses) * 100
        
        return f"'{top_category}' is your largest expense category at ${top_amount:.2f} ({percentage:.1f}% of total expenses)"

    def _get_income_expense_insights(self, monthly_data):
        """Generate insights from income vs expense data"""
        if 'income' not in monthly_data.columns or 'expense' not in monthly_data.columns:
            return "Insufficient data for income-expense comparison"
        
        avg_income = monthly_data['income'].mean()
        avg_expense = monthly_data['expense'].mean()
        savings_rate = ((avg_income - avg_expense) / avg_income * 100) if avg_income > 0 else 0
        
        return f"Average monthly income: ${avg_income:.2f}, expenses: ${avg_expense:.2f}, savings rate: {savings_rate:.1f}%"

    def _get_savings_insights(self, monthly_totals):
        """Generate insights from savings data"""
        if 'savings' not in monthly_totals.columns:
            return "No savings data available"
        
        current_savings = monthly_totals['cumulative_savings'].iloc[-1] if len(monthly_totals) > 0 else 0
        avg_monthly_savings = monthly_totals['savings'].mean() if len(monthly_totals) > 0 else 0
        
        return f"Current savings: ${current_savings:.2f}, average monthly savings: ${avg_monthly_savings:.2f}"

    # Automatic chart selection helpers
    def _is_time_series(self, col, df):
        return pd.api.types.is_datetime64_any_dtype(df[col]) if col in df.columns else False

    def _is_categorical_numeric(self, x_col, y_col, df):
        return (x_col in df.columns and y_col in df.columns and 
                not pd.api.types.is_numeric_dtype(df[x_col]) and 
                pd.api.types.is_numeric_dtype(df[y_col]))

    def _is_two_numeric(self, x_col, y_col, df):
        return (x_col in df.columns and y_col in df.columns and 
                pd.api.types.is_numeric_dtype(df[x_col]) and 
                pd.api.types.is_numeric_dtype(df[y_col]))

    def _is_single_numeric(self, col, df):
        return col in df.columns and pd.api.types.is_numeric_dtype(df[col])

    def _create_time_series_chart(self, df, x_col, y_col, save):
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over Time")
        return self._finalize_chart(fig, "time_series", save, "Time series analysis")

    def _create_bar_chart(self, df, x_col, y_col, save):
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        return self._finalize_chart(fig, "bar_chart", save, "Categorical comparison")

    def _create_scatter_plot(self, df, x_col, y_col, save):
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        return self._finalize_chart(fig, "scatter_plot", save, "Numeric relationship")

    def _create_histogram(self, df, col, save):
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        return self._finalize_chart(fig, "histogram", save, "Data distribution")

    def _apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        df_copy = df.copy()
        if 'category' in filters and filters['category']:
            # Handle NaN values explicitly and remove na parameter
            df_copy['category'] = df_copy['category'].fillna('')
            df_copy = df_copy[df_copy['category'].isin(filters['category'])]
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            df_copy = df_copy[(df_copy['date'] >= start_date) & (df_copy['date'] <= end_date)]
        if 'type' in filters and filters['type']:
            # Handle NaN values explicitly and remove na parameter
            df_copy['type'] = df_copy['type'].fillna('')
            df_copy = df_copy[df_copy['type'].isin(filters['type'])]
        return df_copy

    def _filter_by_period(self, df, period):
        """Filter dataframe by time period"""
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        if period == 'this_month':
            current_date = datetime.now()
            return df_copy[(df_copy['date'].dt.month == current_date.month) & 
                          (df_copy['date'].dt.year == current_date.year)]
        elif period == 'last_month':
            current_date = datetime.now()
            last_month = current_date.month - 1 if current_date.month > 1 else 12
            year = current_date.year if current_date.month > 1 else current_date.year - 1
            return df_copy[(df_copy['date'].dt.month == last_month) & 
                          (df_copy['date'].dt.year == year)]
        elif period == 'this_year':
            current_year = datetime.now().year
            return df_copy[df_copy['date'].dt.year == current_year]
        return df_copy

    def _generate_future_months(self, last_month, periods):
        """Generate future month labels for predictions"""
        last_date = pd.to_datetime(last_month + '-01')
        future_months = []
        for i in range(1, periods + 1):
            future_date = last_date + pd.DateOffset(months=i)
            future_months.append(future_date.strftime('%Y-%m'))
        return future_months
