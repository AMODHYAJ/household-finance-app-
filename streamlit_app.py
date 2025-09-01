import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import os
import sys
from PIL import Image
import io

# Add the project root to Python path to import your agents
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your agents
from agents.architect_agent import ArchitectAgent
from agents.data_collector import DataCollectorAgent
from agents.chart_creator import ChartCreatorAgent
from agents.insight_generator import InsightGeneratorAgent

# Page configuration
st.set_page_config(
    page_title="Household Finance AI Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'architect_agent' not in st.session_state:
    st.session_state.architect_agent = ArchitectAgent()
    
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
    
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.success-message {
    color: #28a745;
    font-weight: bold;
}
.error-message {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">💰 Household Finance AI Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        if not st.session_state.logged_in:
            sidebar_page = st.selectbox("Choose Action", ["Login", "Register"])
        else:
            st.success(f"Welcome, {st.session_state.current_user}!")
            sidebar_page = st.selectbox("Choose Action", [
                "Dashboard", 
                "Add Transaction", 
                "View Transactions", 
                "Charts & Visualizations", 
                "AI Insights",
                "Logout"
            ])

    # Prioritize session_state.page for navigation
    if 'page' in st.session_state and st.session_state.page:
        page = st.session_state.page
        st.session_state.page = None  # Reset after navigation
    else:
        page = sidebar_page

    # Route to appropriate page
    if page == "Register":
        register_page()
    elif page == "Login":
        login_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Add Transaction":
        add_transaction_page()
    elif page == "View Transactions":
        view_transactions_page()
    elif page == "Charts & Visualizations":
        charts_page()
    elif page == "AI Insights":
        insights_page()
    elif page == "Logout":
        logout()

def register_page():
    st.header("🔐 Register New User")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
        
        with col2:
            create_household = st.checkbox("Create New Household")
            household_name = st.text_input("Household Name", disabled=not create_household)
            
            if not create_household:
                # Get existing households
                data_agent = st.session_state.architect_agent.data_agent
                try:
                    from core.database import Household
                    households = data_agent.db.query(Household).all()
                    household_options = {f"{h.id}: {h.name}": h.id for h in households}
                    household_options["None"] = 0
                    
                    selected_household = st.selectbox("Select Household", list(household_options.keys()))
                    household_id = household_options[selected_household]
                except Exception as e:
                    st.warning("Could not load households")
                    household_id = 0
        
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if username and password:
                try:
                    from sqlalchemy.exc import IntegrityError
                    from core.database import User, Household
                    from core.security import hash_password
                    
                    data_agent = st.session_state.architect_agent.data_agent
                    household = None
                    
                    # Handle household creation/selection
                    if create_household and household_name.strip():
                        household = Household(name=household_name.strip())
                        data_agent.db.add(household)
                        data_agent.db.flush()  # Get ID without commit
                    elif not create_household and household_id > 0:
                        household = data_agent.db.query(Household).get(household_id)
                    
                    # Create user
                    user = User(
                        username=username,
                        password_hash=hash_password(password),
                        household=household
                    )
                    data_agent.db.add(user)
                    data_agent.db.commit()
                    
                    st.success("✅ Registration successful! Please login.")
                    st.balloons()
                    
                except IntegrityError:
                    data_agent.db.rollback()
                    st.error("❌ Username already exists.")
                except Exception as e:
                    data_agent.db.rollback()
                    st.error(f"❌ Registration failed: {str(e)}")
            else:
                st.error("Please fill in all required fields.")

def login_page():
    st.header("🔐 User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username and password:
                try:
                    # Get data agent from your backend
                    data_agent = st.session_state.architect_agent.data_agent
                    
                    # Use your existing login logic
                    from core.database import User
                    from core.security import verify_password
                    
                    user = data_agent.db.query(User).filter(User.username == username).first()
                    if user and verify_password(password, user.password_hash):
                        # Set the current user in the data agent
                        data_agent.current_user = user
                        
                        # Update session state
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")
                        
                except Exception as e:
                    st.error(f"❌ Login failed: {str(e)}")
            else:
                st.error("Please enter both username and password.")

def dashboard_page():
    if not check_auth():
        return
        
    st.header("📊 Financial Dashboard")
    
    # Get transactions data
    data_agent = st.session_state.architect_agent.data_agent
    df = data_agent.get_transactions_df()
    
    if df is not None and not df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_income = df[df["type"] == "income"]["amount"].sum()
        total_expense = df[df["type"] == "expense"]["amount"].sum()
        savings = total_income - total_expense
        transaction_count = len(df)
        
        with col1:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col2:
            st.metric("Total Expenses", f"${total_expense:,.2f}")
        with col3:
            st.metric("Net Savings", f"${savings:,.2f}", delta=f"{savings:,.2f}")
        with col4:
            st.metric("Transactions", transaction_count)
        
        # Quick charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Income vs Expense bar chart
            fig_bar = go.Figure(data=[
                go.Bar(name='Amount', x=['Income', 'Expenses'], 
                      y=[total_income, total_expense],
                      marker_color=['green', 'red'])
            ])
            fig_bar.update_layout(title="Income vs Expenses", height=400)
            st.plotly_chart(fig_bar, width='stretch')
        
        with col2:
            # Expenses by category pie chart
            expenses_df = df[df["type"] == "expense"]
            if not expenses_df.empty:
                category_totals = expenses_df.groupby("category")["amount"].sum()
                fig_pie = px.pie(values=category_totals.values, names=category_totals.index,
                               title="Expenses by Category")
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, width='stretch')
        
        # Recent transactions
        st.subheader("Recent Transactions")
        recent_df = df.tail(10).sort_values("date", ascending=False)
        st.dataframe(recent_df, width='stretch')
    else:
        st.info("No transactions found. Start by adding some transactions!")
        if st.button("Add Your First Transaction"):
            st.session_state.page = "Add Transaction"
            st.rerun()

def add_transaction_page():
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            t_type = st.selectbox("Transaction Type", ["income", "expense"])
            category = st.text_input("Category", placeholder="e.g., food, bills, salary")
            amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        with col2:
            date_input = st.date_input("Date", value=date.today())
            note = st.text_area("Note (optional)", placeholder="Additional details...")
        submitted = st.form_submit_button("Add Transaction", width='stretch')
    if submitted:
        if category and amount > 0:
            try:
                data_agent = st.session_state.architect_agent.data_agent
                from core.schemas import TransactionIn
                from core.database import Transaction
                from core.utils import parse_date
                transaction_data = TransactionIn(
                    t_type=t_type,
                    category=category,
                    amount=float(amount),
                    date=date_input,
                    note=note if note.strip() else None
                )
                tx = Transaction(
                    user_id=data_agent.current_user.id,
                    t_type=transaction_data.t_type,
                    category=transaction_data.category,
                    amount=transaction_data.amount,
                    date=transaction_data.date,
                    note=transaction_data.note
                )
                data_agent.db.add(tx)
                data_agent.db.commit()
                st.success("✅ Transaction added successfully!")
                st.balloons()
                st.info(f"Added: {t_type.title()} - {category} - ${amount:.2f} on {date_input}")
            except Exception as e:
                st.error(f"❌ Failed to add transaction: {str(e)}")
                try:
                    data_agent.db.rollback()
                except Exception:
                    pass
        else:
            st.error("Please fill in all required fields.")

def view_transactions_page():
    if not check_auth():
        return
        
    st.header("📝 View Transactions")
    
    data_agent = st.session_state.architect_agent.data_agent
    df = data_agent.get_transactions_df()
    
    if df is not None and not df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            type_filter = st.selectbox("Filter by Type", ["All", "income", "expense"])
        with col2:
            categories = ["All"] + list(df["category"].unique())
            category_filter = st.selectbox("Filter by Category", categories)
        with col3:
            sort_by = st.selectbox("Sort by", ["date", "amount", "category"])
        
        # Apply filters
        filtered_df = df.copy()
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df["type"] == type_filter]
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]
        
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
        
        # Display results
        st.subheader(f"Transactions ({len(filtered_df)} found)")
        
        # Summary of filtered data
        if not filtered_df.empty:
            total_amount = filtered_df["amount"].sum()
            avg_amount = filtered_df["amount"].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Amount", f"${total_amount:,.2f}")
            with col2:
                st.metric("Average Amount", f"${avg_amount:,.2f}")
        
        # Transaction table with better formatting
        if not filtered_df.empty:
            display_df = filtered_df.copy()
            display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, width='stretch')
        else:
            st.info("No transactions match your filters.")
    else:
        st.info("No transactions found. Add some transactions first!")

def charts_page():
    if not check_auth():
        return
        
    st.header("📈 Charts & Visualizations")
    
    data_agent = st.session_state.architect_agent.data_agent
    chart_agent = st.session_state.architect_agent.chart_agent
    df = data_agent.get_transactions_df()
    
    if df is not None and not df.empty:
        # Chart options
        chart_type = st.selectbox("Select Chart Type", [
            "Expenses by Category (Pie)",
            "Income vs Expense (Bar)",
            "Monthly Expense Trend (Line)",
            "All Charts"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            show_chart = st.button("Generate Chart", type="primary")
        with col2:
            save_chart = st.button("Save Charts as PNG")
        
        if show_chart or save_chart:
            if chart_type == "Expenses by Category (Pie)" or chart_type == "All Charts":
                expenses_df = df[df["type"] == "expense"]
                if not expenses_df.empty:
                    category_totals = expenses_df.groupby("category")["amount"].sum()
                    fig = px.pie(values=category_totals.values, names=category_totals.index,
                               title="Expenses by Category")
                    st.plotly_chart(fig, width='stretch')
            
            if chart_type == "Income vs Expense (Bar)" or chart_type == "All Charts":
                income = df[df["type"] == "income"]["amount"].sum()
                expense = df[df["type"] == "expense"]["amount"].sum()
                fig = go.Figure(data=[
                    go.Bar(x=['Income', 'Expenses'], y=[income, expense],
                          marker_color=['green', 'red'])
                ])
                fig.update_layout(title="Income vs Expenses")
                st.plotly_chart(fig, width='stretch')
            
            if chart_type == "Monthly Expense Trend (Line)" or chart_type == "All Charts":
                expenses_df = df[df["type"] == "expense"].copy()
                if not expenses_df.empty:
                    expenses_df["date"] = pd.to_datetime(expenses_df["date"])
                    expenses_df["month"] = expenses_df["date"].dt.to_period("M").astype(str)
                    monthly_expenses = expenses_df.groupby("month")["amount"].sum().sort_index()
                    
                    fig = px.line(x=monthly_expenses.index, y=monthly_expenses.values,
                                title="Monthly Expense Trend")
                    fig.update_xaxes(title="Month")
                    fig.update_yaxes(title="Amount ($)")
                    st.plotly_chart(fig, width='stretch')
        
        if save_chart:
            try:
                # Generate and save static charts using your chart agent
                chart_agent.pie_expenses_by_category(save=True, show=False)
                chart_agent.bar_income_vs_expense(save=True, show=False)
                chart_agent.line_monthly_expense_trend(save=True, show=False)
                st.success("📊 Charts saved to visualizations/static_charts/")
            except Exception as e:
                st.error(f"Failed to save charts: {str(e)}")
                
    else:
        st.info("No data available for charts. Add some transactions first!")

def insights_page():
    if not check_auth():
        return
        
    st.header("🧠 AI-Powered Insights")
    
    insight_agent = st.session_state.architect_agent.insight_agent
    data_agent = st.session_state.architect_agent.data_agent
    df = data_agent.get_transactions_df()
    
    if df is not None and not df.empty:
        # Insight tabs
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "🎯 Category Analysis", "🤖 AI Insights"])
        
        with tab1:
            st.subheader("Financial Summary")
            
            total_income = df[df["type"] == "income"]["amount"].sum()
            total_expense = df[df["type"] == "expense"]["amount"].sum()
            savings = total_income - total_expense
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Income", f"${total_income:,.2f}")
            with col2:
                st.metric("Total Expenses", f"${total_expense:,.2f}")
            with col3:
                savings_color = "normal" if savings >= 0 else "inverse"
                st.metric("Net Savings", f"${savings:,.2f}", 
                         delta=f"${savings:,.2f}", delta_color=savings_color)
            
            # Savings rate
            if total_income > 0:
                savings_rate = (savings / total_income) * 100
                st.metric("Savings Rate", f"{savings_rate:.1f}%")
        
        with tab2:
            st.subheader("Category Analysis")
            
            expenses_df = df[df["type"] == "expense"]
            if not expenses_df.empty:
                category_analysis = expenses_df.groupby("category")["amount"].agg(['sum', 'mean', 'count'])
                category_analysis.columns = ['Total', 'Average', 'Count']
                category_analysis = category_analysis.sort_values('Total', ascending=False)
                
                st.dataframe(category_analysis.style.format({
                    'Total': '${:,.2f}',
                    'Average': '${:,.2f}'
                }), width='stretch')
                
                # Highest expense category
                top_category = category_analysis.index[0]
                top_amount = category_analysis.loc[top_category, 'Total']
                st.info(f"🎯 Highest expense category: *{top_category}* (${top_amount:,.2f})")
        
        with tab3:
            st.subheader("AI-Generated Insights")
            
            if st.button("Generate AI Insights", type="primary"):
                with st.spinner("Analyzing your financial data..."):
                    # Generate insights using your insight agent
                    insights = []
                    
                    # Basic insights
                    if total_expense > total_income:
                        insights.append("⚠ You spent more than you earned. Consider creating a stricter budget for next month.")
                    else:
                        insights.append("✅ Great! You saved money overall. Look for categories to optimize further.")
                    
                    # Category insights
                    if not expenses_df.empty:
                        top_categories = expenses_df.groupby("category")["amount"].sum().nlargest(3)
                        insights.append(f"💡 Your top 3 expense categories are: {', '.join(top_categories.index.tolist())}")
                    
                    # Trend analysis
                    if len(df) >= 5:
                        recent_expenses = expenses_df.tail(5)["amount"].mean()
                        all_expenses_avg = expenses_df["amount"].mean()
                        if recent_expenses > all_expenses_avg * 1.2:
                            insights.append("📈 Your recent expenses are higher than average. Monitor your spending closely.")
                        elif recent_expenses < all_expenses_avg * 0.8:
                            insights.append("📉 Good news! Your recent expenses are below average.")
                    
                    # Display insights
                    for i, insight in enumerate(insights, 1):
                        st.write(f"{i}. {insight}")
                    
                    # Additional recommendations
                    st.subheader("💡 Recommendations")
                    recommendations = [
                        "Set up automatic savings transfers to build an emergency fund",
                        "Review and negotiate recurring subscriptions and bills",
                        "Consider using the 50/30/20 budgeting rule",
                        "Track daily expenses to identify spending patterns"
                    ]
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
    else:
        st.info("No data available for insights. Add some transactions first!")

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.architect_agent.data_agent.current_user = None
    st.success("Successfully logged out!")
    st.rerun()

def check_auth():
    if not st.session_state.logged_in:
        st.warning("Please log in to access this page.")
        return False
    return True

if __name__ == "__main__":
    main()