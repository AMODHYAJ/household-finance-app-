import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
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
.chart-container {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.caption-box {
    background-color: #f8f9fa;
    border-left: 4px solid #1f77b4;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0 5px 5px 0;
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
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Expenses by category pie chart
            expenses_df = df[df["type"] == "expense"]
            if not expenses_df.empty:
                category_totals = expenses_df.groupby("category")["amount"].sum()
                fig_pie = px.pie(values=category_totals.values, names=category_totals.index,
                               title="Expenses by Category")
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent transactions
        st.subheader("Recent Transactions")
        recent_df = df.tail(10).sort_values("date", ascending=False)
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No transactions found. Start by adding some transactions!")
        if st.button("Add Your First Transaction"):
            st.session_state.page = "Add Transaction"
            st.rerun()

def add_transaction_page():
    if not check_auth():
        return
        
    st.header("💳 Add Transaction")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            t_type = st.selectbox("Transaction Type", ["income", "expense"])
            category = st.text_input("Category", placeholder="e.g., food, bills, salary")
            amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        
        with col2:
            date_input = st.date_input("Date", value=date.today())
            note = st.text_area("Note (optional)", placeholder="Additional details...")
        
        submitted = st.form_submit_button("Add Transaction")
        
        if submitted:
            if category and amount > 0:
                try:
                    data_agent = st.session_state.architect_agent.data_agent
                    from core.schemas import TransactionIn
                    from core.database import Transaction
                    
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
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No transactions match your filters.")
    else:
        st.info("No transactions found. Add some transactions first!")

def charts_page():
    if not check_auth():
        return
        
    st.header("📈 Advanced Charts & Visualizations")
    
    data_agent = st.session_state.architect_agent.data_agent
    chart_agent = st.session_state.architect_agent.chart_agent
    df = data_agent.get_transactions_df()
    
    if df is not None and not df.empty:
        # Tab layout for different chart types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Basic Charts", 
            "🎯 Interactive Dashboard", 
            "🔍 Comparative Analysis",
            "🤖 Natural Language Query",
            "🔮 Predictive Charts"
        ])
        
        with tab1:
            st.subheader("Basic Financial Charts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                time_range = st.selectbox("Time Range", 
                                         ["All", "Last 3 Months", "Last 6 Months", "This Year"],
                                         key="basic_time_range")
            
            with col2:
                chart_type = st.selectbox("Select Chart Type", [
                    "Expenses by Category",
                    "Income vs Expenses", 
                    "Savings Over Time",
                    "All Basic Charts"
                ])
            
            if st.button("Generate Basic Charts", type="primary"):
                time_range_param = _get_time_range_param(time_range)  # FIX: Remove self.
                
                with st.spinner("Generating charts..."):
                    if chart_type in ["Expenses by Category", "All Basic Charts"]:
                        st.subheader("Expenses by Category")
                        fig, caption = chart_agent.pie_expenses_by_category(save=False, show=False)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_caption(caption)  # FIX: Remove self.
                    
                    if chart_type in ["Income vs Expenses", "All Basic Charts"]:
                        st.subheader("Income vs Expenses")
                        fig, caption = chart_agent.bar_income_vs_expense(time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_caption(caption)  # FIX: Remove self.
                    
                    if chart_type in ["Savings Over Time", "All Basic Charts"]:
                        st.subheader("Savings Over Time")
                        fig, caption = chart_agent.line_savings_over_time(time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_caption(caption)  # FIX: Remove self.
        
        with tab2:
            st.subheader("Interactive Dashboard")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                categories = st.multiselect("Filter Categories", 
                                          options=df['category'].unique() if 'category' in df.columns else [])
            with col2:
                transaction_types = st.multiselect("Filter Types", 
                                                 options=['income', 'expense'])
            with col3:
                date_range = st.date_input("Date Range", 
                                         value=[df['date'].min(), df['date'].max()] if 'date' in df.columns else [])
            
            if st.button("Generate Dashboard"):
                filters = {
                    'category': categories,
                    'type': transaction_types,
                    'date_range': date_range if len(date_range) == 2 else None
                }
                
                fig = chart_agent.create_interactive_dashboard(filters)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Comparative Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                period1 = st.selectbox("First Period", 
                                      ["This Month", "Last Month", "This Year", "Last Year"],
                                      key="period1")
            with col2:
                period2 = st.selectbox("Second Period", 
                                      ["Last Month", "This Month", "Last Year", "This Year"],
                                      key="period2")
            
            if st.button("Compare Periods"):
                fig = chart_agent.comparative_charts(period1, period2)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Natural Language Query")
            
            query = st.text_input("Ask for a chart", 
                                 placeholder="e.g., 'Show me electricity expenses for the last 3 months'")
            
            if st.button("Generate Chart from Query") and query:
                result = chart_agent.natural_language_to_chart(query)
                if isinstance(result, tuple):  # If it returns (fig, caption)
                    st.plotly_chart(result[0], use_container_width=True)
                    if len(result) > 1 and isinstance(result[1], dict):
                        _display_caption(result[1])  # FIX: Remove self.
                else:
                    st.plotly_chart(result, use_container_width=True)
        
        with tab5:
            st.subheader("Predictive Financial Charts")
            
            periods = st.slider("Months to Predict", 1, 12, 6)
            
            if st.button("Generate Predictions"):
                fig = chart_agent.predictive_charts(periods)
                st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.info("No data available for charts. Add some transactions first!")

# Add these helper functions outside the charts_page function
def _display_caption(caption):
    """Display chart caption in a formatted box"""
    if isinstance(caption, dict) and 'summary' in caption:
        st.markdown(f"""
        <div class="caption-box">
            <h4>📋 Chart Explanation</h4>
            <p><strong>What's being compared:</strong> {caption.get('comparison', 'N/A')}</p>
            <p><strong>Chart type reasoning:</strong> {caption.get('reasoning', 'N/A')}</p>
            <p><strong>Key insight:</strong> {caption.get('insights', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

def _get_time_range_param(time_range):
    """Convert time range selection to parameter"""
    range_map = {
        "Last 3 Months": "last_3_months",
        "Last 6 Months": "last_6_months", 
        "This Year": "this_year",
        "All": None
    }
    return range_map.get(time_range, None)

def insights_page():
    if not check_auth():
        return

    st.header("🧠 AI-Powered Insights")

    insight_agent = st.session_state.architect_agent.insight_agent
    data_agent = st.session_state.architect_agent.data_agent
    df = data_agent.get_transactions_df()

    if df is not None and not df.empty:
        # Tabs for different insights
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "🎯 Category Analysis", "🤖 AI Insights"])

        with tab1:
            st.subheader("Financial Summary")
            # ... (existing summary code)

        with tab2:
            st.subheader("Category Analysis")
            # ... (existing category analysis code)

        with tab3:
            st.subheader("AI-Generated Insights")
            # ... (existing AI insights code)
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
