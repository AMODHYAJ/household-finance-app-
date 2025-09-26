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

# Custom CSS - Update this section
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
.caption-box {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #333333;
}
.caption-box h4 {
    color: #1f77b4;
    margin-bottom: 1rem;
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 0.5rem;
}
.caption-box p {
    margin-bottom: 0.8rem;
    line-height: 1.5;
}
.caption-box strong {
    color: #2c3e50;
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
            # Get user role
            data_agent = st.session_state.architect_agent.data_agent
            from core.database import User
            user_obj = data_agent.db.query(User).filter_by(username=st.session_state.current_user).first()
            role = getattr(user_obj, 'role', None) if user_obj else None
            if role == "admin":
                nav_options = ["Admin Dashboard", "Logout"]
            else:
                nav_options = [
                    "Dashboard", 
                    "Add Transaction", 
                    "View Transactions", 
                    "Charts & Visualizations", 
                    "AI Insights",
                    "Delete My Account & Data",
                    "Logout"
                ]
            sidebar_page = st.selectbox("Choose Action", nav_options)

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
    elif page == "Admin Dashboard":
        admin_dashboard_page()
    elif page == "Delete My Account & Data":
        delete_my_account_page()
    elif page == "Logout":
        logout()

def delete_my_account_page():
    st.header("Delete My Account & Data")
    st.warning("This will permanently delete your account and all your transactions. This action cannot be undone.")
    if st.button("Delete My Account & Data"):
        data_agent = st.session_state.architect_agent.data_agent
        from core.database import User, Transaction
        user_obj = data_agent.db.query(User).filter_by(username=st.session_state.current_user).first()
        if user_obj:
            # Delete all transactions for this user
            data_agent.db.query(Transaction).filter_by(user_id=user_obj.id).delete()
            data_agent.db.delete(user_obj)
            data_agent.db.commit()
            st.success("Your account and all data have been deleted.")
            st.session_state.current_user = None
            st.session_state.logged_in = False
            st.rerun()

def admin_dashboard_page():
    from core.database import Household, Transaction, User
    data_agent = st.session_state.architect_agent.data_agent
    
    st.header("Admin Dashboard")
    
    # User Management
    st.subheader("User Management")
    users = data_agent.db.query(User).all()
    user_data = [(u.id, u.username, u.role, u.household_id) for u in users]
    st.table(user_data)

    # Delete User Section
    st.subheader("Delete a User")
    usernames = [u.username for u in users if u.role != "admin"]
    if usernames:
        user_to_delete = st.selectbox("Select user to delete", usernames)
        if st.button("Delete User"):
            user_obj = data_agent.db.query(User).filter_by(username=user_to_delete).first()
            if user_obj:
                # Delete all transactions for this user
                data_agent.db.query(Transaction).filter_by(user_id=user_obj.id).delete()
                data_agent.db.delete(user_obj)
                data_agent.db.commit()
                st.success(f"User '{user_to_delete}' and their transactions deleted.")
                st.rerun()
    else:
        st.info("No non-admin users to delete.")

    # Change User Role
    st.subheader("Change User Role")
    users_to_change = [u.username for u in users if u.role != "admin"]
    if users_to_change:
        user_to_change = st.selectbox("Select user to change role", users_to_change)
        new_role = st.selectbox("New Role", ["user", "admin"])
        if st.button("Change Role"):
            user_obj = data_agent.db.query(User).filter_by(username=user_to_change).first()
            if user_obj:
                user_obj.role = new_role
                data_agent.db.commit()
                st.success(f"Role for '{user_to_change}' changed to '{new_role}'.")
                st.rerun()
    else:
        st.info("No non-admin users to modify.")

    # Household Management
    st.subheader("Household Management")
    households = data_agent.db.query(Household).all()
    household_data = [(h.id, h.name) for h in households]
    st.table(household_data)

    # Delete Household
    st.subheader("Delete a Household")
    household_names = [h.name for h in households]
    if household_names:
        household_to_delete = st.selectbox("Select household to delete", household_names)
        if st.button("Delete Household"):
            household_obj = data_agent.db.query(Household).filter_by(name=household_to_delete).first()
            if household_obj:
                # Delete all users and transactions in this household
                users_in_household = data_agent.db.query(User).filter_by(household_id=household_obj.id).all()
                for user in users_in_household:
                    data_agent.db.query(Transaction).filter_by(user_id=user.id).delete()
                    data_agent.db.delete(user)
                data_agent.db.delete(household_obj)
                data_agent.db.commit()
                st.success(f"Household '{household_to_delete}' and all related users/transactions deleted.")
                st.rerun()
    else:
        st.info("No households to delete.")

    # Audit and Transparency
    st.subheader("Transparency Dashboard / Logs")
    
    # Show last 50 transactions as a simple log
    logs = data_agent.db.query(Transaction).order_by(Transaction.date.desc(), Transaction.id.desc()).limit(50).all()
    if logs:
        log_data = [(t.id, t.user_id, t.t_type, t.category, t.amount, t.date, t.note) for t in logs]
        st.write("Recent User Actions & Data Changes:")
        st.table(log_data)
    else:
        st.info("No transaction logs available.")
    
    # User events log
    st.write("User Login/Logout/Deletion Events:")
    event_log = st.session_state.user_event_log[-50:] if 'user_event_log' in st.session_state else []
    if event_log:
        st.table([(e['event'], e['user'], e['timestamp']) for e in event_log])
    else:
        st.info("No user events logged.")

    # Search functionality
    st.subheader("Search Users & Households")
    search_user = st.text_input("Search user by username")
    search_household = st.text_input("Search household by name")
    
    filtered_users = [u for u in users if search_user.lower() in u.username.lower()] if search_user else users
    filtered_households = [h for h in households if search_household.lower() in h.name.lower()] if search_household else households
    
    st.write("Filtered Users:")
    st.table([(u.id, u.username, u.role, u.household_id) for u in filtered_users])
    
    st.write("Filtered Households:")
    st.table([(h.id, h.name) for h in filtered_households])

    # Data Export
    st.subheader("Export All Data (CSV)")
    if st.button("Export Users & Transactions"):
        users_df = pd.DataFrame([{"id": u.id, "username": u.username, "role": u.role, "household_id": u.household_id} for u in users])
        transactions_df = pd.DataFrame([{"id": t.id, "user_id": t.user_id, "type": t.t_type, "category": t.category, "amount": t.amount, "date": t.date, "note": t.note} for t in data_agent.db.query(Transaction).all()])
        users_df.to_csv("users_export.csv", index=False)
        transactions_df.to_csv("transactions_export.csv", index=False)
        st.success("Exported users_export.csv and transactions_export.csv to project folder.")

    # System Management
    st.subheader("System Management")
    if st.button("Delete ALL Data (Irreversible)"):
        # Delete all transactions, users, households
        data_agent.db.query(Transaction).delete()
        data_agent.db.query(User).delete()
        data_agent.db.query(Household).delete()
        data_agent.db.commit()
        st.success("All data deleted. App is now reset.")
        st.rerun()

def register_page():
    st.header("🔐 Register New User")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["user", "admin"])
            create_household = st.checkbox("Create New Household")
            household_name = st.text_input("Household Name")
        with col2:
            # Get existing households
            data_agent = st.session_state.architect_agent.data_agent
            try:
                from core.database import Household
                households = data_agent.db.query(Household).all()
                household_options = {f"{h.id}: {h.name}": h.id for h in households}
                household_options["None"] = 0
                default_index = list(household_options.keys()).index("None") if create_household else 0
                selected_household = st.selectbox(
                    "Select Household",
                    list(household_options.keys()),
                    index=default_index
                )
                household_id = household_options[selected_household]
            except Exception as e:
                st.warning("Could not load households")
                household_id = 0
        submitted = st.form_submit_button("Register")
        
        if submitted:
            import re
            username = username.strip()
            password = password.strip()
            household_name = household_name.strip()
            # Username: alphanumeric, 3-20 chars
            if not re.match(r'^[A-Za-z0-9_]{3,20}$', username):
                st.error("Username must be 3-20 characters, letters/numbers/underscores only.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif create_household and (not household_name or len(household_name) > 30):
                st.error("Household name must be 1-30 characters.")
            else:
                try:
                    from sqlalchemy.exc import IntegrityError
                    from core.database import User, Household
                    from core.security import hash_password
                    data_agent = st.session_state.architect_agent.data_agent
                    household = None
                    # Handle household creation/selection
                    if create_household and household_name:
                        household = Household(name=household_name)
                        data_agent.db.add(household)
                        data_agent.db.flush()  # Get ID without commit
                    elif not create_household and household_id > 0:
                        household = data_agent.db.query(Household).get(household_id)
                    
                    # Create user with role
                    user = User(
                        username=username,
                        password_hash=hash_password(password),
                        household=household,
                        role=role
                    )
                    data_agent.db.add(user)
                    data_agent.db.commit()
                    
                    st.success(f"✅ Registration successful as {role}! Please login.")
                    st.balloons()
                    # Log registration event
                    if 'user_event_log' not in st.session_state:
                        st.session_state.user_event_log = []
                    st.session_state.user_event_log.append({
                        "event": "register",
                        "user": username,
                        "timestamp": datetime.now().isoformat()
                    })

                except IntegrityError:
                    data_agent.db.rollback()
                    st.error("❌ Username already exists.")
                except Exception as e:
                    data_agent.db.rollback()
                    st.error(f"❌ Registration failed: {str(e)}")

def login_page():
    st.header("🔐 User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", width='stretch')
        
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
                        
                        # Log login event
                        if 'user_event_log' not in st.session_state:
                            st.session_state.user_event_log = []
                        st.session_state.user_event_log.append({
                            "event": "login",
                            "user": username,
                            "timestamp": datetime.now().isoformat()
                        })
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
            st.plotly_chart(fig_bar, config={"responsive": True})
        
        with col2:
            # Expenses by category pie chart
            expenses_df = df[df["type"] == "expense"]
            if not expenses_df.empty:
                category_totals = expenses_df.groupby("category")["amount"].sum()
                fig_pie = px.pie(values=category_totals.values, names=category_totals.index,
                               title="Expenses by Category")
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, config={"responsive": True})
        
        # Recent transactions
        st.subheader("Recent Transactions")
        recent_df = df.tail(10).sort_values("date", ascending=False)
        st.dataframe(recent_df, width='stretch')
        
        # Add export section
        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export to CSV",
                csv,
                "my_transactions.csv",
                "text/csv",
                key='download-csv'
            )
        
        with col2:
            # JSON export
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Export to JSON",
                json_str,
                "my_transactions.json",
                "application/json",
                key='download-json'
            )
    else:
        st.info("No transactions found. Start by adding some transactions!")
        if st.button("Add Your First Transaction"):
            st.session_state.page = "Add Transaction"
            st.rerun()

def add_transaction_page():
    if not check_auth():
        return

    st.header("➕ Add Transaction")

    # Ensure classifier is available
    if 'classifier' not in st.session_state:
        try:
            from models.category_classifier import CategoryClassifier
            st.session_state.classifier = CategoryClassifier()
        except Exception as e:
            st.warning(f"Classifier not loaded: {e}")

    # Ensure anomaly detector is available
    if 'anomaly_detector' not in st.session_state:
        try:
            from models.anomaly_detector import AnomalyDetector
            st.session_state.anomaly_detector = AnomalyDetector()
        except Exception as e:
            st.warning(f"Anomaly detector not loaded: {e}")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            t_type = st.selectbox("Transaction Type", ["income", "expense"])

            # Note input
            note_input = st.text_area("Note (optional)", placeholder="Additional details...")

            # Auto-suggest category
            predicted_category = None
            if note_input.strip() and 'classifier' in st.session_state:
                try:
                    predicted_category = st.session_state.classifier.predict(note_input)
                except Exception as e:
                    st.warning(f"Auto-suggestion failed: {e}")

            category_input = st.text_input(
                "Category",
                value=predicted_category if predicted_category else "",
                placeholder="e.g., food, bills, salary"
            )

            amount = st.number_input("Amount", min_value=0.01, format="%.2f")

        with col2:
            date_input = st.date_input("Date", value=date.today())

        submitted = st.form_submit_button("Add Transaction", width='stretch')

    if submitted:
        # Use predicted category if user left it empty
        final_category = category_input.strip() if category_input.strip() else predicted_category
        if not final_category:
            st.error("❌ Category cannot be empty. Please enter or select a category.")
            return

        if amount > 0:
            try:
                data_agent = st.session_state.architect_agent.data_agent
                from core.schemas import TransactionIn
                from core.database import Transaction

                transaction_data = TransactionIn(
                    t_type=t_type,
                    category=final_category,
                    amount=float(amount),
                    date=date_input,
                    note=note_input if note_input.strip() else None
                )

                # Save transaction
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
                st.info(f"Added: {t_type.title()} - {final_category} - ${amount:.2f} on {date_input}")

                # --- Anomaly Detection ---
                if t_type == "expense" and 'anomaly_detector' in st.session_state:
                    df = data_agent.get_transactions_df()
                    anomalies = st.session_state.anomaly_detector.detect(df)
                    if not anomalies.empty and tx.id in anomalies.index:
                        st.warning("⚠️ This expense looks unusual compared to your past expenses!")

            except Exception as e:
                st.error(f"❌ Failed to add transaction: {str(e)}")
                try:
                    data_agent.db.rollback()
                except Exception:
                    pass
        else:
            st.error("Please enter a valid amount.")

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

        # --- Mini Metrics ---
        if not filtered_df.empty:
            total_amount = filtered_df["amount"].sum()
            avg_amount = filtered_df["amount"].mean()
            top_categories = (filtered_df[filtered_df["type"]=="expense"]
                              .groupby("category")["amount"]
                              .sum()
                              .sort_values(ascending=False)
                              .head(3))
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Amount", f"${total_amount:,.2f}")
            with col2:
                st.metric("Average Amount", f"${avg_amount:,.2f}")
            with col3:
                st.metric("Top Category", f"{top_categories.index[0]} (${top_categories.iloc[0]:,.2f})" if not top_categories.empty else "-")
            with col4:
                st.metric("2nd Top Category", f"{top_categories.index[1]} (${top_categories.iloc[1]:,.2f})" if len(top_categories) > 1 else "-")

        # Display transactions
        display_df = filtered_df.copy()
        display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, width='stretch')
        
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
                time_range_param = _get_time_range_param(time_range)
                
                with st.spinner("Generating charts..."):
                    if chart_type in ["Expenses by Category", "All Basic Charts"]:
                        st.subheader("Expenses by Category")
                        fig, caption = chart_agent.pie_expenses_by_category(save=False, show=False, time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_caption(caption)
                    
                    if chart_type in ["Income vs Expenses", "All Basic Charts"]:
                        st.subheader("Income vs Expenses")
                        fig, caption = chart_agent.bar_income_vs_expense(time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_caption(caption)
                    
                    if chart_type in ["Savings Over Time", "All Basic Charts"]:
                        st.subheader("Savings Over Time")
                        fig, caption = chart_agent.line_savings_over_time(time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_caption(caption)
        
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
                if isinstance(result, tuple):
                    st.plotly_chart(result[0], use_container_width=True)
                    if len(result) > 1 and isinstance(result[1], dict):
                        _display_caption(result[1])
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

def insights_page():
    import streamlit as st
    from agents.insight_generator import InsightGeneratorAgent

    st.title("💡 Financial Insights Dashboard")

    data_agent = st.session_state.architect_agent.data_agent
    agent = InsightGeneratorAgent(data_agent=data_agent)
    insights = agent.generate_all()
    rai = insights.get("responsible_ai", [])

    def get_rai_message(keywords):
        for msg in rai:
            for keyword in keywords:
                if keyword.lower() in msg.lower():
                    return msg
        return None

    # --- Summary Section ---
    st.subheader("📊 Summary Statistics")
    stats = insights.get("summary_stats", {})
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Income", f"${stats.get('income', 0):,.2f}")
        col2.metric("Expense", f"${stats.get('expense', 0):,.2f}")
        col3.metric("Savings", f"${stats.get('savings', 0):,.2f}")
        col4.metric("Savings Rate", f"{stats.get('savings_rate', 0):.1f}%")
        rai_msg = get_rai_message(["category"])
        if rai_msg:
            st.caption(f"Responsible AI: {rai_msg}")

        # Show top categories
        df = data_agent.get_transactions_df()
        if df is not None and not df.empty and "category" in df.columns:
            st.markdown("**Top Expense Categories:**")
            top_cats = df[df["type"]=="expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(3)
            for cat, amt in top_cats.items():
                st.write(f"- {cat}: ${amt:,.2f}")

    else:
        st.info("No summary statistics available.")

    # --- Trend Section ---
    st.subheader("📈 Trends & Forecasts")
    trend = insights.get("trend", "")
    forecast = insights.get("forecast", {})
    if trend:
        st.write(trend)
        rai_msg = get_rai_message(["transparency"])
        if rai_msg:
            st.caption(f"Responsible AI: {rai_msg}")
    if forecast:
        st.write(f"**Last Month's Spending:** ${forecast.get('last_month', 0):,.2f}")
        st.write(f"**Forecast Next Month:** ${forecast.get('forecast_next_month', 0):,.2f}")

    # --- Goal Tracking ---
    st.subheader("🎯 Goal Tracking")
    goal = insights.get("goal", "")
    if goal:
        st.write(goal)
        # Progress bar for savings vs. goal
        savings = stats.get("savings", 0)
        savings_goal = agent.savings_target if hasattr(agent, "savings_target") else 5000
        progress = min(max(savings / savings_goal, 0), 1) if savings_goal > 0 else 0
        st.progress(progress, text=f"Savings Progress: ${savings:,.2f} / ${savings_goal:,.2f}")
    else:
        st.info("No savings goal data available.")

    # --- Alerts ---
    st.subheader("🚨 Alerts")
    alerts = insights.get("alerts", [])
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No alerts at this time.")

    # --- Anomalies ---
    st.subheader("🔎 Anomaly Detection")
    anomaly_msgs = insights.get("anomaly_messages", [])
    anomalies = insights.get("anomalies", None)
    if anomaly_msgs:
        for msg in anomaly_msgs:
            st.error(msg)
        rai_msg = get_rai_message(["sensitive"])
        if rai_msg:
            st.caption(f"Responsible AI: {rai_msg}")
        # Show anomaly table if available
        if anomalies is not None and not anomalies.empty:
            st.markdown("**Anomalous Transactions:**")
            st.dataframe(anomalies)
    else:
        st.success("No anomalies detected.")

    # --- Budget Recommendations ---
    st.subheader("💡 Budget Recommendations")
    recs = insights.get("budget_recommendations", [])
    if recs:
        for rec in recs:
            st.info(rec)
    else:
        st.info("No budget recommendations available.")

    # --- Data Coverage ---
    st.subheader("📅 Data Coverage")
    df = data_agent.get_transactions_df()
    if df is not None and not df.empty and "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()
        st.write(f"Data from **{min_date}** to **{max_date}**")
        st.write(f"Total records: {len(df)}")
        st.write(f"Categories used: {df['category'].nunique() if 'category' in df.columns else 0}")

    # --- Show any remaining RAI notes not already shown ---
    shown = [
        get_rai_message(["category"]),
        get_rai_message(["transparency"]),
        get_rai_message(["sensitive"])
    ]
    for msg in rai:
        if msg and msg not in shown:
            st.caption(f"Responsible AI: {msg}")

def logout():
    # Log the logout event
    if 'user_event_log' not in st.session_state:
        st.session_state.user_event_log = []
    st.session_state.user_event_log.append({
        "event": "logout",
        "user": st.session_state.current_user,
        "timestamp": datetime.now().isoformat()
    })
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

# Helper functions
def _display_caption(caption):
    """Display chart caption in a formatted box with dark text"""
    if isinstance(caption, dict) and 'summary' in caption:
        st.markdown(f"""
        <div class="caption-box">
            <h4>📋 Chart Explanation</h4>
            <p><strong>What's being compared:</strong> {caption.get('comparison', 'N/A')}</p>
            <p><strong>Chart type reasoning:</strong> {caption.get('reasoning', 'N/A')}</p>
            <p><strong>Key insight:</strong> {caption.get('insights', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    elif isinstance(caption, dict):
        # Handle cases where the structure might be different
        st.markdown(f"""
        <div class="caption-box">
            <h4>📋 Chart Explanation</h4>
            <p><strong>Summary:</strong> {caption.get('summary', 'No description available')}</p>
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

if __name__ == "__main__":
    main()