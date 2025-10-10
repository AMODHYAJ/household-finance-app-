import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from datetime import timedelta
from datetime import date, datetime
import os
import sys
from PIL import Image
import io
from typing import Dict, Any, List

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
.admin-card {
    background: linear-gradient(145deg, #ffffff, #f8f9fa);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #6c757d;
}
.danger-zone {
    border-left: 4px solid #dc3545 !important;
    background: linear-gradient(145deg, #fff5f5, #f8d7da) !important;
}
.success-zone {
    border-left: 4px solid #28a745 !important;
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
    
    st.header("👑 Admin Dashboard")
    
    # Modern tabs layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "👥 User Management", 
        "🏠 Household Management",
        "📋 Audit Logs",
        "🔍 Search",
        "⚙️ System Tools"
    ])
    
    with tab1:
        st.subheader("📈 System Overview")
        
        # Quick stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        users = data_agent.db.query(User).all()
        households = data_agent.db.query(Household).all()
        transactions = data_agent.db.query(Transaction).all()
        recent_logs = data_agent.db.query(Transaction).order_by(Transaction.date.desc()).limit(50).all()
        
        with col1:
            st.metric("Total Users", len(users), delta=f"{len([u for u in users if u.role == 'admin'])} admins")
        with col2:
            st.metric("Households", len(households))
        with col3:
            st.metric("Transactions", len(transactions))
        with col4:
            st.metric("Recent Activity", len(recent_logs))
        
        # System health indicators
        st.subheader("🩺 System Health")
        health_col1, health_col2, health_col3 = st.columns(3)
        
        with health_col1:
            st.info("✅ Database Connection Active")
            st.info("✅ Authentication System Ready")
        
        with health_col2:
            avg_transactions = len(transactions) / max(len(users), 1)
            st.metric("Avg Transactions/User", f"{avg_transactions:.1f}")
        
        with health_col3:
            recent_users = [u for u in users if u.id in [t.user_id for t in recent_logs]]
            st.metric("Active Users", len(set(recent_users)))
    
    with tab2:
        st.subheader("👥 User Management")
        
        # User table with better styling
        if users:
            user_df = pd.DataFrame([{
                "ID": u.id,
                "Username": u.username,
                "Role": u.role,
                "Household ID": u.household_id,
                "Actions": "⚙️"
            } for u in users])
            
            # Use aggrid for interactive table
            try:
                from st_aggrid import AgGrid, GridOptionsBuilder
                gb = GridOptionsBuilder.from_dataframe(user_df)
                gb.configure_pagination(paginationPageSize=10)
                gb.configure_side_bar()
                gb.configure_default_column(filterable=True, sortable=True, editable=False)
                grid_options = gb.build()
                
                grid_response = AgGrid(
                    user_df,
                    gridOptions=grid_options,
                    height=300,
                    theme='streamlit',
                    fit_columns_on_grid_load=True
                )
            except ImportError:
                st.dataframe(user_df, height=300)
        
        # User actions in expanders
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🗑️ Delete User", expanded=False):
                usernames = [u.username for u in users if u.role != "admin"]
                if usernames:
                    user_to_delete = st.selectbox("Select user to delete", usernames, key="delete_user")
                    if st.button("Delete User", type="primary", key="delete_btn"):
                        user_obj = data_agent.db.query(User).filter_by(username=user_to_delete).first()
                        if user_obj:
                            data_agent.db.query(Transaction).filter_by(user_id=user_obj.id).delete()
                            data_agent.db.delete(user_obj)
                            data_agent.db.commit()
                            st.success(f"✅ User '{user_to_delete}' deleted successfully!")
                            st.rerun()
                else:
                    st.info("No non-admin users to delete.")
        
        with col2:
            with st.expander("🔄 Change User Role", expanded=False):
                users_to_change = [u.username for u in users if u.role != "admin"]
                if users_to_change:
                    user_to_change = st.selectbox("Select user", users_to_change, key="role_user")
                    new_role = st.selectbox("New Role", ["user", "admin"], key="new_role")
                    if st.button("Change Role", type="primary", key="role_btn"):
                        user_obj = data_agent.db.query(User).filter_by(username=user_to_change).first()
                        if user_obj:
                            user_obj.role = new_role
                            data_agent.db.commit()
                            st.success(f"✅ Role changed to '{new_role}' for '{user_to_change}'!")
                            st.rerun()
                else:
                    st.info("No non-admin users to modify.")
    
    with tab3:
        st.subheader("🏠 Household Management")
        
        if households:
            household_df = pd.DataFrame([{
                "ID": h.id,
                "Name": h.name,
                "Member Count": len([u for u in users if u.household_id == h.id]),
                "Actions": "⚙️"
            } for h in households])
            
            st.dataframe(household_df, height=200)
        
        with st.expander("🗑️ Delete Household", expanded=False):
            household_names = [h.name for h in households]
            if household_names:
                household_to_delete = st.selectbox("Select household", household_names)
                if st.button("Delete Household", type="primary"):
                    household_obj = data_agent.db.query(Household).filter_by(name=household_to_delete).first()
                    if household_obj:
                        users_in_household = data_agent.db.query(User).filter_by(household_id=household_obj.id).all()
                        for user in users_in_household:
                            data_agent.db.query(Transaction).filter_by(user_id=user.id).delete()
                            data_agent.db.delete(user)
                        data_agent.db.delete(household_obj)
                        data_agent.db.commit()
                        st.success(f"✅ Household '{household_to_delete}' deleted successfully!")
                        st.rerun()
            else:
                st.info("No households to delete.")
    
    with tab4:
        st.subheader("📋 Audit & Transparency")
        
        # Recent transactions log
        st.write("**Recent User Actions (Last 50):**")
        if recent_logs:
            log_df = pd.DataFrame([{
                "ID": t.id,
                "User ID": t.user_id,
                "Type": t.t_type,
                "Category": t.category,
                "Amount": f"${t.amount:.2f}",
                "Date": t.date,
                "Note": t.note[:50] + "..." if t.note and len(t.note) > 50 else t.note
            } for t in recent_logs])
            st.dataframe(log_df, height=300)
        else:
            st.info("No transaction logs available.")
        
        # User events log
        st.write("**System Events:**")
        event_log = st.session_state.user_event_log[-20:] if 'user_event_log' in st.session_state else []
        if event_log:
            event_df = pd.DataFrame(event_log)
            st.dataframe(event_df, height=200)
        else:
            st.info("No system events logged.")
    
    with tab5:
        st.subheader("🔍 Advanced Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_user = st.text_input("🔎 Search users by username")
            if search_user:
                filtered_users = [u for u in users if search_user.lower() in u.username.lower()]
                if filtered_users:
                    st.write(f"**Found {len(filtered_users)} users:**")
                    for user in filtered_users:
                        st.write(f"- {user.username} (Role: {user.role}, Household: {user.household_id})")
                else:
                    st.info("No users found.")
        
        with col2:
            search_household = st.text_input("🔎 Search households by name")
            if search_household:
                filtered_households = [h for h in households if search_household.lower() in h.name.lower()]
                if filtered_households:
                    st.write(f"**Found {len(filtered_households)} households:**")
                    for household in filtered_households:
                        member_count = len([u for u in users if u.household_id == household.id])
                        st.write(f"- {household.name} (Members: {member_count})")
                else:
                    st.info("No households found.")
    
    with tab6:
        st.subheader("⚙️ System Management Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Export**")
            if st.button("📥 Export All Data to CSV", type="secondary"):
                users_df = pd.DataFrame([{"id": u.id, "username": u.username, "role": u.role, "household_id": u.household_id} for u in users])
                transactions_df = pd.DataFrame([{"id": t.id, "user_id": t.user_id, "type": t.t_type, "category": t.category, "amount": t.amount, "date": t.date, "note": t.note} for t in transactions])
                
                csv1 = users_df.to_csv(index=False)
                csv2 = transactions_df.to_csv(index=False)
                
                st.download_button("Download Users CSV", csv1, "users_export.csv", "text/csv")
                st.download_button("Download Transactions CSV", csv2, "transactions_export.csv", "text/csv")
        
        with col2:
            st.write("**Danger Zone**")
            with st.expander("🚨 Reset System", expanded=False):
                st.error("This will delete ALL data permanently!")
                confirmation = st.text_input("Type 'DELETE ALL' to confirm")
                if st.button("💀 Delete ALL Data", type="primary", disabled=confirmation != "DELETE ALL"):
                    if confirmation == "DELETE ALL":
                        data_agent.db.query(Transaction).delete()
                        data_agent.db.query(User).delete()
                        data_agent.db.query(Household).delete()
                        data_agent.db.commit()
                        st.success("✅ All data has been deleted. System reset complete.")
                        st.rerun()

    # Add some modern touches
    st.markdown("---")
    st.caption("🔒 Admin Dashboard • Last refreshed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
 
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

    # Ensure anomaly detector is available (main branch feature)
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

            # Auto-suggest category with confidence levels (vishmi branch feature)
            suggestions = []
            predicted_category = None
            if note_input.strip() and 'classifier' in st.session_state:
                try:
                    # Get top 3 suggestions with confidence levels
                    suggestions = st.session_state.classifier.predict_top(note_input, top_n=3)
                    predicted_category = suggestions[0][0] if suggestions else None
                except Exception as e:
                    # Fallback to main branch behavior if predict_top is not available
                    try:
                        predicted_category = st.session_state.classifier.predict(note_input)
                        suggestions = [(predicted_category, 1.0)]  # Default confidence
                    except Exception as e2:
                        st.warning(f"Auto-suggestion failed: {e2}")

            # Category input with suggestions display (merged feature)
            category_input = st.text_input(
                "Category",
                value=predicted_category if predicted_category else "",
                placeholder="e.g., food, bills, salary"
            )

            # Display suggestions with confidence levels if available (vishmi branch feature)
            if suggestions:
                st.markdown("💡 Suggested categories:")
                for cat, conf in suggestions:
                    st.markdown(f"- {cat} ({conf:.2%} confidence)")

            amount = st.number_input("Amount", min_value=0.01, format="%.2f")

        with col2:
            date_input = st.date_input("Date", value=date.today(), key="add_transaction_date_input")

        submitted = st.form_submit_button("Add Transaction", use_container_width=True)

    if submitted:
        # Use predicted category if user left it empty (main branch logic with vishmi enhancement)
        final_category = category_input.strip() 
        if not final_category and suggestions:
            final_category = suggestions[0][0]  # best suggestion from vishmi branch
        elif not final_category and predicted_category:
            final_category = predicted_category  # fallback to main branch behavior

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

                # Save transaction (main branch functionality)
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

                # --- Anomaly Detection (main branch feature) ---
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
    # Ensure chart_agent is properly initialized
    if not hasattr(st.session_state.architect_agent, 'chart_agent') or st.session_state.architect_agent.chart_agent is None:
        st.session_state.architect_agent.chart_agent = ChartCreatorAgent(data_agent)
    
    chart_agent = st.session_state.architect_agent.chart_agent
    df = data_agent.get_transactions_df()
    
    if df is not None and not df.empty:
        # Tab layout for different chart types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Basic Charts", 
            "🎯 Interactive Dashboard", 
            "🔍 Comparative Analysis",
            "🔮 Predictive Charts",
            "🎨 Visual Query Builder"
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
                        _display_enhanced_caption(caption, "Expenses by Category Analysis")  # FIXED: Added original_query
                    
                    if chart_type in ["Income vs Expenses", "All Basic Charts"]:
                        st.subheader("Income vs Expenses")
                        fig, caption = chart_agent.bar_income_vs_expense(time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_enhanced_caption(caption, "Income vs Expenses Comparison")  # FIXED: Added original_query
                    
                    if chart_type in ["Savings Over Time", "All Basic Charts"]:
                        st.subheader("Savings Over Time")
                        fig, caption = chart_agent.line_savings_over_time(time_range=time_range_param)
                        st.plotly_chart(fig, use_container_width=True)
                        _display_enhanced_caption(caption, "Savings Trend Analysis")  # FIXED: Added original_query
        
        with tab2:
            st.subheader("🎯 Interactive Dashboard")
            st.markdown("**Customize your dashboard by applying filters below:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_categories = df['category'].unique() if 'category' in df.columns else []
                categories = st.multiselect(
                    "Filter by Category", 
                    options=available_categories,
                    placeholder="Select categories...",
                    key="dashboard_category"
                )
            
            with col2:
                transaction_types = st.multiselect(
                    "Filter by Type", 
                    options=['income', 'expense'],
                    default=['income', 'expense'],
                    placeholder="Select transaction types...",
                    key="dashboard_type"
                )
            
            with col3:
                if 'date' in df.columns:
                    min_date = pd.to_datetime(df['date']).min().date()
                    max_date = pd.to_datetime(df['date']).max().date()
                    date_range = st.date_input(
                        "Filter by Date Range", 
                        value=[min_date, max_date],
                        min_value=min_date,
                        max_value=max_date,
                        key="dashboard_date"
                    )
                else:
                    date_range = st.date_input("Filter by Date Range", value=[], key="dashboard_date_fallback")
                    st.warning("No date data available")
            
            if st.button("🚀 Generate Dashboard", type="primary", key="generate_dashboard"):
                with st.spinner("Creating interactive dashboard..."):
                    try:
                        filters = {
                            'category': categories if categories else None,
                            'type': transaction_types if transaction_types else None,
                            'date_range': date_range if len(date_range) == 2 else None
                        }
                        result = chart_agent.create_interactive_dashboard(filters)
                        
                        # Handle both return types: figure only OR (figure, caption) tuple
                        if isinstance(result, tuple) and len(result) == 2:
                            fig, caption = result
                            st.plotly_chart(fig, use_container_width=True)
                            _display_enhanced_caption(caption, "Interactive Dashboard")  # Display the caption
                        else:
                            # If it returns just the figure
                            fig = result
                            st.plotly_chart(fig, use_container_width=True)
                            
                        st.success("✅ Dashboard generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to create dashboard: {str(e)}")
        
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
            
            if st.button("Compare Periods", key="compare_btn"):
                with st.spinner("Generating comparison..."):
                    fig, caption = chart_agent.comparative_charts(
                        period1.lower().replace(" ", "_"), 
                        period2.lower().replace(" ", "_")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    _display_enhanced_caption(caption, f"Compare {period1} vs {period2}")  # FIXED: Added original_query
        
        with tab4:
            st.subheader("🔮 Predictive Financial Charts")
            
            periods = st.slider("Months to Predict", 1, 12, 6, key="prediction_slider")
            
            if st.button("Generate Predictions", key="predict_btn"):
                with st.spinner("Generating predictions..."):
                    fig, caption = chart_agent.predictive_charts(periods)
                    st.plotly_chart(fig, use_container_width=True)
                    _display_enhanced_caption(caption, f"Predict next {periods} months")  # FIXED: Added original_query
        
        with tab5:
            _render_enhanced_visual_query_builder(chart_agent, df)
                
    else:
        st.info("No data available for charts. Add some transactions first!")
        if st.button("Add Your First Transaction"):
            st.session_state.page = "Add Transaction"
            st.rerun()

def _render_enhanced_visual_query_builder(chart_agent, df):
    """Render the enhanced Visual Query Builder tab content"""
    st.markdown("## 🎨 Visual Query Builder")
    st.markdown("Build custom charts using intuitive visual components - **Charts generate automatically as you make selections**")
    
    # Step 1: Chart Configuration
    st.markdown("### 1. Chart Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 What to Analyze")
        chart_focus = st.selectbox(
            "Focus on:",
            ["Expenses", "Income", "Savings", "Comparison"],
            key="vb_focus",
            help="Select what type of data to analyze"
        )
    
    with col2:
        st.subheader("⏰ Time Period")
        time_period = st.selectbox(
            "Time range:",
            ["All Time", "This Month", "Last Month", "Last 3 Months", "This Year"],
            key="vb_time",
            help="Select the time period for analysis"
        )
    
    with col3:
        st.subheader("🔍 Detail Level")
        detail_level = st.selectbox(
            "Show data as:",
            ["Summary", "By Category", "Trend Over Time", "Detailed Breakdown"],
            key="vb_detail",
            help="Choose how detailed you want the visualization to be"
        )
    
    # Step 2: Advanced Filters
    st.markdown("### 2. Advanced Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_amount = st.number_input(
            "Minimum Amount", 
            min_value=0.0, 
            value=0.0, 
            key="vb_min_amount",
            help="Filter out transactions below this amount"
        )
    
    with col2:
        # Get available categories dynamically - MULTI-SELECT with "All Categories" option
        available_cats = chart_agent.get_visual_builder_categories()
        # Add "All Categories" as the first option
        all_categories_option = ["All Categories"]
        category_options = all_categories_option + available_cats
        
        specific_categories = st.multiselect(
            "Select Categories", 
            options=category_options,
            default=["All Categories"],  # Default to "All Categories"
            key="vb_categories",
            help="Select one or more categories to focus on. Choose 'All Categories' to include everything."
        )
        
        # Handle "All Categories" selection logic
        if "All Categories" in specific_categories:
            # If "All Categories" is selected along with other categories, show warning
            if len(specific_categories) > 1:
                st.warning("⚠️ 'All Categories' is selected. Other category selections will be ignored.")
                # Keep only "All Categories"
                specific_categories = ["All Categories"]
    
    # Build components dictionary
    components = {
        'chart_focus': chart_focus,
        'time_period': time_period,
        'detail_level': detail_level,
        'min_amount': min_amount,
        'specific_categories': specific_categories
    }
    
    # Step 3: Auto-generate chart based on current selections
    st.markdown("### 3. Live Chart Preview")
    
    # Preview section with dynamic chart type description
    with st.expander("👁️ Query Preview", expanded=True):
        query_parts = []
        if chart_focus: query_parts.append(f"**{chart_focus}**")
        if time_period != "All Time": query_parts.append(f"for **{time_period}**")
        if detail_level != "Summary": query_parts.append(f"as **{detail_level}**")
        
        # Handle category display
        if specific_categories:
            if "All Categories" in specific_categories:
                query_parts.append("across **All Categories**")
            elif len(specific_categories) == 1:
                query_parts.append(f"in **{specific_categories[0]}**")
            else:
                query_parts.append(f"in **{len(specific_categories)} selected categories**")
        
        if min_amount > 0: query_parts.append(f"over **${min_amount}**")
        
        generated_query = " • ".join(query_parts)
        
        st.info(f"**Your Chart Will Show:** {generated_query}")
        
        # Dynamic chart type description based on focus and detail level
        selected_cat_count = 0 if "All Categories" in specific_categories else len(specific_categories)
        chart_description = _get_chart_description(chart_focus, detail_level, selected_cat_count)
        st.success(f"🎯 **Chart Type:** {chart_description}")
    
    # AUTO-GENERATE CHART based on current selections
    with st.spinner("🔄 Generating chart based on your selections..."):
        try:
            result = chart_agent.visual_components_to_chart(components)
            
            # Handle different return types from chart agent
            if isinstance(result, tuple) and len(result) == 2:
                fig, caption = result
                st.plotly_chart(fig, use_container_width=True)
                _display_visual_builder_caption(caption, components)
                st.success("✅ Chart generated successfully!")
                
                # Show data summary
                with st.expander("📊 Data Summary", expanded=False):
                    filtered_df = chart_agent.get_filtered_data_summary(components)
                    if not filtered_df.empty:
                        total_amount = filtered_df['amount'].sum()
                        transaction_count = len(filtered_df)
                        st.write(f"**Transactions analyzed:** {transaction_count}")
                        st.write(f"**Total amount:** ${total_amount:,.2f}")
                        
                        # Show breakdown by type if applicable
                        if 'type' in filtered_df.columns:
                            type_breakdown = filtered_df.groupby('type')['amount'].sum()
                            for t_type, amount in type_breakdown.items():
                                st.write(f"**{t_type.title()}:** ${amount:,.2f}")
                        
                        # Show categories breakdown
                        st.write("**Categories Breakdown:**")
                        category_breakdown = filtered_df.groupby('category')['amount'].sum().sort_values(ascending=False)
                        for category, amount in category_breakdown.head(10).items():  # Show top 10
                            st.write(f"- {category}: ${amount:,.2f}")
                            
                        if len(category_breakdown) > 10:
                            st.write(f"- ... and {len(category_breakdown) - 10} more categories")
                    else:
                        st.info("No data matches your current filters.")
            else:
                # Handle error responses
                st.error("❌ Could not generate chart. Please check your data and filters.")
                
        except Exception as e:
            st.error(f"❌ Failed to generate chart: {str(e)}")
            st.info("💡 Try adjusting your filters or adding more transaction data.")

    # Reset button
    if st.button("🔄 Reset All Selections", type="secondary", use_container_width=True, key="vb_reset"):
        # Clear relevant session state keys
        keys_to_clear = ['vb_focus', 'vb_time', 'vb_detail', 'vb_min_amount', 'vb_categories']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def _get_chart_description(focus, detail_level, selected_categories_count):
    """Get dynamic chart description based on focus and detail level"""
    
    chart_types = {
        "Expenses": {
            "Summary": "Expense Overview with Key Metrics",
            "By Category": "Expense Distribution by Category",
            "Trend Over Time": "Monthly Expense Trends", 
            "Detailed Breakdown": "Comprehensive Expense Analysis with Multiple Charts"
        },
        "Income": {
            "Summary": "Income Overview with Key Metrics", 
            "By Category": "Income Sources Breakdown",
            "Trend Over Time": "Monthly Income Trends",
            "Detailed Breakdown": "Comprehensive Income Analysis with Multiple Charts"
        },
        "Savings": {
            "Summary": "Savings Overview with Net Position",
            "By Category": "Savings by Period", 
            "Trend Over Time": "Savings Accumulation Over Time",
            "Detailed Breakdown": "Comprehensive Savings Analysis with Trends"
        },
        "Comparison": {
            "Summary": "Income vs Expenses Comparison",
            "By Category": "Category-wise Income vs Expense Comparison",
            "Trend Over Time": "Income & Expense Trends Over Time",
            "Detailed Breakdown": "Comprehensive Financial Comparison Dashboard"
        }
    }
    
    base_description = chart_types.get(focus, {}).get(detail_level, "Custom Chart")
    
    # Add category info if specific categories are selected
    if selected_categories_count > 0:
        if selected_categories_count == 1:
            base_description += " (Single Category Focus)"
        else:
            base_description += f" ({selected_categories_count} Categories)"
    
    return base_description

def _display_visual_builder_caption(caption, components):
    """Display caption for visual query builder charts"""
    if not isinstance(caption, dict):
        st.info("📊 Chart generated successfully!")
        return
    
    st.markdown("---")
    st.subheader("📋 Chart Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
            <h4 style="color: #1f77b4; margin-bottom: 1rem;">🎨 Builder Configuration</h4>
            <p><strong>Focus:</strong> {components.get('chart_focus', 'N/A')}</p>
            <p><strong>Time Period:</strong> {components.get('time_period', 'N/A')}</p>
            <p><strong>Detail Level:</strong> {components.get('detail_level', 'N/A')}</p>
            <p><strong>Categories:</strong> {', '.join(components.get('specific_categories', ['All Categories']))}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        chart_type = caption.get('chart_type_used', 'N/A')
        data_points = caption.get('data_points', 'N/A')
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-bottom: 1rem;">📊 Chart Information</h4>
            <p><strong>Chart Type:</strong> {chart_type.title() if chart_type != 'N/A' else 'Auto-selected'}</p>
            <p><strong>Data Focus:</strong> {caption.get('data_focus', 'N/A').title()}</p>
            <p><strong>Query Type:</strong> Visual Builder</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Insights
    if caption.get('key_insights'):
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin-top: 1rem;">
            <h4 style="color: #856404; margin-bottom: 1rem;">💡 Key Insights</h4>
            <p style="color: #856404;">{caption.get('key_insights', 'No specific insights available.')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced insights if available
    if caption.get('enhanced_insights'):
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745; margin-top: 1rem;">
            <h4 style="color: #155724; margin-bottom: 1rem;">🔍 Enhanced Insights</h4>
            <p style="color: #155724;">{caption.get('enhanced_insights')}</p>
        </div>
        """, unsafe_allow_html=True)

def _display_enhanced_caption(caption, original_query):  # ← ADD this parameter
    """Display enhanced chart caption with better formatting"""
    if not isinstance(caption, dict):
        return
    
    st.markdown("---")
    st.subheader("📋 Chart Explanation")
    
    # Create a more comprehensive caption display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
            <h4 style="color: #1f77b4; margin-bottom: 1rem;">🔍 Query Analysis</h4>
            <p><strong>Your Query:</strong> "{original_query}"</p>  # ← USE the parameter here
            <p><strong>Understood Intent:</strong> {str(caption.get('understood_intent', 'N/A')).replace('_', ' ').title() if caption.get('understood_intent') is not None else 'N/A'}</p>
            <p><strong>Confidence Level:</strong> {caption.get('confidence_level', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-bottom: 1rem;">📊 Chart Details</h4>
            <p><strong>Chart Type:</strong> {str(caption.get('chart_type_used', 'N/A')).title() if caption.get('chart_type_used') is not None else 'N/A'}</p>
            <p><strong>Time Period:</strong> {str(caption.get('time_period_analyzed', 'N/A')).replace('_', ' ').title() if caption.get('time_period_analyzed') is not None else 'N/A'}</p>
            <p><strong>Data Focus:</strong> {str(caption.get('data_focus', 'N/A')).title() if caption.get('data_focus') is not None else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Insights section
    if caption.get('key_insights'):
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin-top: 1rem;">
            <h4 style="color: #856404; margin-bottom: 1rem;">💡 Key Insights</h4>
            <p style="color: #856404;">{caption.get('key_insights', 'No specific insights available.')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis Reasoning
    if caption.get('analysis_reasoning'):
        st.markdown("**🔎 Analysis Reasoning:**")
        reasoning = caption.get('analysis_reasoning', [])
        if isinstance(reasoning, list):
            for reason in reasoning:
                st.write(f"- {reason}")
        else:
            st.write(f"- {reasoning}")
    
    # Full Summary
    if caption.get('summary'):
        st.markdown(f"""
        <div style="background-color: #e7f3ff; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <p><strong>📝 Summary:</strong> {caption.get('summary')}</p>
        </div>
        """, unsafe_allow_html=True)

def _display_caption(caption):
    """Display chart caption in a formatted box with dark text"""
    if isinstance(caption, dict):
        # Handle the new enhanced caption format
        if 'key_insights' in caption:
            _display_enhanced_caption(caption)
        else:
            # Fallback to original format for backward compatibility
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
                <h4 style="color: #1f77b4; margin-bottom: 1rem;">📋 Chart Explanation</h4>
                <p><strong>What's being compared:</strong> {caption.get('comparison', 'N/A')}</p>
                <p><strong>Chart type reasoning:</strong> {caption.get('reasoning', 'N/A')}</p>
                <p><strong>Key insight:</strong> {caption.get('insights', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📋 Chart generated successfully!")

def _create_empty_chart(message):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor="center", yanchor="middle",
        showarrow=False, 
        font=dict(size=16, color="#7f8c8d")
    )
    fig.update_layout(
        title=message,
        template="plotly_white",
        height=400
    )
    return fig

def _apply_visual_filters(df: pd.DataFrame, components: Dict) -> pd.DataFrame:
    """Apply filters based on visual query components with multi-category support"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    filtered_df = df.copy()
    
    chart_focus = components.get('chart_focus', 'Expenses')
    if chart_focus == 'Expenses':
        filtered_df = filtered_df[filtered_df['type'] == 'expense']
    elif chart_focus == 'Income':
        filtered_df = filtered_df[filtered_df['type'] == 'income']
    
    time_period = components.get('time_period', 'All Time')
    if time_period != "All Time":
        filtered_df = _apply_time_filter_to_df(filtered_df, time_period)
    
    specific_categories = components.get('specific_categories', [])
    if specific_categories and "All Categories" not in specific_categories:
        category_mask = False
        for category in specific_categories:
            category_lower = category.strip().lower()
            category_mask |= (filtered_df['category'].astype(str).str.strip().str.lower() == category_lower)
        filtered_df = filtered_df[category_mask]
    
    min_amount = components.get('min_amount', 0)
    if min_amount > 0:
        filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
    
    return filtered_df

def _apply_time_filter_to_df(df: pd.DataFrame, time_period: str) -> pd.DataFrame:
    """Apply time filter to dataframe for visual query builder"""
    df_copy = df.copy()
    
    if 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['date'])
    
    if df_copy.empty:
        return df_copy
    
    now = datetime.now()
    if time_period == 'This Month':
        start_date = now.replace(day=1)
        return df_copy[df_copy['date'] >= start_date]
    elif time_period == 'Last Month':
        first_day_this_month = now.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return df_copy[
            (df_copy['date'] >= first_day_last_month) & 
            (df_copy['date'] <= last_day_last_month)
        ]
    elif time_period == 'Last 3 Months':
        cutoff = now - timedelta(days=90)
        return df_copy[df_copy['date'] >= cutoff]
    elif time_period == 'This Year':
        start_date = now.replace(month=1, day=1)
        return df_copy[df_copy['date'] >= start_date]
    else:
        return df_copy

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
        if savings_goal > 0:
            progress = float(savings) / float(savings_goal)
            progress = max(0.0, min(progress, 1.0))  # Clamp between 0.0 and 1.0
        else:
            progress = 0.0
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
    if isinstance(caption, dict):
        # Handle the new enhanced caption format
        if 'key_insights' in caption:
            _display_enhanced_caption(caption, caption.get('original_query', 'User Query'))
        else:
            # Fallback to original format for backward compatibility
            st.markdown(f"""
            <div class="caption-box">
                <h4>📋 Chart Explanation</h4>
                <p><strong>What's being compared:</strong> {caption.get('comparison', 'N/A')}</p>
                <p><strong>Chart type reasoning:</strong> {caption.get('reasoning', 'N/A')}</p>
                <p><strong>Key insight:</strong> {caption.get('insights', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📋 Chart generated successfully!")

def _display_enhanced_caption(caption, original_query):
    """Display enhanced chart caption with better formatting"""
    if not isinstance(caption, dict):
        return
    
    st.markdown("---")
    st.subheader("📋 Chart Explanation")
    
    # Create a more comprehensive caption display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
            <h4 style="color: #1f77b4; margin-bottom: 1rem;">🔍 Query Analysis</h4>
            <p><strong>Your Query:</strong> "{original_query}"</p>
            <p><strong>Understood Intent:</strong> {str(caption.get('understood_intent', 'N/A')).replace('_', ' ').title() if caption.get('understood_intent') is not None else 'N/A'}</p>
            <p><strong>Confidence Level:</strong> {caption.get('confidence_level', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-bottom: 1rem;">📊 Chart Details</h4>
            <p><strong>Chart Type:</strong> {str(caption.get('chart_type_used', 'N/A')).title() if caption.get('chart_type_used') is not None else 'N/A'}</p>
            <p><strong>Time Period:</strong> {str(caption.get('time_period_analyzed', 'N/A')).replace('_', ' ').title() if caption.get('time_period_analyzed') is not None else 'N/A'}</p>
            <p><strong>Data Focus:</strong> {str(caption.get('data_focus', 'N/A')).title() if caption.get('data_focus') is not None else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Insights section
    if caption.get('key_insights'):
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin-top: 1rem;">
            <h4 style="color: #856404; margin-bottom: 1rem;">💡 Key Insights</h4>
            <p style="color: #856404;">{caption.get('key_insights', 'No specific insights available.')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis Reasoning
    if caption.get('analysis_reasoning'):
        st.markdown("**🔎 Analysis Reasoning:**")
        reasoning = caption.get('analysis_reasoning', [])
        if isinstance(reasoning, list):
            for reason in reasoning:
                st.write(f"- {reason}")
        else:
            st.write(f"- {reasoning}")
    
    # Full Summary
    if caption.get('summary'):
        st.markdown(f"""
        <div style="background-color: #e7f3ff; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <p><strong>📝 Summary:</strong> {caption.get('summary')}</p>
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
