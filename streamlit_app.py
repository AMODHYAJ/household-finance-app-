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
    st.subheader("Transparency Dashboard / Logs")
    # Show last 50 transactions as a simple log
    logs = data_agent.db.query(Transaction).order_by(Transaction.date.desc(), Transaction.id.desc()).limit(50).all()
    log_data = [(t.id, t.user_id, t.t_type, t.category, t.amount, t.date, t.note) for t in logs]
    st.write("Recent User Actions & Data Changes:")
    st.table(log_data)
    st.write("User Login/Logout/Deletion Events:")
    event_log = st.session_state.user_event_log[-50:] if 'user_event_log' in st.session_state else []
    st.table([(e['event'], e['user'], e['timestamp']) for e in event_log])
    households = data_agent.db.query(Household).all()
    users = data_agent.db.query(User).all()
    st.subheader("Search Users & Households")
    search_user = st.text_input("Search user by username")
    search_household = st.text_input("Search household by name")
    filtered_users = [u for u in users if search_user.lower() in u.username.lower()] if search_user else users
    filtered_households = [h for h in households if search_household.lower() in h.name.lower()] if search_household else households
    st.write("Filtered Users:")
    st.table([(u.id, u.username, u.role, u.household_id) for u in filtered_users])
    st.write("Filtered Households:")
    st.table([(h.id, h.name) for h in filtered_households])
    st.subheader("All Households")
    household_data = [(h.id, h.name) for h in households]
    st.table(household_data)

    st.subheader("Delete a Household")
    household_names = [h.name for h in households]
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

    st.subheader("Export All Data (CSV)")
    if st.button("Export Users & Transactions"):
        import pandas as pd
        users = data_agent.db.query(User).all()
        transactions = data_agent.db.query(Transaction).all()
        users_df = pd.DataFrame([{"id": u.id, "username": u.username, "role": u.role, "household_id": u.household_id} for u in users])
        transactions_df = pd.DataFrame([{"id": t.id, "user_id": t.user_id, "type": t.t_type, "category": t.category, "amount": t.amount, "date": t.date, "note": t.note} for t in transactions])
        users_df.to_csv("users_export.csv", index=False)
        transactions_df.to_csv("transactions_export.csv", index=False)
        st.success("Exported users_export.csv and transactions_export.csv to project folder.")
    st.header("Admin Dashboard")
    from core.database import User
    data_agent = st.session_state.architect_agent.data_agent
    users = data_agent.db.query(User).all()
    st.subheader("All Users")
    user_data = [(u.id, u.username, u.role) for u in users]
    st.table(user_data)

    st.subheader("Delete a User")
    usernames = [u.username for u in users if u.role != "admin"]
    user_to_delete = st.selectbox("Select user to delete", usernames)
    if st.button("Delete User"):
        user_obj = data_agent.db.query(User).filter_by(username=user_to_delete).first()
        if user_obj:
            from core.database import Transaction
            # Delete all transactions for this user
            data_agent.db.query(Transaction).filter_by(user_id=user_obj.id).delete()
            data_agent.db.delete(user_obj)
            data_agent.db.commit()
            st.success(f"User '{user_to_delete}' and their transactions deleted.")
            st.rerun()

    st.subheader("Change User Role")
    user_to_change = st.selectbox("Select user to change role", [u.username for u in users if u.role != "admin"])
    new_role = st.selectbox("New Role", ["user", "admin"])
    if st.button("Change Role"):
        user_obj = data_agent.db.query(User).filter_by(username=user_to_change).first()
        if user_obj:
            user_obj.role = new_role
            data_agent.db.commit()
            st.success(f"Role for '{user_to_change}' changed to '{new_role}'.")
            st.rerun()

    st.subheader("Audit Log (Recent Actions)")
    # Simple audit log: show last 10 transactions as recent actions
    recent_transactions = data_agent.db.query(Transaction).order_by(Transaction.date.desc()).limit(10).all()
    audit_data = [(t.id, t.user_id, t.t_type, t.category, t.amount, t.date, t.note) for t in recent_transactions]
    st.table(audit_data)

    st.subheader("Privacy & Data Deletion")
    if st.button("Delete ALL Data (Irreversible)"):
        # Delete all transactions, users, households
        data_agent.db.query(Transaction).delete()
        data_agent.db.query(User).delete()
        data_agent.db.query(Household).delete()
        data_agent.db.commit()
        st.success("All data deleted. App is now reset.")
        st.rerun()

    st.subheader("Reset User Password")
    reset_user = st.selectbox("Select user to reset password", [u.username for u in users if u.role != "admin"])
    new_password = st.text_input("New Password", type="password")
    if st.button("Reset Password"):
        user_obj = data_agent.db.query(User).filter_by(username=reset_user).first()
        if user_obj:
            user_obj.password = new_password  # Assumes password field exists
            data_agent.db.commit()
            st.success(f"Password for '{reset_user}' has been reset.")

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
    data_agent = st.session_state.architect_agent.data_agent
    st.subheader("Exportable Reports")
    df = data_agent.get_transactions_df()
    if not df.empty:
        # Summary charts
        import plotly.express as px
        st.subheader("Summary Charts")
        st.markdown("**Pie Chart: Spending by Category**")
        st.caption("This chart shows how your expenses are distributed across different categories. It helps you identify where most of your money is spent.")
        st.plotly_chart(px.pie(df, names="category", values="amount", title="Spending by Category"))
        st.markdown("**Bar Chart: Transactions Over Time**")
        st.caption("This chart displays your income and expenses over time, allowing you to spot trends and changes in your financial activity.")
        st.plotly_chart(px.bar(df, x="date", y="amount", color="type", title="Transactions Over Time"))
        st.write("Download your transactions as:")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Export CSV", csv, "my_transactions.csv", "text/csv")
        try:
            import io
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.setFillColorRGB(0.12, 0.47, 0.71)
            c.rect(0, 740, 600, 40, fill=1)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(30, 755, "Household Finance AI Platform")
            c.setFont("Helvetica", 12)
            c.drawString(30, 735, "Your trusted partner for smart financial management.")
            y = 710
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(30, y, "Transactions:")
            y -= 20
            c.setFont("Helvetica", 10)
            for i, row in df.iterrows():
                c.drawString(30, y, str(row.to_dict()))
                y -= 15
                if y < 50:
                    c.showPage()
                    y = 750
            c.save()
            pdf_bytes = pdf_buffer.getvalue()
            st.download_button("Export PDF", pdf_bytes, "my_transactions.pdf", "application/pdf")
        except Exception as e:
            st.warning(f"PDF export failed: {e}")
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
        st.dataframe(recent_df)
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
        # Input sanitization
        sanitized_category = category.strip()
        if not sanitized_category or not sanitized_category.replace(' ', '').isalnum():
            st.error("Category must be non-empty and contain only letters, numbers, and spaces.")
            return
        sanitized_note = note.strip() if note else None
        if sanitized_note and len(sanitized_note) > 200:
            st.error("Note is too long (max 200 characters).")
            return
        try:
            data_agent = st.session_state.architect_agent.data_agent
            from core.schemas import TransactionIn
            from core.database import Transaction
            from core.utils import parse_date
            transaction_data = TransactionIn(
                t_type=t_type,
                category=sanitized_category,
                amount=float(amount),
                date=date_input,
                note=sanitized_note if sanitized_note else None
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
            # Log transaction event
            if 'user_event_log' not in st.session_state:
                st.session_state.user_event_log = []
            st.session_state.user_event_log.append({
                "event": "add_transaction",
                "user": st.session_state.current_user,
                "timestamp": datetime.now().isoformat()
            })
            st.success("✅ Transaction added successfully!")
            st.balloons()
            st.info(f"Added: {t_type.title()} - {sanitized_category} - ${amount:.2f} on {date_input}")
        except Exception as e:
            st.error(f"❌ Failed to add transaction: {str(e)}")
            try:
                data_agent.db.rollback()
            except Exception:
                pass

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
            st.dataframe(display_df)
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

# Add these helper functions at the bottom of your streamlit_app.py
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

    st.header("🧠 AI-Powered Financial Insights")

    insight_agent = st.session_state.architect_agent.insight_agent
    data_agent = st.session_state.architect_agent.data_agent
    df = data_agent.get_transactions_df()

    if df is None or df.empty:
        st.info("No data available for insights. Add some transactions first!")
        return

    # Use the original generate_insights method to avoid complexity
    try:
        insights = insight_agent.generate_insights(df)
        
        # Simple tab structure
        tab1, tab2 = st.tabs(["📊 Summary", "🤖 AI Insights"])

        with tab1:
            st.subheader("Financial Summary")
            summary = insight_agent.summarize(df)
            if isinstance(summary, dict):
                col1, col2, col3 = st.columns(3)
                with col1: 
                    st.metric("Total Income", f"${summary.get('Total Income', 0):,.2f}")
                with col2: 
                    st.metric("Total Expenses", f"${summary.get('Total Expense', 0):,.2f}")
                with col3: 
                    st.metric("Net Savings", f"${summary.get('Savings', 0):,.2f}")

        with tab2:
            st.subheader("AI-Generated Insights")
            for insight in insights:
                if "⚠" in insight or "anomaly" in insight.lower():
                    st.warning(insight)
                elif "✅" in insight or "saved" in insight.lower():
                    st.success(insight)
                else:
                    st.info(insight)

    except Exception as e:
        st.error(f"❌ Failed to generate insights: {str(e)}")

def logout():
    # Log the logout event
    if 'user_event_log' not in st.session_state:
        st.session_state.user_event_log = []
    st.session_state.user_event_log.append({
        "event": "logout",
        "user": st.session_state.current_user,
        "timestamp": datetime.now().isoformat()  # This should work with 'from datetime import datetime'
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

if __name__ == "__main__":
    main()