# System Architecture Diagram

Below is a textual description of the system architecture for the Household Finance AI Platform. For a visual diagram, use tools like draw.io, Lucidchart, or Mermaid (Markdown).

## Components

- **Frontend (Streamlit UI):**
  - User registration, login, dashboard, transaction entry, charts, admin features
  - Communicates with backend agents via session state

- **Agents (Python Classes):**
  - `ArchitectAgent`: Orchestrates other agents and manages session state
  - `DataCollectorAgent`: Handles transaction entry, encryption, and database operations
  - `ChartCreatorAgent`: Generates charts and visualizations
  - `InsightGeneratorAgent`: Provides AI-powered insights

- **Backend (Core Modules):**
  - `core/database.py`: SQLAlchemy ORM models for User, Household, Transaction
  - `core/security.py`: Password hashing, encryption/decryption functions
  - `core/utils.py`: Utility functions (date parsing, etc.)
  - `api/app.py`: FastAPI endpoints (if used for agent communication)

- **ML Models (models/):**
  - Expense predictor, anomaly detector, category classifier (trained via `train_models.py`)

- **Data Storage:**
  - SQLite database (`data/finance.db`)
  - Pickle files for ML models
  - CSV files for datasets

- **Transparency & Logging:**
  - In-memory event log (`user_event_log` in session state)
  - Transparency dashboard in UI

- **Exportable Reports:**
  - PDF/Excel generation via `reportlab` and other libraries

## Data Flow
1. User interacts with Streamlit UI
2. UI calls agent methods via session state
3. Agents interact with database and ML models
4. Sensitive data is encrypted before storage
5. User actions are logged in event log
6. Charts and reports are generated and displayed/exported

## Example Mermaid Diagram

```mermaid
flowchart TD
    A[User (Streamlit UI)] -->|register/login| B(ArchitectAgent)
    B --> C(DataCollectorAgent)
    B --> D(ChartCreatorAgent)
    B --> E(InsightGeneratorAgent)
    C --> F[(Database)]
    C --> G[(Encryption)]
    D --> H[(Charts)]
    E --> I[(ML Models)]
    B --> J[(Transparency Log)]
    B --> K[(Exportable Reports)]
```

## How to Use
- Copy the Mermaid diagram above into your markdown documentation or use a diagramming tool to visualize.
- Update with any additional components as needed.

## Contact
For architecture questions, contact the project owner.
