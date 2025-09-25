# Multi-Agent Coordination

## Overview
The Household Finance AI Platform uses a modular, multi-agent architecture to separate concerns and enable coordinated, intelligent behavior across the system.

## Agents and Their Roles
- **ArchitectAgent:**
  - Orchestrates all other agents
  - Manages session state and user flow
  - Delegates tasks to specialized agents

- **DataCollectorAgent:**
  - Handles user registration, login, transaction entry, and database operations
  - Provides transaction data to other agents

- **ChartCreatorAgent:**
  - Generates explainable charts and visualizations using transaction data
  - Saves and displays charts for user and admin review

- **InsightGeneratorAgent:**
  - Produces AI-powered financial insights, predictions, anomaly detection, and recommendations
  - Uses trained ML models and transaction data

## Coordination Flow
1. **User Action:**
   - User interacts with the Streamlit UI (e.g., adds a transaction)
2. **ArchitectAgent:**
   - Receives the action and determines which agent should handle it
3. **Specialized Agent:**
   - DataCollectorAgent processes the transaction and updates the database
   - ChartCreatorAgent updates visualizations
   - InsightGeneratorAgent generates new insights
4. **Result Display:**
   - ArchitectAgent collects results and updates the UI

## Benefits
- **Separation of Concerns:** Each agent focuses on a specific domain (data, charts, insights)
- **Scalability:** New agents (e.g., for notifications, external APIs) can be added easily
- **Explainability:** Each agent can provide explainable outputs and logs
- **Maintainability:** Modular codebase is easier to test and extend

## Example Coordination
- When a transaction is added:
  - DataCollectorAgent saves it
  - ChartCreatorAgent updates charts
  - InsightGeneratorAgent analyzes the new data for insights
  - ArchitectAgent ensures all agents are synchronized

## Extensibility
- Agents can communicate via method calls, shared session state, or message passing
- Future agents (e.g., for advanced NLP, external integrations) can be coordinated by ArchitectAgent

## References
- See `agents/architect_agent.py`, `agents/data_collector.py`, `agents/chart_creator.py`, and `agents/insight_generator.py` for implementation details

## Contact
For questions or to extend agent coordination, contact the project owner.
