import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/finance.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHART_DIR = os.getenv("CHART_DIR", "visualizations/static_charts")
EXPORT_DIR = os.getenv("EXPORT_DIR", "data/exports")
