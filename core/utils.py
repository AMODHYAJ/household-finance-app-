from datetime import datetime

def parse_date(s: str):
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()
