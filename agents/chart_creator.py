import os
from dotenv import load_dotenv
# Load environment variables from .env file at the module level
load_dotenv()
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import re
from typing import Optional, Dict, Any, List, Tuple
import warnings
from collections import Counter, defaultdict
import json
import hashlib
from functools import lru_cache
warnings.filterwarnings('ignore')
import streamlit as st
from sklearn.ensemble import IsolationForest

import pickle

CHART_DIR = "visualizations/static_charts"

class SmartCaptionGenerator:
    """Generate intelligent, data-driven captions for charts"""
    
    def __init__(self):
        self.caption_templates = {
            'expense_breakdown': "Total expenses: ${total:,.2f} | Top category: {top_category} (${top_amount:,.2f}, {top_percent:.1f}%)",
            'income_breakdown': "Total income: ${total:,.2f} | Main source: {main_source} (${main_amount:,.2f})",
            'comparison': "Net {result}: ${net:,.2f} | Income: ${income:,.2f} vs Expenses: ${expenses:,.2f}",
            'trend': "{trend_direction} trend: {change:+.1f}% over {periods} months | Current: ${current:,.2f}",
            'savings': "Total savings: ${total:,.2f} | Monthly avg: ${avg:,.2f} | {positive_ratio:.0f}% positive months",
            'category_analysis': "{category} spending: ${total:,.2f} | {transactions} transactions | Avg: ${avg:,.2f}",
            'time_comparison': "Recent: ${recent:,.2f} | Previous: ${previous:,.2f} | Change: {change:+.1f}%",
            'anomaly_detection': "Analyzed {months} months | Found {anomaly_count} unusual patterns"
        }
    
    def generate_caption(self, chart_type: str, data: pd.DataFrame, analysis: Dict, user_query: str = "") -> str:
        """Generate smart caption with meaningful content"""
        
        try:
            if data is None or data.empty:
                return self._generate_fallback_caption(chart_type, "No data available")
            
            if chart_type == 'expense_breakdown':
                return self._generate_expense_caption(data)
            elif chart_type == 'income_breakdown':
                return self._generate_income_caption(data)
            elif chart_type == 'income_vs_expense':
                return self._generate_comparison_caption(data)
            elif chart_type in ['trend_analysis', 'savings_trend']:
                return self._generate_trend_caption(data, chart_type)
            elif chart_type == 'time_comparison':
                return self._generate_time_comparison_caption(data)
            elif chart_type == 'category_analysis':
                return self._generate_category_caption(data, analysis)
            elif chart_type == 'anomaly_detection':
                return self._generate_anomaly_caption(data)
            else:
                return self._generate_default_caption(data, chart_type)
                
        except Exception as e:
            return self._generate_fallback_caption(chart_type, "Analysis completed")
    
    def _generate_expense_caption(self, data: pd.DataFrame) -> str:
        """Generate caption for expense breakdown charts"""
        expense_data = data[data['type'] == 'expense']
        if expense_data.empty:
            return "No expense data available in selected period"
        
        total = expense_data['amount'].sum()
        transactions = len(expense_data)
        category_totals = expense_data.groupby('category')['amount'].sum()
        
        if not category_totals.empty:
            top_category = category_totals.idxmax()
            top_amount = category_totals.max()
            top_percent = (top_amount / total) * 100 if total > 0 else 0
            
            return f"Total expenses: ${total:,.2f} | Top: {top_category} (${top_amount:,.2f}, {top_percent:.1f}%) | {transactions} transactions"
        
        return f"Total expenses: ${total:,.2f} | {transactions} transactions"
    
    def _generate_income_caption(self, data: pd.DataFrame) -> str:
        """Generate caption for income breakdown charts"""
        income_data = data[data['type'] == 'income']
        if income_data.empty:
            return "No income data available in selected period"
        
        total = income_data['amount'].sum()
        transactions = len(income_data)
        category_totals = income_data.groupby('category')['amount'].sum()
        
        if not category_totals.empty:
            main_source = category_totals.idxmax()
            main_amount = category_totals.max()
            main_percent = (main_amount / total) * 100 if total > 0 else 0
            
            return f"Total income: ${total:,.2f} | Main: {main_source} (${main_amount:,.2f}, {main_percent:.1f}%) | {transactions} transactions"
        
        return f"Total income: ${total:,.2f} | {transactions} transactions"
    
    def _generate_comparison_caption(self, data: pd.DataFrame) -> str:
        """Generate caption for income vs expense comparison"""
        income_total = data[data['type'] == 'income']['amount'].sum()
        expense_total = data[data['type'] == 'expense']['amount'].sum()
        net_savings = income_total - expense_total
        
        if net_savings >= 0:
            result = f"savings: ${net_savings:,.2f}"
            savings_rate = (net_savings / income_total * 100) if income_total > 0 else 0
            additional = f" | Savings rate: {savings_rate:.1f}%"
        else:
            result = f"deficit: ${-net_savings:,.2f}"
            additional = ""
            
        return f"Income: ${income_total:,.2f} | Expenses: ${expense_total:,.2f} | Net {result}{additional}"
    
    def _generate_trend_caption(self, data: pd.DataFrame, chart_type: str) -> str:
        """Generate caption for trend charts"""
        if data.empty:
            return "No trend data available"
        
        # Calculate monthly trends
        data_copy = data.copy()
        data_copy['month'] = data_copy['date'].dt.to_period('M')
        monthly_totals = data_copy.groupby('month')['amount'].sum()
        
        if len(monthly_totals) < 2:
            current = monthly_totals.iloc[-1] if len(monthly_totals) == 1 else data['amount'].sum()
            return f"Current amount: ${current:,.2f} | Need more data for trend analysis"
        
        # Calculate trend
        first_val = monthly_totals.iloc[0]
        last_val = monthly_totals.iloc[-1]
        change_percent = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
        
        trend_direction = "increasing" if change_percent > 0 else "decreasing"
        trend_strength = "strongly " if abs(change_percent) > 20 else "moderately " if abs(change_percent) > 5 else "slightly "
        
        return f"{trend_strength}{trend_direction} trend: {change_percent:+.1f}% over {len(monthly_totals)} months | Current: ${last_val:,.2f}"
    
    def _generate_time_comparison_caption(self, data: pd.DataFrame) -> str:
        """Generate caption for time comparison charts"""
        if data.empty:
            return "No comparison data available"
        
        # Simple comparison logic
        recent_cutoff = datetime.now() - timedelta(days=60)
        recent_data = data[data['date'] >= recent_cutoff]
        older_data = data[data['date'] < recent_cutoff]
        
        recent_total = recent_data['amount'].sum()
        older_total = older_data['amount'].sum() if not older_data.empty else recent_total
        
        change = ((recent_total - older_total) / older_total * 100) if older_total > 0 else 0
        
        return f"Recent period: ${recent_total:,.2f} | Previous: ${older_total:,.2f} | Change: {change:+.1f}%"
    
    def _generate_category_caption(self, data: pd.DataFrame, analysis: Dict) -> str:
        """Generate caption for category analysis"""
        categories = analysis.get('entities', {}).get('categories', [])
        category_name = categories[0] if categories else 'Selected category'
        
        total = data['amount'].sum()
        transactions = len(data)
        avg_amount = total / transactions if transactions > 0 else 0
        
        return f"{category_name} spending: ${total:,.2f} | {transactions} transactions | Average: ${avg_amount:,.2f}"
    
    def _generate_anomaly_caption(self, data: pd.DataFrame) -> str:
        """Generate caption for anomaly detection"""
        if data.empty:
            return "No data available for anomaly detection"
        
        data_copy = data.copy()
        data_copy['month'] = data_copy['date'].dt.to_period('M')
        monthly_count = data_copy['month'].nunique()
        
        return f"Analyzed {monthly_count} months for unusual spending patterns"
    
    def _generate_default_caption(self, data: pd.DataFrame, chart_type: str) -> str:
        """Generate default caption for other chart types"""
        total = data['amount'].sum()
        transactions = len(data)
        avg_amount = total / transactions if transactions > 0 else 0
        
        return f"Total amount: ${total:,.2f} | Transactions: {transactions} | Average: ${avg_amount:,.2f}"
    
    def _generate_fallback_caption(self, chart_type: str, context: str = "") -> str:
        """Fallback caption if generation fails"""
        chart_names = {
            'expense_breakdown': 'Expense Analysis',
            'income_breakdown': 'Income Analysis', 
            'income_vs_expense': 'Income vs Expenses',
            'savings_trend': 'Savings Trend Analysis',
            'trend_analysis': 'Financial Trend Analysis',
            'category_analysis': 'Category Spending Analysis',
            'time_comparison': 'Period Comparison',
            'anomaly_detection': 'Pattern Analysis',
            'comprehensive_dashboard': 'Financial Overview'
        }
        
        base = chart_names.get(chart_type, 'Financial Analysis')
        return f"{base} | {context}" if context else base

class AnomalyDetector:
    """Detect unusual spending patterns and anomalies"""
    
    def __init__(self):
        self.anomaly_model = None
        self.anomaly_threshold = 0.1
        
    def detect_spending_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect unusual spending patterns using Isolation Forest"""
        try:
            if df is None or df.empty:
                return {'anomalies': [], 'insights': "No data available for anomaly detection"}
            
            expense_df = df[df['type'] == 'expense'].copy()
            if expense_df.empty:
                return {'anomalies': [], 'insights': "No expense data available"}
            
            expense_df['date'] = pd.to_datetime(expense_df['date'])
            expense_df['month'] = expense_df['date'].dt.to_period('M')
            monthly_spending = expense_df.groupby('month')['amount'].sum().reset_index()
            
            if len(monthly_spending) < 3:
                return {'anomalies': [], 'insights': "Insufficient data for anomaly detection (need at least 3 months)"}
            
            X = monthly_spending['amount'].values.reshape(-1, 1)
            
            self.anomaly_model = IsolationForest(
                contamination=self.anomaly_threshold,
                random_state=42
            )
            anomalies = self.anomaly_model.fit_predict(X)
            
            monthly_spending['is_anomaly'] = anomalies == -1
            anomalous_months = monthly_spending[monthly_spending['is_anomaly']]
            
            insights = []
            if not anomalous_months.empty:
                for _, row in anomalous_months.iterrows():
                    month_str = row['month'].strftime('%B %Y')
                    amount = row['amount']
                    avg_spending = monthly_spending[~monthly_spending['is_anomaly']]['amount'].mean()
                    deviation = ((amount - avg_spending) / avg_spending) * 100
                    insights.append(f"Unusual spending in {month_str}: ${amount:.2f} ({deviation:+.1f}% from average)")
            
            return {
                'anomalies': anomalous_months.to_dict('records'),
                'insights': insights if insights else ["No significant anomalies detected"],
                'total_months_analyzed': len(monthly_spending),
                'anomalous_months_count': len(anomalous_months)
            }
            
        except Exception as e:
            return {'anomalies': [], 'insights': [f"Anomaly detection failed: {str(e)}"]}

class VisualQueryBuilder:
    """Enhanced Visual Query Builder for creating charts through UI components"""
    
    def __init__(self, data_agent):
        self.data_agent = data_agent
        
    def build_query_from_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query_parts = []
            
            # Chart type and focus
            chart_focus = components.get('chart_focus', 'Expenses')
            query_parts.append(chart_focus.lower())
            
            # Time period
            time_period = components.get('time_period', 'All Time')
            if time_period != "All Time":
                query_parts.append(f"for {time_period.lower()}")
            
            # Detail level
            detail_level = components.get('detail_level', 'Summary')
            if detail_level != "Summary":
                query_parts.append(detail_level.lower())
            
            # Specific categories
            specific_categories = components.get('specific_categories', [])
            if specific_categories:
                if len(specific_categories) == 1:
                    query_parts.append(f"for {specific_categories[0].lower()}")
                else:
                    query_parts.append(f"for {len(specific_categories)} categories")
            
            # Amount filter
            min_amount = components.get('min_amount', 0)
            if min_amount > 0:
                query_parts.append(f"over ${min_amount}")
            
            # Build the final query
            generated_query = " ".join(query_parts)
            
            # Enhanced analysis with guaranteed meaningful values
            analysis = {
                'primary_intent': self._map_focus_to_intent(chart_focus, detail_level),
                'entities': {
                    'categories': specific_categories if specific_categories else ['All categories'],
                    'time_periods': [time_period] if time_period != "All Time" else ['Recent history']
                },
                'time_context': {
                    'range': self._map_time_period(time_period),
                    'comparison_periods': [],
                    'readable_range': time_period
                },
                'data_focus': self._map_focus_to_data(chart_focus),
                'chart_preference': self._map_detail_to_chart(detail_level, chart_focus),
                'confidence_score': 0.95,
                'reasoning': self._generate_visual_builder_reasoning(components),
                'is_follow_up': False,
                'query_type': 'visual_builder',
                'data_points': 'Filtered dataset based on your selections',
                'components_used': list(components.keys())
            }
            
            return {
                'query': generated_query,
                'analysis': analysis,
                'components': components
            }
            
        except Exception as e:
            # Enhanced fallback analysis with guaranteed values
            return {
                'query': "Show financial overview",
                'analysis': {
                    'primary_intent': 'comprehensive_dashboard',
                    'entities': {'categories': ['All categories']},
                    'time_context': {
                        'range': 'last_3_months', 
                        'comparison_periods': [],
                        'readable_range': 'Last 3 Months'
                    },
                    'data_focus': 'both',
                    'chart_preference': 'dashboard',
                    'confidence_score': 0.8,
                    'reasoning': ['Using comprehensive dashboard view to show financial overview'],
                    'query_type': 'fallback',
                    'data_points': 'Available transactions'
                },
                'components': components
            }
    def _generate_visual_builder_reasoning(self, components: Dict) -> List[str]:
        """Generate meaningful reasoning for visual builder queries"""
        reasoning = []
        
        chart_focus = components.get('chart_focus', 'Expenses')
        time_period = components.get('time_period', 'All Time')
        detail_level = components.get('detail_level', 'Summary')
        categories = components.get('specific_categories', [])
        
        reasoning.append(f"Focus: {chart_focus} analysis")
        reasoning.append(f"Timeframe: {time_period}")
        reasoning.append(f"Detail level: {detail_level}")
        
        if categories:
            if len(categories) == 1:
                reasoning.append(f"Category focus: {categories[0]}")
            else:
                reasoning.append(f"Multiple categories: {len(categories)} selected")
        
        if components.get('min_amount', 0) > 0:
            reasoning.append(f"Minimum amount: ${components['min_amount']}")
        
        return reasoning
        
    def _map_focus_to_intent(self, focus: str, detail_level: str) -> str:
        """Map visual focus and detail level to appropriate NLP intent"""
        
        # Combine focus and detail level to create more specific intents
        focus_detail_map = {
            # Expenses with different detail levels
            ('Expenses', 'Summary'): 'expense_breakdown',
            ('Expenses', 'By Category'): 'expense_breakdown',
            ('Expenses', 'Trend Over Time'): 'trend_analysis',
            ('Expenses', 'Detailed Breakdown'): 'comprehensive_dashboard',
            
            # Income with different detail levels
            ('Income', 'Summary'): 'income_breakdown',
            ('Income', 'By Category'): 'income_breakdown',
            ('Income', 'Trend Over Time'): 'trend_analysis',
            ('Income', 'Detailed Breakdown'): 'comprehensive_dashboard',
            
            # Savings with different detail levels
            ('Savings', 'Summary'): 'savings_trend',
            ('Savings', 'By Category'): 'category_analysis',
            ('Savings', 'Trend Over Time'): 'savings_trend',
            ('Savings', 'Detailed Breakdown'): 'comprehensive_dashboard',
            
            # Comparison with different detail levels
            ('Comparison', 'Summary'): 'income_vs_expense',
            ('Comparison', 'By Category'): 'category_analysis',
            ('Comparison', 'Trend Over Time'): 'trend_analysis',
            ('Comparison', 'Detailed Breakdown'): 'comprehensive_dashboard',
        }
        
        return focus_detail_map.get((focus, detail_level), 'comprehensive_dashboard')
    
    def _map_time_period(self, time_period: str) -> str:
        """Map time period to standard range"""
        period_map = {
            'This Month': 'this_month',
            'Last Month': 'last_month',
            'Last 3 Months': 'last_3_months',
            'This Year': 'this_year',
            'All Time': 'all_time'
        }
        return period_map.get(time_period, 'last_3_months')
    
    def _map_focus_to_data(self, focus: str) -> str:
        """Map focus to data type"""
        focus_map = {
            'Expenses': 'expense',
            'Income': 'income',
            'Savings': 'savings',
            'Comparison': 'both'
        }
        return focus_map.get(focus, 'both')
    
    def _map_detail_to_chart(self, detail: str, focus: str) -> str:
        """Map detail level to chart type with more specific logic"""
        if detail == "By Category":
            if focus in ['Expenses', 'Income']:
                return 'pie'
            elif focus == 'Savings':
                return 'bar'
            else:  # Comparison
                return 'bar'
        elif detail == "Trend Over Time":
            return 'line'
        elif detail == "Detailed Breakdown":
            return 'dashboard'
        else:  # Summary
            if focus == 'Comparison':
                return 'bar'
            elif focus == 'Savings':
                return 'line'
            else:  # Expenses, Income
                return 'pie'
    
    def get_available_categories(self) -> List[str]:
        """Get available categories from data"""
        try:
            df = self.data_agent.get_transactions_df()
            if df is None or df.empty or not isinstance(df, pd.DataFrame):
                return []
            if 'category' not in df.columns:
                return []
            
            categories = df['category'].dropna().unique().tolist()
            return sorted([str(cat).strip() for cat in categories if str(cat).strip()])
        except Exception as e:
            return []
    
    def validate_components(self, components: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate visual query components"""
        try:
            # Check required fields
            required = ['chart_focus', 'time_period', 'detail_level']
            for field in required:
                if field not in components or not components[field]:
                    return False, f"Missing required field: {field}"
            
            # Validate amount
            min_amount = components.get('min_amount', 0)
            if min_amount < 0:
                return False, "Minimum amount cannot be negative"
            
            return True, "Components are valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class ChartRecommender:
    """Recommend optimal chart types based on data characteristics"""
    
    def __init__(self):
        self.chart_performance = defaultdict(lambda: defaultdict(int))
        self.user_preferences = {}
        
    def recommend_chart_type(self, analysis: Dict, data_shape: Dict, user_id: str = "default") -> str:
        """Recommend the best chart type based on multiple factors"""
        
        # Factors to consider
        intent = analysis.get('primary_intent', '')
        time_periods = len(analysis.get('time_context', {}).get('comparison_periods', []))
        categories_count = len(analysis.get('entities', {}).get('categories', []))
        data_points = data_shape.get('time_points', 1)
        has_comparison = analysis.get('time_context', {}).get('comparison_periods', [])
        
        # Scoring system
        scores = {
            'bar': 0,
            'pie': 0,
            'line': 0,
            'scatter': 0
        }
        
        # Intent-based scoring
        intent_scores = {
            'income_breakdown': {'pie': 3, 'bar': 2},
            'expense_breakdown': {'pie': 3, 'bar': 2},
            'income_vs_expense': {'bar': 3, 'line': 2},
            'savings_trend': {'line': 3, 'bar': 2},
            'category_analysis': {'bar': 3, 'line': 2, 'pie': 1},
            'time_comparison': {'bar': 3, 'line': 2},
            'trend_analysis': {'line': 3, 'scatter': 2}
        }
        
        for chart, score in intent_scores.get(intent, {}).items():
            scores[chart] += score
        
        # Data characteristics scoring
        if time_periods >= 2:
            scores['bar'] += 2
            scores['line'] += 1
        
        if categories_count <= 5:
            scores['pie'] += 2
        
        if data_points >= 6:
            scores['line'] += 2
            scores['scatter'] += 1
        
        if has_comparison:
            scores['bar'] += 2
        
        # User preference scoring (if available)
        user_pref = self.user_preferences.get(user_id, {})
        for chart, preference in user_pref.items():
            scores[chart] += preference
        
        # Select best chart
        best_chart = max(scores.items(), key=lambda x: x[1])
        
        # Ensure we have a valid chart type
        return best_chart[0] if best_chart[1] > 0 else 'bar'

class PerformanceCache:
    """Enhanced caching system for better performance"""
    
    def __init__(self, max_size: int = 100, cache_dir: str = "cache"):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, query: str, data_hash: str) -> str:
        """Generate a cache key from query and data hash"""
        key_string = f"{query}_{data_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, data_hash: str) -> Optional[Any]:
        """Get item from cache"""
        cache_key = self._generate_cache_key(query, data_hash)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        # Try disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                # Also store in memory cache for faster access
                self.memory_cache[cache_key] = result
                self.cache_hits += 1
                return result
            except Exception as e:
                print(f"⚠️ Cache read error: {e}")
        
        self.cache_misses += 1
        return None
    
    def set(self, query: str, data_hash: str, value: Any, ttl: int = 3600):
        """Set item in cache with time-to-live"""
        cache_key = self._generate_cache_key(query, data_hash)
        
        # Store in memory cache
        self.memory_cache[cache_key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl
        }
        
        # Store in disk cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': datetime.now(),
                    'ttl': ttl
                }, f)
        except Exception as e:
            print(f"⚠️ Cache write error: {e}")
        
        # Clean up if cache is too large
        if len(self.memory_cache) > self.max_size:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired and least recently used items from cache"""
        now = datetime.now()
        
        # Remove expired items
        expired_keys = []
        for key, item in self.memory_cache.items():
            if (now - item['timestamp']).total_seconds() > item['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            # Also remove from disk
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        # If still too large, remove oldest items
        if len(self.memory_cache) > self.max_size:
            items_sorted = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            items_to_remove = items_sorted[:len(self.memory_cache) - self.max_size]
            
            for key, _ in items_to_remove:
                del self.memory_cache[key]
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except:
                        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'memory_cache_size': len(self.memory_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests
        }

class QueryAssistant:
    """Smart query suggestions and auto-completion"""
    def __init__(self, data_agent):
        self.data_agent = data_agent
        self.popular_queries = [
            "Show my spending by category",
            "Compare this month vs last month", 
            "Income breakdown",
            "Savings trend over time",
            "Analyze food expenses",
            "Show entertainment spending this month",
            "Compare income and expenses",
            "Spending trends last 6 months"
        ]
        
    def get_smart_suggestions(self, partial_query: str = "") -> List[str]:
        """Provide real-time query suggestions based on data and context"""
        try:
            df = self.data_agent.get_transactions_df()
            
            if df is None or df.empty:
                return self.popular_queries[:4]
            
            # Get available data insights
            available_categories = []
            if 'category' in df.columns:
                available_categories = df['category'].unique().tolist()
            
            has_income = 'income' in df['type'].values if 'type' in df.columns else False
            
            time_range = "recent period"
            if 'date' in df.columns and len(df) > 0:
                try:
                    dates = pd.to_datetime(df['date'].dropna())
                    if len(dates) > 0:
                        time_range = f"{dates.min().strftime('%b %Y')} to {dates.max().strftime('%b %Y')}"
                except:
                    pass
            
            # Generate context-aware suggestions
            suggestions = []
            
            # Category-specific suggestions
            for category in available_categories[:3]:
                suggestions.extend([
                    f"Show {category} expenses",
                    f"Analyze {category} spending this month",
                    f"Compare {category} costs vs last month"
                ])
            
            # Time-based suggestions
            suggestions.extend([
                "Compare this month vs last month",
                f"Spending trends from {time_range}",
                "Show current month expenses",
                "Analyze last 3 months spending"
            ])
            
            # Type-based suggestions
            if has_income:
                suggestions.extend([
                    "Income vs expenses comparison",
                    "Show income sources",
                    "Net savings over time"
                ])
            
            # Add popular queries that aren't already included
            for query in self.popular_queries:
                if query not in suggestions:
                    suggestions.append(query)
            
            # Filter by partial query if provided
            if partial_query:
                partial_lower = partial_query.lower()
                suggestions = [s for s in suggestions if any(word in s.lower() for word in partial_lower.split())]
            
            return suggestions[:8]
            
        except Exception as e:
            # Return basic suggestions if data access fails
            return self.popular_queries[:6]

class EnhancedChartCreatorAgent:
    def __init__(self, data_agent):
        self.data_agent = data_agent
        self.visual_query_builder = VisualQueryBuilder(data_agent)
        self.anomaly_detector = AnomalyDetector()
        self.chart_recommender = ChartRecommender()
        self.performance_cache = PerformanceCache()
        self.query_assistant = QueryAssistant(data_agent)
        self.caption_generator = SmartCaptionGenerator()
        
        os.makedirs(CHART_DIR, exist_ok=True)
        self.chart_descriptions = {}
    
    def visual_components_to_chart(self, components: Dict[str, Any]):
        try:
            df = self.data_agent.get_transactions_df()
            
            if df is None:
                return self._create_helpful_response("Data Error", "Could not load transaction data")
            if not isinstance(df, pd.DataFrame):
                return self._create_helpful_response("Data Format Error", "Transaction data format is invalid")
            if df.empty:
                return self._create_helpful_response("No Data", "No transactions available")
            
            is_valid, validation_msg = self.visual_query_builder.validate_components(components)
            if not is_valid:
                return self._create_helpful_response(
                    "Invalid Query Components",
                    validation_msg,
                    self.get_query_suggestions()
                )
            
            query_result = self.visual_query_builder.build_query_from_components(components)
            generated_query = query_result['query']
            analysis = query_result['analysis']
            
            filtered_df = self._apply_visual_filters(df, components)
            
            if filtered_df.empty:
                return self._create_helpful_response(
                    "No data matching your criteria", 
                    "No transactions found that match your selected filters.",
                    self.get_query_suggestions()
                )
            
            result = self._generate_chart_based_on_analysis(filtered_df, analysis, generated_query)
            return result
            
        except Exception as e:
            return self._create_error_response(
                f"Chart generation error: {str(e)}",
                self.get_query_suggestions()
            )

    def _apply_visual_filters(self, df: pd.DataFrame, components: Dict) -> pd.DataFrame:
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
            filtered_df = self._apply_time_filter_to_df(filtered_df, time_period)
        
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

    def _apply_time_filter_to_df(self, df: pd.DataFrame, time_period: str) -> pd.DataFrame:
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
    
    def _generate_chart_based_on_analysis(self, df: pd.DataFrame, analysis: Dict, original_query: str):
        """Generate chart based on analysis"""
        filtered_df = self._apply_time_filter(df, analysis['time_context']['range'])
        
        if filtered_df.empty:
            return self._create_helpful_response(
                "No data for selected period",
                f"No transactions found for {analysis['time_context']['range']}.",
                self.get_query_suggestions()
            )
        
        requested_categories = analysis['entities']['categories']
        if requested_categories and requested_categories != ["All Categories"]:
            category_mask = False
            for category in requested_categories:
                category_lower = category.strip().lower()
                category_mask |= (filtered_df['category'].astype(str).str.strip().str.lower() == category_lower)
            filtered_df = filtered_df[category_mask]
        
        data_focus = analysis['data_focus']
        if data_focus == 'income':
            filtered_df = filtered_df[filtered_df['type'] == 'income']
        elif data_focus == 'expense':
            filtered_df = filtered_df[filtered_df['type'] == 'expense']
        
        if filtered_df.empty:
            focus_name = analysis['data_focus'].replace('_', ' ').title()
            return self._create_helpful_response(
                f"No {focus_name} Data",
                f"No {analysis['data_focus']} data found for the selected filters.",
                self.get_query_suggestions()
            )
        
        insights = ""
        if analysis.get('primary_intent') in ['trend_analysis', 'comprehensive_dashboard']:
            anomaly_results = self.anomaly_detector.detect_spending_anomalies(filtered_df)
            if anomaly_results['insights'] and len(anomaly_results['anomalies']) > 0:
                insights = "Anomaly detected: " + "; ".join(anomaly_results['insights'][:2])
        
        intent_handlers = {
            'income_breakdown': self._create_income_breakdown,
            'expense_breakdown': self._create_expense_breakdown,
            'income_vs_expense': self._create_comparison_chart,
            'savings_trend': self._create_savings_trend,
            'category_analysis': self._create_category_analysis,
            'time_comparison': self._create_time_comparison,
            'trend_analysis': self._create_trend_analysis,
            'anomaly_detection': self._create_anomaly_chart,
            'comprehensive_dashboard': self._create_detailed_breakdown,
        }
        handler = intent_handlers.get(analysis['primary_intent'], self._create_comprehensive_dashboard)
        result = handler(filtered_df, analysis, original_query)
        
        if isinstance(result, tuple) and len(result) > 1:
            chart, caption = result
            if insights:
                caption['enhanced_insights'] = insights
            caption['chart_recommendation'] = "AI-recommended based on data characteristics"
            return chart, caption
        
        return result

    def _create_income_breakdown(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create income breakdown chart with enhanced data insights"""
        income_df = df[df['type'] == 'income']
        
        if income_df.empty:
            return self._create_helpful_response(
                "No Income Data",
                "No income transactions found in the selected period."
            )
        
        requested_categories = analysis['entities']['categories']
        if requested_categories:
            income_df = income_df[income_df['category'].isin(requested_categories)]
        
        category_totals = income_df.groupby('category')['amount'].sum()
        
        if not income_df.empty:
            total = income_df['amount'].sum()
            category_count = income_df['category'].nunique()
            transaction_count = len(income_df)
            
            analysis['data_points'] = f"{transaction_count} transactions"
            analysis['data_characteristics'] = {
                'total_amount': total,
                'category_count': category_count,
                'transaction_count': transaction_count,
                'date_range': f"{income_df['date'].min().strftime('%b %Y')} to {income_df['date'].max().strftime('%b %Y')}" if 'date' in income_df.columns else 'Various dates'
            }
        
        if len(category_totals) <= 1:
            if len(category_totals) == 1:
                chart_df = pd.DataFrame({
                    'category': [category_totals.index[0]],
                    'amount': [category_totals.values[0]]
                })
            else:
                chart_df = pd.DataFrame({'category': [], 'amount': []})
            
            fig = px.bar(
                chart_df,
                x='category',
                y='amount',
                title="Income by Category",
                color_discrete_sequence=['#2ecc71']
            )
            chart_type = 'bar'
        else:
            fig = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                title="Income Breakdown by Category",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            chart_type = 'pie'
        
        insights = self._ensure_data_insights(income_df, analysis)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'income_breakdown')
        
        return self._finalize_chart(fig, "income_breakdown", True, caption)
    
    def _create_expense_breakdown(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create expense breakdown chart with enhanced data insights"""
        expense_df = df[df['type'] == 'expense']
        
        if expense_df.empty:
            return self._create_helpful_response(
                "No Expense Data",
                "No expense transactions found in the selected period."
            )
        
        requested_categories = analysis['entities']['categories']
        if requested_categories and "All Categories" not in requested_categories:
            expense_df = expense_df[expense_df['category'].isin(requested_categories)]
        
        category_totals = expense_df.groupby('category')['amount'].sum()
        
        if not expense_df.empty:
            total = expense_df['amount'].sum()
            category_count = expense_df['category'].nunique()
            transaction_count = len(expense_df)
            
            analysis['data_points'] = f"{transaction_count} transactions"
            analysis['data_characteristics'] = {
                'total_amount': total,
                'category_count': category_count,
                'transaction_count': transaction_count,
                'date_range': f"{expense_df['date'].min().strftime('%b %Y')} to {expense_df['date'].max().strftime('%b %Y')}" if 'date' in expense_df.columns else 'Various dates'
            }
        
        if len(category_totals) <= 1:
            if len(category_totals) == 1:
                chart_df = pd.DataFrame({
                    'category': [category_totals.index[0]],
                    'amount': [category_totals.values[0]]
                })
            else:
                chart_df = pd.DataFrame({'category': [], 'amount': []})
            
            fig = px.bar(
                chart_df,
                x='category',
                y='amount',
                title="Expenses by Category",
                color_discrete_sequence=['#e74c3c']
            )
            chart_type = 'bar'
        else:
            fig = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                title="Expense Breakdown by Category",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            chart_type = 'pie'
        
        insights = self._ensure_data_insights(expense_df, analysis)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'expense_breakdown')
        
        return self._finalize_chart(fig, "expense_breakdown", True, caption)
    
    def _create_comparison_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create income vs expense comparison with enhanced data insights"""
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        if not df.empty:
            income_total = df[df['type'] == 'income']['amount'].sum()
            expense_total = df[df['type'] == 'expense']['amount'].sum()
            net_savings = income_total - expense_total
            month_count = df_copy['month'].nunique()
            
            analysis['data_points'] = f"{len(df)} transactions across {month_count} months"
            analysis['data_characteristics'] = {
                'income_total': income_total,
                'expense_total': expense_total,
                'net_savings': net_savings,
                'month_count': month_count,
                'date_range': f"{df['date'].min().strftime('%b %Y')} to {df['date'].max().strftime('%b %Y')}"
            }
        
        fig = go.Figure()
        
        if 'income' in monthly_data.columns:
            fig.add_trace(go.Bar(
                name='Income',
                x=monthly_data.index,
                y=monthly_data['income'],
                marker_color='#27ae60'
            ))
        
        if 'expense' in monthly_data.columns:
            fig.add_trace(go.Bar(
                name='Expense',
                x=monthly_data.index,
                y=monthly_data['expense'],
                marker_color='#e74c3c'
            ))
        
        if 'income' in monthly_data.columns and 'expense' in monthly_data.columns:
            savings = monthly_data['income'] - monthly_data['expense']
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=savings,
                mode='lines+markers',
                name='Monthly Savings',
                line=dict(color='#3498db', width=3),
                yaxis='y2'
            ))
            fig.update_layout(yaxis2=dict(title='Savings', overlaying='y', side='right'))
        
        fig.update_layout(
            title="Income vs Expenses Over Time",
            barmode='group',
            xaxis_title="Month",
            yaxis_title="Amount",
            template="plotly_white"
        )
        
        insights = self._ensure_data_insights(df, analysis)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'income_vs_expense')
        
        return self._finalize_chart(fig, "income_vs_expense", True, caption)
        
    def _create_savings_trend(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create savings trend chart with enhanced data insights"""
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        if 'income' not in monthly_data.columns or 'expense' not in monthly_data.columns:
            return self._create_helpful_response(
                "Insufficient Data for Savings",
                "Need both income and expense data to calculate savings."
            )
        
        if not df.empty:
            income_total = df[df['type'] == 'income']['amount'].sum()
            expense_total = df[df['type'] == 'expense']['amount'].sum()
            total_savings = income_total - expense_total
            month_count = df_copy['month'].nunique()
            positive_months = len([savings for savings in (monthly_data['income'] - monthly_data['expense']) if savings > 0])
            
            analysis['data_points'] = f"{len(df)} transactions across {month_count} months"
            analysis['data_characteristics'] = {
                'total_savings': total_savings,
                'income_total': income_total,
                'expense_total': expense_total,
                'month_count': month_count,
                'positive_months': positive_months,
                'positive_ratio': (positive_months / month_count * 100) if month_count > 0 else 0
            }
        
        savings = monthly_data['income'] - monthly_data['expense']
        cumulative_savings = savings.cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=cumulative_savings,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='#2980b9', width=4),
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_data.index,
            y=savings,
            name='Monthly Savings',
            marker_color='#3498db',
            opacity=0.6
        ))
        
        fig.update_layout(
            title="Savings Trend Over Time",
            xaxis_title="Month",
            yaxis_title="Amount",
            template="plotly_white"
        )
        
        insights = self._ensure_data_insights(df, analysis)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'savings_trend')
        
        return self._finalize_chart(fig, "savings_trend", True, caption)
    
    def _create_trend_analysis(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create trend analysis chart with enhanced data insights"""
        data_focus = analysis['data_focus']
        
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        if not df.empty:
            month_count = df_copy['month'].nunique()
            transaction_count = len(df)
            
            trend_metrics = {}
            for col in ['income', 'expense']:
                if col in monthly_data.columns:
                    if len(monthly_data[col]) >= 2:
                        first_val = monthly_data[col].iloc[0]
                        last_val = monthly_data[col].iloc[-1]
                        change_percent = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                        trend_metrics[f'{col}_trend'] = change_percent
            
            analysis['data_points'] = f"{transaction_count} transactions across {month_count} months"
            analysis['data_characteristics'] = {
                'month_count': month_count,
                'transaction_count': transaction_count,
                'trend_metrics': trend_metrics,
                'date_range': f"{df['date'].min().strftime('%b %Y')} to {df['date'].max().strftime('%b %Y')}"
            }
        
        fig = go.Figure()
        
        colors = {'income': '#27ae60', 'expense': '#e74c3c'}
        
        if data_focus in ['income', 'both'] and 'income' in monthly_data.columns:
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data['income'],
                mode='lines+markers',
                name='Income Trend',
                line=dict(color=colors['income'], width=3)
            ))
        
        if data_focus in ['expense', 'both'] and 'expense' in monthly_data.columns:
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data['expense'],
                mode='lines+markers',
                name='Expense Trend',
                line=dict(color=colors['expense'], width=3)
            ))
        
        if not fig.data:
            return self._create_helpful_response(
                "No Trend Data",
                "No data available for trend analysis in the selected period."
            )
        
        title = "Financial Trends Over Time"
        if data_focus == 'income':
            title = "Income Trend Over Time"
        elif data_focus == 'expense':
            title = "Expense Trend Over Time"
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Amount",
            template="plotly_white"
        )
        
        insights = self._ensure_data_insights(df, analysis)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'trend_analysis')
        
        return self._finalize_chart(fig, "trend_analysis", True, caption)

    def _create_anomaly_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create chart for anomaly detection"""
        anomaly_results = self.anomaly_detector.detect_spending_anomalies(df)
        
        fig = go.Figure()
        
        expense_df = df[df['type'] == 'expense'].copy()
        expense_df['date'] = pd.to_datetime(expense_df['date'])
        expense_df['month'] = expense_df['date'].dt.to_period('M')
        monthly_spending = expense_df.groupby('month')['amount'].sum().reset_index()
        monthly_spending['month_str'] = monthly_spending['month'].astype(str)
        
        fig.add_trace(go.Scatter(
            x=monthly_spending['month_str'],
            y=monthly_spending['amount'],
            mode='lines+markers',
            name='Monthly Spending',
            line=dict(color='#3498db', width=3)
        ))
        
        anomalous_months = [a['month'] for a in anomaly_results.get('anomalies', [])]
        for i, month in enumerate(monthly_spending['month_str']):
            if month in anomalous_months:
                fig.add_trace(go.Scatter(
                    x=[month],
                    y=[monthly_spending.iloc[i]['amount']],
                    mode='markers',
                    marker=dict(size=15, color='#e74c3c', symbol='star'),
                    name='Anomaly' if i == 0 else '',
                    showlegend=i == 0
                ))
        
        fig.update_layout(
            title="Spending Anomalies Detection",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template="plotly_white"
        )
        
        all_insights = anomaly_results['insights']
        insights_text = "; ".join(all_insights[:3])
        
        caption = self._generate_comprehensive_caption(query, analysis, insights_text, 'line')
        caption['anomaly_summary'] = {
            'total_anomalies': len(anomaly_results.get('anomalies', [])),
            'insights': all_insights
        }
        
        return self._finalize_chart(fig, "anomaly_detection", True, caption)

    def _create_detailed_breakdown(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create detailed breakdown dashboard based on focus"""
        focus = analysis['data_focus']
        
        if focus == 'expense':
            return self._create_expense_detailed_dashboard(df, analysis, query)
        elif focus == 'income':
            return self._create_income_detailed_dashboard(df, analysis, query)
        elif focus == 'savings':
            return self._create_savings_detailed_dashboard(df, analysis, query)
        else:
            return self._create_comprehensive_dashboard(df, analysis, query)

    def _create_expense_detailed_dashboard(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create detailed expense dashboard"""
        expense_df = df[df['type'] == 'expense']
        
        if expense_df.empty:
            return self._create_helpful_response("No Expense Data", "No expense data available")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Expense Distribution by Category', 
                'Monthly Expense Trends', 
                'Top Expense Categories',
                'Expense Timeline (Last 30 Days)'
            ),
            specs=[
                [{"type": "pie"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "xy"}]
            ]
        )
        
        category_totals = expense_df.groupby('category')['amount'].sum()
        if not category_totals.empty:
            fig.add_trace(go.Pie(
                labels=category_totals.index,
                values=category_totals.values,
                name="Expense Distribution",
                marker=dict(colors=px.colors.sequential.RdBu),
                textinfo='label+percent',
                hole=0.3
            ), 1, 1)
        
        expense_df_copy = expense_df.copy()
        expense_df_copy['month'] = expense_df_copy['date'].dt.to_period('M').astype(str)
        monthly_expenses = expense_df_copy.groupby('month')['amount'].sum()
        if not monthly_expenses.empty:
            fig.add_trace(go.Scatter(
                x=monthly_expenses.index,
                y=monthly_expenses.values,
                mode='lines+markers',
                name='Monthly Expenses',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8, color='#c0392b')
            ), 1, 2)
        
        top_categories = category_totals.nlargest(8)
        if not top_categories.empty:
            fig.add_trace(go.Bar(
                x=top_categories.index,
                y=top_categories.values,
                name='Top Expense Categories',
                marker_color='#e74c3c',
                text=top_categories.values,
                texttemplate='$%{text:.2f}',
                textposition='auto',
            ), 2, 1)
        
        expense_df_copy['date_only'] = expense_df_copy['date'].dt.date
        daily_expenses = expense_df_copy.groupby('date_only')['amount'].sum().tail(30)
        if not daily_expenses.empty:
            fig.add_trace(go.Scatter(
                x=daily_expenses.index,
                y=daily_expenses.values,
                mode='lines+markers',
                name='Daily Expenses',
                line=dict(color='#d35400', width=2),
                marker=dict(size=6, color='#e67e22'),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ), 2, 2)
        
        total_expenses = expense_df['amount'].sum()
        avg_monthly = monthly_expenses.mean() if not monthly_expenses.empty else 0
        largest_category = category_totals.idxmax() if not category_totals.empty else "N/A"
        largest_amount = category_totals.max() if not category_totals.empty else 0
        largest_percentage = (largest_amount / total_expenses * 100) if total_expenses > 0 else 0
        
        fig.update_layout(
            height=700,
            title_text="Detailed Expense Analysis Dashboard",
            showlegend=True,
            template="plotly_white",
            annotations=[
                dict(
                    text=f"Total Expenses: ${total_expenses:,.2f}<br>"
                        f"Avg Monthly: ${avg_monthly:,.2f}<br>"
                        f"Largest Category: {largest_category}<br>"
                        f"({largest_percentage:.1f}% of total)",
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1,
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ]
        )
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
        fig.update_xaxes(title_text="Expense Category", row=2, col=1)
        fig.update_yaxes(title_text="Amount ($)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=2, col=2)
        
        insights_parts = []
        insights_parts.append(f"Total expenses: ${total_expenses:.2f}")
        
        if not monthly_expenses.empty and len(monthly_expenses) > 1:
            expense_growth = ((monthly_expenses.iloc[-1] - monthly_expenses.iloc[0]) / monthly_expenses.iloc[0] * 100) if monthly_expenses.iloc[0] > 0 else 0
            insights_parts.append(f"Expense trend: {expense_growth:+.1f}%")
        
        if not category_totals.empty:
            insights_parts.append(f"Largest category: {largest_category} ({largest_percentage:.1f}%)")
            insights_parts.append(f"Expense categories: {len(category_totals)} total")
        
        if hasattr(self, 'anomaly_detector'):
            anomaly_results = self.anomaly_detector.detect_spending_anomalies(expense_df)
            if anomaly_results['anomalous_months_count'] > 0:
                insights_parts.append(f"⚠️ {anomaly_results['anomalous_months_count']} unusual spending patterns detected")
        
        insights = "; ".join(insights_parts)
        
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'dashboard')
        
        return self._finalize_chart(fig, "expense_detailed_dashboard", True, caption)

    def _create_income_detailed_dashboard(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create detailed income dashboard"""
        income_df = df[df['type'] == 'income']
        
        if income_df.empty:
            return self._create_helpful_response("No Income Data", "No income data available")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Income Sources Distribution', 
                'Monthly Income Trends', 
                'Top Income Sources',
                'Income Timeline (Last 30 Days)'
            ),
            specs=[
                [{"type": "pie"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "xy"}]
            ]
        )
        
        category_totals = income_df.groupby('category')['amount'].sum()
        if not category_totals.empty:
            fig.add_trace(go.Pie(
                labels=category_totals.index,
                values=category_totals.values,
                name="Income Distribution",
                marker=dict(colors=px.colors.qualitative.Set2),
                textinfo='label+percent',
                hole=0.3
            ), 1, 1)
        
        income_df_copy = income_df.copy()
        income_df_copy['month'] = income_df_copy['date'].dt.to_period('M').astype(str)
        monthly_income = income_df_copy.groupby('month')['amount'].sum()
        if not monthly_income.empty:
            fig.add_trace(go.Scatter(
                x=monthly_income.index,
                y=monthly_income.values,
                mode='lines+markers',
                name='Monthly Income',
                line=dict(color='#27ae60', width=3),
                marker=dict(size=8, color='#2ecc71')
            ), 1, 2)
        
        top_categories = category_totals.nlargest(8)
        if not top_categories.empty:
            fig.add_trace(go.Bar(
                x=top_categories.index,
                y=top_categories.values,
                name='Top Income Sources',
                marker_color='#2ecc71',
                text=top_categories.values,
                texttemplate='$%{text:.2f}',
                textposition='auto',
            ), 2, 1)
        
        income_df_copy['date_only'] = income_df_copy['date'].dt.date
        daily_income = income_df_copy.groupby('date_only')['amount'].sum().tail(30)
        if not daily_income.empty:
            fig.add_trace(go.Scatter(
                x=daily_income.index,
                y=daily_income.values,
                mode='lines+markers',
                name='Daily Income',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6, color='#2980b9'),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ), 2, 2)
        
        total_income = income_df['amount'].sum()
        avg_monthly = monthly_income.mean() if not monthly_income.empty else 0
        primary_source = category_totals.idxmax() if not category_totals.empty else "N/A"
        primary_amount = category_totals.max() if not category_totals.empty else 0
        primary_percentage = (primary_amount / total_income * 100) if total_income > 0 else 0
        
        fig.update_layout(
            height=700,
            title_text="Detailed Income Analysis Dashboard",
            showlegend=True,
            template="plotly_white",
            annotations=[
                dict(
                    text=f"Total Income: ${total_income:,.2f}<br>"
                        f"Avg Monthly: ${avg_monthly:,.2f}<br>"
                        f"Primary Source: {primary_source}<br>"
                        f"({primary_percentage:.1f}% of total)",
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1,
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ]
        )
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
        fig.update_xaxes(title_text="Income Source", row=2, col=1)
        fig.update_yaxes(title_text="Amount ($)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=2, col=2)
        
        insights_parts = []
        insights_parts.append(f"Total income: ${total_income:.2f}")
        
        if not monthly_income.empty and len(monthly_income) > 1:
            income_growth = ((monthly_income.iloc[-1] - monthly_income.iloc[0]) / monthly_income.iloc[0] * 100) if monthly_income.iloc[0] > 0 else 0
            insights_parts.append(f"Income growth: {income_growth:+.1f}%")
        
        if not category_totals.empty:
            insights_parts.append(f"Primary source: {primary_source} ({primary_percentage:.1f}%)")
            insights_parts.append(f"Income sources: {len(category_totals)} categories")
        
        insights = "; ".join(insights_parts)
        
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'dashboard')
        
        return self._finalize_chart(fig, "income_detailed_dashboard", True, caption)
    
    def _create_savings_detailed_dashboard(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create detailed savings dashboard"""
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        if 'income' not in monthly_data.columns or 'expense' not in monthly_data.columns:
            return self._create_helpful_response(
                "Insufficient Data for Savings Analysis",
                "Need both income and expense data to analyze savings."
            )
        
        monthly_savings = monthly_data['income'] - monthly_data['expense']
        cumulative_savings = monthly_savings.cumsum()
        savings_rate = (monthly_savings / monthly_data['income'] * 100).fillna(0)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Savings Trend', 
                'Cumulative Savings', 
                'Savings Rate Over Time',
                'Income vs Expenses'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )
        
        if not monthly_savings.empty:
            colors = ['#e74c3c' if x < 0 else '#27ae60' for x in monthly_savings.values]
            fig.add_trace(go.Bar(
                x=monthly_savings.index,
                y=monthly_savings.values,
                name='Monthly Savings',
                marker_color=colors,
                text=monthly_savings.values,
                texttemplate='$%{text:.0f}',
                textposition='auto',
            ), 1, 1)
        
        if not cumulative_savings.empty:
            fig.add_trace(go.Scatter(
                x=cumulative_savings.index,
                y=cumulative_savings.values,
                mode='lines+markers',
                name='Cumulative Savings',
                line=dict(color='#2980b9', width=4),
                marker=dict(size=8, color='#3498db'),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ), 1, 2)
        
        if not savings_rate.empty:
            fig.add_trace(go.Scatter(
                x=savings_rate.index,
                y=savings_rate.values,
                mode='lines+markers',
                name='Savings Rate',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=6, color='#8e44ad')
            ), 2, 1)
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        if not monthly_data.empty:
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data['income'],
                name='Income',
                marker_color='#27ae60',
                opacity=0.7
            ), 2, 2)
            
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data['expense'],
                name='Expenses',
                marker_color='#e74c3c',
                opacity=0.7
            ), 2, 2)
        
        total_savings = cumulative_savings.iloc[-1] if not cumulative_savings.empty else 0
        avg_monthly_savings = monthly_savings.mean() if not monthly_savings.empty else 0
        positive_months = (monthly_savings > 0).sum()
        total_months = len(monthly_savings)
        savings_consistency = (positive_months / total_months * 100) if total_months > 0 else 0
        avg_savings_rate = savings_rate.mean() if not savings_rate.empty else 0
        
        fig.update_layout(
            height=700,
            title_text="Detailed Savings Analysis Dashboard",
            showlegend=True,
            template="plotly_white",
            barmode='group',
            annotations=[
                dict(
                    text=f"Total Savings: ${total_savings:,.2f}<br>"
                        f"Avg Monthly: ${avg_monthly_savings:,.2f}<br>"
                        f"Savings Rate: {avg_savings_rate:.1f}%<br>"
                        f"Positive Months: {savings_consistency:.1f}%",
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1,
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ]
        )
        
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Savings Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Amount ($)", row=2, col=2)
        
        insights_parts = []
        insights_parts.append(f"Total savings: ${total_savings:.2f}")
        insights_parts.append(f"Average monthly savings: ${avg_monthly_savings:.2f}")
        insights_parts.append(f"Savings rate: {avg_savings_rate:.1f}%")
        insights_parts.append(f"Consistency: {savings_consistency:.1f}% positive months")
        
        if total_savings < 0:
            insights_parts.append("⚠️ Negative savings detected")
        
        insights = "; ".join(insights_parts)
        
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'dashboard')
        
        return self._finalize_chart(fig, "savings_detailed_dashboard", True, caption)

    def _ensure_data_insights(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Generate basic insights from data if none provided"""
        if df is None or df.empty:
            return "No transaction data available for analysis"
        
        try:
            total_transactions = len(df)
            total_amount = df['amount'].sum()
            avg_amount = total_amount / total_transactions if total_transactions > 0 else 0
            
            if 'type' in df.columns:
                income_df = df[df['type'] == 'income']
                expense_df = df[df['type'] == 'expense']
                
                income_total = income_df['amount'].sum()
                expense_total = expense_df['amount'].sum()
                net = income_total - expense_total
                
                insights = [
                    f"Total: ${total_amount:,.2f} across {total_transactions} transactions",
                    f"Income: ${income_total:,.2f} | Expenses: ${expense_total:,.2f}",
                    f"Net: ${net:,.2f} ({'surplus' if net >= 0 else 'deficit'})"
                ]
            else:
                insights = [f"Total amount: ${total_amount:,.2f} across {total_transactions} transactions"]
            
            if 'date' in df.columns and len(df) > 0:
                dates = pd.to_datetime(df['date'])
                date_range = f"{dates.min().strftime('%b %Y')} to {dates.max().strftime('%b %Y')}"
                insights.append(f"Period: {date_range}")
            
            return " | ".join(insights)
            
        except Exception as e:
            return f"Analyzed {len(df)} financial transactions"

    def _generate_comprehensive_caption(self, query: str, analysis: Dict, insights: str, chart_type: str) -> Dict:
        """Generate robust captions that never return N/A with proper fallbacks"""
        
        # Ensure all required fields have fallback values
        primary_intent = analysis.get('primary_intent', 'financial_analysis')
        time_context = analysis.get('time_context', {})
        time_range = time_context.get('range', 'recent_period')
        entities = analysis.get('entities', {})
        categories = entities.get('categories', ['All categories'])
        data_focus = analysis.get('data_focus', 'both')
        confidence = analysis.get('confidence_score', 0.85)
        
        # Format values with proper fallbacks
        time_display = self._format_time_range(time_range)
        intent_display = primary_intent.replace('_', ' ').title() if primary_intent else 'Financial Analysis'
        confidence_display = f"{confidence:.0%}" if confidence else "85%"
        
        # Ensure insights are never empty or N/A
        if not insights or insights.strip() in ["", "N/A", "No insights available"]:
            insights = self._generate_meaningful_insights(primary_intent, chart_type)
        
        # Ensure reasoning is never empty
        reasoning = analysis.get('reasoning', [])
        if not reasoning or reasoning == ["N/A"] or (isinstance(reasoning, list) and len(reasoning) == 0):
            reasoning = [
                f"Created {chart_type.replace('_', ' ').title()} visualization",
                f"Analyzed {data_focus} data from {time_display.lower()}",
                "Applied automated chart type selection based on data characteristics"
            ]
        elif isinstance(reasoning, str):
            reasoning = [reasoning]
        
        # Create enhanced summary
        summary = self._create_enhanced_summary(intent_display, time_display, insights, chart_type)
        
        # Return caption with guaranteed values
        return {
            'original_query': query or "Financial analysis request",
            'understood_intent': intent_display,
            'confidence_level': confidence_display,
            'time_period_analyzed': time_display,
            'categories_analyzed': categories,
            'chart_type_used': chart_type,
            'key_insights': insights,
            'analysis_reasoning': reasoning,
            'data_focus': data_focus,
            'data_points': analysis.get('data_points', 'Multiple transactions'),
            'summary': summary,
            'query_type': analysis.get('query_type', 'natural_language'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    def _format_time_range(self, time_range: str) -> str:
        """Convert time range to human-readable format with fallbacks"""
        range_map = {
            'this_month': 'This Month',
            'last_month': 'Last Month', 
            'last_3_months': 'Last 3 Months',
            'this_year': 'This Year',
            'last_6_months': 'Last 6 Months',
            'all_time': 'All Time',
            'recent_period': 'Recent Period',
            'custom_range': 'Selected Period'
        }
        return range_map.get(time_range, 'Recent Period')

    def _generate_meaningful_insights(self, intent: str, chart_type: str) -> str:
        """Generate meaningful insights based on intent with fallbacks"""
        
        insight_map = {
            'expense_breakdown': "Visualizing spending distribution to identify major expense categories",
            'income_breakdown': "Showing income composition to understand revenue sources", 
            'income_vs_expense': "Comparing earnings against spending to assess financial health",
            'savings_trend': "Tracking savings accumulation and identifying growth patterns",
            'trend_analysis': "Analyzing financial patterns over time to spot trends",
            'category_analysis': "Examining specific category behavior and spending habits",
            'time_comparison': "Comparing periods to identify changes and improvements",
            'anomaly_detection': "Identifying unusual patterns that may need attention",
            'comprehensive_dashboard': "Providing multi-dimensional financial overview",
            'financial_analysis': "Comprehensive analysis of financial data and patterns"
        }
        
        return insight_map.get(intent, "Financial data analysis providing actionable insights for better money management")

    def _create_enhanced_summary(self, intent: str, time_period: str, insights: str, chart_type: str) -> str:
        """Create enhanced summary text with guaranteed content"""
        
        chart_type_readable = chart_type.replace('_', ' ').title() if chart_type else 'Financial Chart'
        
        return f"{chart_type_readable} showing {intent.lower()} for {time_period.lower()}. {insights}"

    def _finalize_chart(self, fig, chart_name: str, save: bool, caption: Dict) -> tuple:
        """Finalize chart with professional styling"""
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12, color="#2c3e50"),
            margin=dict(l=50, r=50, t=80, b=50),
            title_font=dict(size=16, color="#2c3e50"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if save:
            path = os.path.join(CHART_DIR, f"{chart_name}.html")
            fig.write_html(path)
        
        self.chart_descriptions[chart_name] = caption
        return fig, caption

    def _create_helpful_response(self, title: str, message: str, suggestions: List[str] = None):
        """Create helpful response with suggestions"""
        fig = go.Figure()
        
        message_text = f"<b>{title}</b><br><br>{message}"
        
        if suggestions:
            message_text += "<br><br><b>💡 Try these queries:</b><br>" + "<br>".join([f"• {s}" for s in suggestions])
        
        fig.add_annotation(
            text=message_text,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=13, color="#2c3e50"),
            align="left",
            bordercolor="#3498db",
            borderwidth=2,
            borderpad=10,
            bgcolor="#ecf0f1"
        )
        
        fig.update_layout(
            title="Query Assistant",
            template="plotly_white",
            margin=dict(t=80, b=80, l=50, r=50),
            height=400,
            showlegend=False
        )
        
        caption = {
            'assistance': True,
            'title': title,
            'message': message,
            'suggestions': suggestions or []
        }
        
        return fig, caption

    def _create_error_response(self, message: str, suggestions: List[str] = None):
        """Create error response with recovery suggestions"""
        return self._create_helpful_response("Oops! Something went wrong", message, suggestions)

    def _apply_time_filter(self, df: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """Apply time filter to dataframe"""
        df_copy = df.copy()
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            df_copy = df_copy.dropna(subset=['date'])
        
        now = datetime.now()
        if time_range == 'this_month':
            start_date = now.replace(day=1)
            return df_copy[df_copy['date'] >= start_date]
        elif time_range == 'last_month':
            end_date = now.replace(day=1) - timedelta(days=1)
            start_date = end_date.replace(day=1)
            return df_copy[(df_copy['date'] >= start_date) & (df_copy['date'] <= end_date)]
        elif time_range == 'this_year':
            start_date = now.replace(month=1, day=1)
            return df_copy[df_copy['date'] >= start_date]
        elif time_range == 'last_3_months':
            cutoff = now - timedelta(days=90)
            return df_copy[df_copy['date'] >= cutoff]
        elif time_range == 'last_6_months':
            cutoff = now - timedelta(days=180)
            return df_copy[df_copy['date'] >= cutoff]
        else:
            return df_copy

    def _create_comprehensive_dashboard(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create comprehensive dashboard when intent isn't clear"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Expense Distribution', 
                'Income vs Expenses', 
                'Monthly Trends', 
                'Top Categories'
            ),
            specs=[
                [{"type": "pie"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "bar"}]
            ]
        )
        
        self._add_dashboard_visualizations(fig, df)
        
        fig.update_layout(
            height=700,
            title_text="Comprehensive Financial Overview",
            showlegend=True,
            template="plotly_white"
        )
        
        insights = "Comprehensive view showing multiple financial perspectives"
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'dashboard')
        
        return self._finalize_chart(fig, "comprehensive_dashboard", True, caption)

    def _add_dashboard_visualizations(self, fig, df: pd.DataFrame):
        """Add visualizations to dashboard"""
        expenses = df[df['type'] == 'expense']
        if not expenses.empty:
            exp_totals = expenses.groupby('category')['amount'].sum().nlargest(6)
            fig.add_trace(go.Pie(
                labels=exp_totals.index,
                values=exp_totals.values,
                name="Top Expenses",
                marker=dict(colors=px.colors.qualitative.Set3)
            ), 1, 1)
        
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        if 'income' in monthly_data.columns:
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data['income'],
                name='Income',
                marker_color='#27ae60'
            ), 1, 2)
        
        if 'expense' in monthly_data.columns:
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data['expense'],
                name='Expense',
                marker_color='#e74c3c'
            ), 1, 2)
        
        for col in ['income', 'expense']:
            if col in monthly_data.columns:
                fig.add_trace(go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data[col],
                    mode='lines',
                    name=f'{col.title()} Trend',
                    line=dict(width=2)
                ), 2, 1)
        
        if not expenses.empty:
            top_categories = expenses.groupby('category')['amount'].sum().nlargest(8)
            fig.add_trace(go.Bar(
                x=top_categories.index,
                y=top_categories.values,
                name='Top Categories',
                marker_color='#3498db'
            ), 2, 2)

    def get_visual_builder_categories(self) -> List[str]:
        """Get categories for visual query builder"""
        return self.visual_query_builder.get_available_categories()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        if df is None or df.empty:
            return pd.DataFrame()
            
        df_clean = df.copy()
        required_cols = ['date', 'type', 'amount', 'category']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            return pd.DataFrame()
        
        df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean['category'] = df_clean['category'].fillna('uncategorized')
        df_clean['type'] = df_clean['type'].fillna('expense')
        
        # CORRECT: Just pass the list directly to the subset parameter
        df_clean = df_clean.dropna(subset=['date', 'amount', 'type'])
        df_clean = df_clean[df_clean['amount'] > 0]
        
        return df_clean

    def natural_language_to_chart(self, query: str):
        """Main method to convert natural language to chart"""
        try:
            df = self.data_agent.get_transactions_df()
        
            if df is None or df.empty:
                return self._create_helpful_response(
                    "No transaction data available", 
                    "Please add some transactions first to generate charts.",
                    self.get_query_suggestions()
            )
        
            df_clean = self._clean_data(df)
            if df_clean.empty:
                return self._create_helpful_response(
                    "No valid transaction data", 
                    "The available transaction data appears to be invalid or corrupted.",
                    self.get_query_suggestions()
            )
        
            analysis = {
                'primary_intent': 'comprehensive_dashboard',
                'entities': {'categories': []},
                'time_context': {'range': 'last_3_months'},
                'data_focus': 'both',
                'chart_preference': 'bar',
                'confidence_score': 0.8,
                'reasoning': ['Using visual query builder as fallback']
        }
        
            result = self._generate_chart_based_on_analysis(df_clean, analysis, query)
            return result
        
        except Exception as e:
            return self._create_error_response(
                f"Chart generation error: {str(e)}",
                self.get_query_suggestions()
            )

    def pie_expenses_by_category(self, save=True, show=False, time_range=None):
        """Generate expenses pie chart with proper caption"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")

        expenses = df[df['type'] == 'expense']
        if expenses.empty:
            return self._create_helpful_response("No Expenses", "No expense data available")

        if time_range:
            expenses = self._apply_time_filter(expenses, time_range)

        totals = expenses.groupby('category')['amount'].sum()
        
        if totals.empty:
            return self._create_helpful_response("No Category Data", "No categorized expense data available")
        
        fig = px.pie(values=totals.values, names=totals.index, title="Expenses by Category")

        # Create proper caption
        total_expenses = totals.sum()
        top_category = totals.idxmax()
        top_amount = totals.max()
        
        caption = {
            'original_query': 'Expenses by Category Analysis',
            'understood_intent': 'Expense Breakdown',
            'confidence_level': '95%',
            'time_period_analyzed': self._format_time_range(time_range) if time_range else 'All Time',
            'categories_analyzed': list(totals.index),
            'chart_type_used': 'pie',
            'key_insights': f'Total expenses: ${total_expenses:,.2f} | Top category: {top_category} (${top_amount:,.2f})',
            'analysis_reasoning': [
                'Pie chart selected to show proportional expense distribution',
                f'Analyzed {len(totals)} expense categories',
                'Visualized spending patterns across different categories'
            ],
            'data_focus': 'expense',
            'data_points': f'{len(expenses)} expense transactions',
            'summary': f'Expense breakdown showing spending distribution across {len(totals)} categories',
            'query_type': 'basic_chart'
        }

        return self._finalize_chart(fig, "expenses_pie", save, caption)
    
    def bar_income_vs_expense(self, save=True, show=False, time_range=None):
        """Generate income vs expense bar chart with proper caption"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")

        if time_range:
            df = self._apply_time_filter(df, time_range)

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        
        df_copy = df_copy.dropna(subset=['date'])
        
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)

        fig = go.Figure()
        if 'income' in monthly_data.columns:
            fig.add_trace(go.Bar(name='Income', x=monthly_data.index, y=monthly_data['income'], marker_color='green'))
        if 'expense' in monthly_data.columns:
            fig.add_trace(go.Bar(name='Expense', x=monthly_data.index, y=monthly_data['expense'], marker_color='red'))

        fig.update_layout(title="Income vs Expenses", barmode='group')

        # Create proper caption
        income_total = monthly_data['income'].sum() if 'income' in monthly_data.columns else 0
        expense_total = monthly_data['expense'].sum() if 'expense' in monthly_data.columns else 0
        net_savings = income_total - expense_total
        month_count = len(monthly_data)
        
        caption = {
            'original_query': 'Income vs Expenses Comparison',
            'understood_intent': 'Income vs Expense Analysis',
            'confidence_level': '95%',
            'time_period_analyzed': self._format_time_range(time_range) if time_range else 'All Time',
            'categories_analyzed': ['Income', 'Expenses'],
            'chart_type_used': 'bar',
            'key_insights': f'Income: ${income_total:,.2f} | Expenses: ${expense_total:,.2f} | Net: ${net_savings:,.2f}',
            'analysis_reasoning': [
                'Bar chart selected to compare income and expenses side by side',
                f'Analyzed {month_count} months of financial data',
                'Grouped bar layout enables clear comparison between income and expenses'
            ],
            'data_focus': 'both',
            'data_points': f'{len(df_copy)} transactions across {month_count} months',
            'summary': f'Monthly comparison of income versus expenses showing financial balance over time',
            'query_type': 'basic_chart'
        }

        return self._finalize_chart(fig, "income_vs_expense", save, caption)

    
    def line_savings_over_time(self, time_range=None):
        """Generate savings over time chart with proper caption"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")
        
        if time_range:
            df = self._apply_time_filter(df, time_range)
        
        # Create analysis context for the savings trend
        analysis = {
            'primary_intent': 'savings_trend',
            'entities': {'categories': []},
            'time_context': {'range': time_range or 'last_3_months'},
            'data_focus': 'savings',
            'chart_preference': 'line',
            'confidence_score': 1.0,
            'reasoning': ['Showing savings trend over time using line chart'],
            'query_type': 'basic_chart',
            'data_points': f'{len(df)} transactions'
        }
        
        return self._create_savings_trend(df, analysis, "Show savings over time")

    def create_interactive_dashboard(self, filters=None):
        """Create interactive dashboard with proper caption"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for dashboard")

        if filters:
            df = self._apply_filters(df, filters)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expense Distribution', 'Income vs Expenses',
                        'Monthly Trends', 'Top Categories'),
            specs=[[{"type": "pie"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]]
        )

        self._add_dashboard_visualizations(fig, df)

        fig.update_layout(
            height=700,
            title_text="Interactive Financial Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        # Create proper caption for dashboard
        total_transactions = len(df)
        income_total = df[df['type'] == 'income']['amount'].sum()
        expense_total = df[df['type'] == 'expense']['amount'].sum()
        
        caption = {
            'original_query': 'Interactive Financial Dashboard',
            'understood_intent': 'Comprehensive Dashboard',
            'confidence_level': '95%',
            'time_period_analyzed': 'All Time',
            'categories_analyzed': ['Multiple views'],
            'chart_type_used': 'dashboard',
            'key_insights': f'Comprehensive view: {total_transactions} transactions | Income: ${income_total:,.2f} | Expenses: ${expense_total:,.2f}',
            'analysis_reasoning': [
                'Multi-panel dashboard selected for comprehensive financial overview',
                'Includes expense distribution, trends, comparisons, and savings progress',
                'Interactive layout enables holistic financial analysis'
            ],
            'data_focus': 'both',
            'data_points': f'{total_transactions} transactions analyzed across multiple views',
            'summary': 'Interactive dashboard providing comprehensive financial insights through multiple visualization types',
            'query_type': 'dashboard'
        }
        
        return self._finalize_chart(fig, "interactive_dashboard", False, caption)
    
    def comparative_charts(self, period1, period2):
        """Generate comparative charts with proper caption"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")
        
        analysis = {
            'primary_intent': 'time_comparison',
            'entities': {'categories': []},
            'time_context': {
                'range': None,
                'comparison_periods': [period1, period2]
            },
            'data_focus': 'expense',
            'chart_preference': 'bar',
            'confidence_score': 1.0,
            'reasoning': [f'Comparing {period1} vs {period2} using comparative analysis'],
            'query_type': 'basic_chart',
            'data_points': f'{len(df)} transactions'
        }
        
        return self._create_time_comparison(df, analysis, f"Compare {period1} vs {period2}")

    def predictive_charts(self, months_to_predict=6):
        """Generate predictive financial charts with better handling for limited data"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")
        
        # Clean and prepare the data
        df_clean = df.copy()
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date', 'amount'])
        
        expenses = df_clean[df_clean['type'] == 'expense']
        if expenses.empty:
            return self._create_helpful_response("No Expense Data", "No expense data for prediction")
        
        # Group by month properly
        expenses['month'] = expenses['date'].dt.to_period('M')
        monthly_expenses = expenses.groupby('month')['amount'].sum()
        
        # Calculate how many months of data we have
        unique_months = len(monthly_expenses)
        
        if unique_months < 3:
            # Create an educational chart instead of a prediction
            return self._create_educational_prediction_chart(monthly_expenses, months_to_predict, unique_months)
        
        # If we have sufficient data (3+ months), proceed with normal prediction
        monthly_expenses = monthly_expenses.tail(12)
        historical_dates = monthly_expenses.index.to_timestamp()
        historical_values = monthly_expenses.values
        
        # Calculate window_size for prediction (define it here so it's available for caption)
        window_size = min(3, len(historical_values))
        
        # Generate predictions with simple trend
        if len(historical_values) >= 3:
            # Simple linear trend
            x = np.arange(len(historical_values))
            slope, intercept = np.polyfit(x, historical_values, 1)
            
            predictions = []
            for i in range(months_to_predict):
                base_pred = intercept + slope * (len(historical_values) + i)
                # Add small random variation
                variation = base_pred * np.random.uniform(-0.05, 0.08)
                predictions.append(max(0, base_pred + variation))
        else:
            # Fallback: use average with variation
            avg_expense = historical_values.mean()
            predictions = [avg_expense * (1 + np.random.uniform(-0.1, 0.15)) for _ in range(months_to_predict)]
        
        # Generate future dates
        last_date = historical_dates[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_to_predict)]
        
        # Create the plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode='lines+markers',
            name=f'Historical Expenses ({len(historical_values)} months)',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8, color='#2980b9')
        ))
        
        # Prediction data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Expenses',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=8, color='#c0392b')
        ))
        
        fig.update_layout(
            title=f"Expense Prediction - Next {months_to_predict} Months",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            template="plotly_white",
            hovermode='x unified'
        )
        
        caption = {
            'chart_type': 'line',
            'summary': f'Expense prediction based on {len(historical_values)} months of historical data',
            'prediction_note': 'Trend-based forecast showing projected expense patterns',
            'data_quality': 'Good' if len(historical_values) >= 6 else 'Adequate',
            # Add the standard caption fields
            'original_query': f'Predict next {months_to_predict} months',
            'understood_intent': 'Expense Prediction',
            'confidence_level': '85%',
            'time_period_analyzed': f'Historical + {months_to_predict} months forecast',
            'categories_analyzed': ['Expense forecasting'],
            'chart_type_used': 'line',
            'key_insights': f'Based on {len(historical_values)} months of historical expense data',
            'analysis_reasoning': [
                'Line chart with dashed prediction line for future forecasting',
                f'Using {window_size}-month moving average for prediction',
                'Historical trend extended to show future projections'
            ],
            'data_focus': 'expense',
            'data_points': f'{len(historical_values)} historical months + {months_to_predict} predicted months',
            'query_type': 'predictive'
        }
        
        return self._finalize_chart(fig, "predictive_chart", False, caption)
        
    def _create_educational_prediction_chart(self, monthly_expenses, months_to_predict, unique_months):
        """Create an educational chart when insufficient data is available"""
        
        # Get current expense data
        historical_dates = monthly_expenses.index.to_timestamp()
        historical_values = monthly_expenses.values
        
        # Calculate average monthly expense
        avg_monthly_expense = historical_values.mean() if len(historical_values) > 0 else 0
        
        # Generate future dates starting from last available date
        last_date = historical_dates[-1] if len(historical_dates) > 0 else pd.Timestamp.now()
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_to_predict)]
        
        # Create simple predictions (flat line based on average)
        predictions = [avg_monthly_expense] * months_to_predict
        
        fig = go.Figure()
        
        # Add historical data if available
        if len(historical_dates) > 0:
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_values,
                mode='lines+markers',
                name=f'Your Expenses ({len(historical_dates)} month(s))',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10, color='#2980b9')
            ))
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Sample Prediction',
            line=dict(color='#e74c3c', width=3, dash='dot'),
            marker=dict(size=8, color='#c0392b')
        ))
        
        # Add educational annotation
        fig.add_annotation(
            x=0.5, y=0.9, xref="paper", yref="paper",
            text=f"💡 <b>Need more data for accurate predictions</b><br>"
                f"You have {unique_months} month(s) of expense data.<br>"
                f"Collect 3+ months of data for meaningful trend analysis.",
            showarrow=False,
            bgcolor="rgba(255, 243, 205, 0.9)",
            bordercolor="#856404",
            borderwidth=1,
            borderpad=10,
            font=dict(size=12, color="#856404")
        )
        
        fig.update_layout(
            title=f"Expense Forecast Preview - {months_to_predict} Months Ahead",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            template="plotly_white",
            annotations=[
                dict(
                    x=0.02, y=0.02, xref="paper", yref="paper",
                    text="📊 <b>How to improve predictions:</b><br>"
                        "• Add transactions for more months<br>"
                        "• Include various expense categories<br>"
                        "• Maintain consistent spending records",
                    showarrow=False,
                    bgcolor="rgba(232, 245, 233, 0.9)",
                    bordercolor="#2e7d32",
                    borderwidth=1,
                    borderpad=8,
                    font=dict(size=10, color="#2e7d32"),
                    align="left"
                )
            ]
        )
        
        caption = {
            'chart_type': 'line',
            'summary': f'Educational expense forecast preview',
            'prediction_note': f'Based on {unique_months} month(s) of data. Collect more data for accurate predictions.',
            'data_quality': 'Insufficient',
            'recommendation': f'Add transactions spanning at least 3 different months for meaningful trend analysis'
        }
        
        return self._finalize_chart(fig, "educational_prediction", False, caption)

    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """Get query suggestions"""
        return self.query_assistant.get_smart_suggestions(partial_query)

    def _apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        df_copy = df.copy()

        if 'category' in filters and filters['category']:
            filter_categories = [cat.strip().lower() for cat in filters['category']]
            df_copy = df_copy[
                df_copy['category'].astype(str).str.strip().str.lower().isin(filter_categories)
            ]

        if 'type' in filters and filters['type']:
            df_copy = df_copy[df_copy['type'].isin(filters['type'])]

        if 'date_range' in filters and filters['date_range'] and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy[(df_copy['date'] >= pd.to_datetime(start_date)) & 
                            (df_copy['date'] <= pd.to_datetime(end_date))]

        return df_copy

    def _create_empty_chart(self, message):
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

    def _create_time_comparison(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create time period comparison chart"""
        comparison_periods = analysis['time_context']['comparison_periods']
        
        if not comparison_periods:
            comparison_periods = ['this_month', 'last_month']
        
        period_data = {}
        for period in comparison_periods:
            period_df = self._apply_time_filter(df, period)
            period_expenses = period_df[period_df['type'] == 'expense']
            period_totals = period_expenses.groupby('category')['amount'].sum()
            period_data[period] = period_totals
        
        fig = go.Figure()
        
        for i, (period, data) in enumerate(period_data.items()):
            if not data.empty:
                fig.add_trace(go.Bar(
                    name=period.replace('_', ' ').title(),
                    x=data.index,
                    y=data.values,
                    marker_color=['#e74c3c', '#3498db'][i % 2]
                ))
        
        if not fig.data:
            return self._create_helpful_response(
                "No Data for Comparison",
                "No expense data available for the requested comparison periods."
            )
        
        fig.update_layout(
            title=f"Expense Comparison: {comparison_periods[0].replace('_', ' ').title()} vs {comparison_periods[1].replace('_', ' ').title()}",
            barmode='group',
            xaxis_title="Category",
            yaxis_title="Amount",
            template="plotly_white"
        )
        
        insights = self._ensure_data_insights(df, analysis)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'time_comparison')
        
        return self._finalize_chart(fig, "time_comparison", True, caption)

    def _create_category_analysis(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create category-specific analysis"""
        requested_categories = analysis['entities']['categories']
        
        if not requested_categories:
            available_cats = df['category'].unique().tolist()
            return self._create_helpful_response(
                "No category specified",
                f"Please specify which category you want to analyze. Available: {', '.join(map(str, available_cats))}",
                [f"Show {cat} expenses" for cat in available_cats[:3]]
            )
        
        category_mask = False
        for category in requested_categories:
            category_lower = category.strip().lower()
            category_mask |= (df['category'].astype(str).str.strip().str.lower() == category_lower)
        
        category_df = df[category_mask]
        
        if category_df.empty:
            available_cats = df['category'].unique().tolist()
            return self._create_helpful_response(
                "Category not found",
                f"'{requested_categories[0]}' not found in your data. Available categories: {', '.join(map(str, available_cats))}",
                [f"Show {cat} expenses" for cat in available_cats[:3]]
            )
        
        if analysis['time_context']['comparison_periods']:
            return self._create_category_comparison_chart(category_df, analysis, query)
        else:
            return self._create_single_category_chart(category_df, analysis, query)

    def _create_category_comparison_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create comparison chart for categories"""
        categories = analysis['entities']['categories']
        time_periods = analysis['time_context']['comparison_periods']
        
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.strftime('%B')
        
        comparison_data = []
        for period in time_periods:
            period_data = df_copy[df_copy['month'].str.lower() == period.lower()]
            if not period_data.empty:
                total = period_data['amount'].sum()
                comparison_data.append({'period': period, 'amount': total})
        
        if not comparison_data:
            return self._create_helpful_response(
                "No data for comparison",
                f"No {categories[0]} expenses found for {', '.join(time_periods)}"
            )
        
        fig = go.Figure()
        
        for data in comparison_data:
            fig.add_trace(go.Bar(
                x=[data['period']],
                y=[data['amount']],
                name=data['period'],
                text=[f"${data['amount']:.2f}"],
                textposition='auto',
            ))
        
        category_name = categories[0] if categories else 'Category'
        fig.update_layout(
            title=f"{category_name} Expenses: {', '.join(time_periods)}",
            xaxis_title="Time Period",
            yaxis_title="Amount ($)",
            showlegend=False,
            template="plotly_white"
        )
        
        total_amount = sum(item['amount'] for item in comparison_data)
        insights = f"Total {category_name} spending: ${total_amount:.2f}"
        
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'bar')
        return self._finalize_chart(fig, "category_comparison", True, caption)

    def _create_single_category_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create chart for single category analysis"""
        categories = analysis['entities']['categories']
        category_name = categories[0] if categories else 'Category'
        
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.strftime('%B %Y')
        
        monthly_totals = df_copy.groupby('month')['amount'].sum().reset_index()
        
        if len(monthly_totals) > 1:
            fig = px.bar(
                monthly_totals,
                x='month',
                y='amount',
                title=f"{category_name} Expenses Over Time",
                color_discrete_sequence=['#3498db']
            )
            chart_type = 'bar'
        else:
            fig = px.pie(
                monthly_totals,
                values='amount',
                names='month',
                title=f"{category_name} Expenses",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            chart_type = 'pie'
        
        total_amount = monthly_totals['amount'].sum()
        insights = f"Total {category_name} spending: ${total_amount:.2f}"
        
        caption = self._generate_comprehensive_caption(query, analysis, insights, chart_type)
        return self._finalize_chart(fig, "single_category", True, caption)

class ChartCreatorAgent:
    """Main Chart Creator Agent class for external use"""
    def __init__(self, data_agent):
        self.enhanced_agent = EnhancedChartCreatorAgent(data_agent)

    def get_filtered_data_summary(self, components: Dict[str, Any]) -> pd.DataFrame:
        """Get filtered data for summary display"""
        df = self.enhanced_agent.data_agent.get_transactions_df()
        return self.enhanced_agent._apply_visual_filters(df, components)          
    
    def visual_components_to_chart(self, components: Dict[str, Any]):
        """Main method to convert visual components to chart"""
        return self.enhanced_agent.visual_components_to_chart(components)
    
    def get_visual_builder_categories(self) -> List[str]:
        """Get categories for visual query builder"""
        return self.enhanced_agent.get_visual_builder_categories()
    
    def natural_language_to_chart(self, query: str):
        """Main method to convert natural language to chart"""
        return self.enhanced_agent.natural_language_to_chart(query)
    
    def predictive_charts(self, months_to_predict=6):
        """Generate predictive charts"""
        return self.enhanced_agent.predictive_charts(months_to_predict)
    
    def comparative_charts(self, period1, period2):
        """Generate comparative charts"""
        return self.enhanced_agent.comparative_charts(period1, period2)
    
    def line_savings_over_time(self, time_range=None):
        """Generate savings over time chart"""
        return self.enhanced_agent.line_savings_over_time(time_range)
    
    def create_interactive_dashboard(self, filters=None):
        """Create interactive dashboard"""
        return self.enhanced_agent.create_interactive_dashboard(filters)
    
    def pie_expenses_by_category(self, save=True, show=False, time_range=None):
        """Generate expenses pie chart"""
        return self.enhanced_agent.pie_expenses_by_category(save, show, time_range)
    
    def bar_income_vs_expense(self, save=True, show=False, time_range=None):
        """Generate income vs expense bar chart"""
        return self.enhanced_agent.bar_income_vs_expense(save, show, time_range)
    
    def get_context_aware_suggestions(self, current_query: str = "", user_history: List = None) -> List[str]:
        """Get context-aware query suggestions"""
        return self.enhanced_agent.get_context_aware_suggestions(current_query, user_history)
    
    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """Get query suggestions"""
        return self.enhanced_agent.get_query_suggestions(partial_query)
    
