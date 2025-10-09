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


from sklearn.ensemble import IsolationForest
import pickle

CHART_DIR = "visualizations/static_charts"

class AnomalyDetector:
    """Detect unusual spending patterns and anomalies"""
    
    def __init__(self):
        self.anomaly_model = None
        self.anomaly_threshold = 0.1  # 10% of data as anomalies
        
    def detect_spending_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect unusual spending patterns using Isolation Forest"""
        try:
            if df is None or df.empty:
                return {'anomalies': [], 'insights': "No data available for anomaly detection"}
            
            # Prepare data for anomaly detection
            expense_df = df[df['type'] == 'expense'].copy()
            if expense_df.empty:
                return {'anomalies': [], 'insights': "No expense data available"}
            
            # Create monthly spending data
            expense_df['date'] = pd.to_datetime(expense_df['date'])
            expense_df['month'] = expense_df['date'].dt.to_period('M')
            monthly_spending = expense_df.groupby('month')['amount'].sum().reset_index()
            
            if len(monthly_spending) < 3:
                return {'anomalies': [], 'insights': "Insufficient data for anomaly detection (need at least 3 months)"}
            
            # Prepare features for anomaly detection
            X = monthly_spending['amount'].values.reshape(-1, 1)
            
            # Train Isolation Forest
            self.anomaly_model = IsolationForest(
                contamination=self.anomaly_threshold,
                random_state=42
            )
            anomalies = self.anomaly_model.fit_predict(X)
            
            # Identify anomalous months
            monthly_spending['is_anomaly'] = anomalies == -1
            anomalous_months = monthly_spending[monthly_spending['is_anomaly']]
            
            # Generate insights
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
            print(f"⚠️ Anomaly detection error: {e}")
            return {'anomalies': [], 'insights': [f"Anomaly detection failed: {str(e)}"]}
    
    def detect_category_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in category spending"""
        try:
            expense_df = df[df['type'] == 'expense'].copy()
            if expense_df.empty:
                return {'category_anomalies': [], 'insights': "No expense data available"}
            
            # Calculate category spending patterns
            category_monthly = expense_df.groupby(['category', expense_df['date'].dt.to_period('M')])['amount'].sum().unstack(fill_value=0)
            
            anomalies = []
            for category in category_monthly.index:
                spending = category_monthly.loc[category].values
                if len(spending) >= 3 and np.std(spending) > 0:
                    # Use z-score for category anomalies
                    z_scores = np.abs((spending - np.mean(spending)) / np.std(spending))
                    high_z_indices = np.where(z_scores > 2.5)[0]
                    
                    for idx in high_z_indices:
                        month = category_monthly.columns[idx]
                        amount = spending[idx]
                        avg_amount = np.mean(spending)
                        deviation = ((amount - avg_amount) / avg_amount) * 100
                        
                        if abs(deviation) > 50:  # Only flag significant deviations
                            anomalies.append({
                                'category': category,
                                'month': str(month),
                                'amount': amount,
                                'deviation_percent': deviation,
                                'type': 'high' if deviation > 0 else 'low'
                            })
            
            # Generate insights
            insights = []
            for anomaly in anomalies[:3]:  # Top 3 anomalies
                trend = "spike" if anomaly['type'] == 'high' else "drop"
                insights.append(
                    f"{anomaly['category']} had a {trend} in {anomaly['month']}: "
                    f"${anomaly['amount']:.2f} ({anomaly['deviation_percent']:+.1f}%)"
                )
            
            return {
                'category_anomalies': anomalies,
                'insights': insights if insights else ["No significant category anomalies detected"]
            }
            
        except Exception as e:
            print(f"⚠️ Category anomaly detection error: {e}")
            return {'category_anomalies': [], 'insights': [f"Category anomaly detection failed: {str(e)}"]}

class ChartRecommender:
    """Recommend optimal chart types based on data characteristics and user preferences"""
    
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
    
    def record_chart_performance(self, chart_type: str, user_interaction: str, user_id: str = "default"):
        """Record how users interact with different chart types"""
        if user_interaction in ['click', 'hover', 'zoom', 'save']:
            self.chart_performance[user_id][chart_type] += 1
    
    def get_user_preferences(self, user_id: str = "default") -> Dict[str, int]:
        """Get user's chart preferences based on interaction history"""
        user_perf = self.chart_performance[user_id]
        total_interactions = sum(user_perf.values())
        
        if total_interactions == 0:
            return {}
        
        preferences = {}
        for chart_type, count in user_perf.items():
            preferences[chart_type] = int((count / total_interactions) * 10)  # Scale to 1-10
        
        self.user_preferences[user_id] = preferences
        return preferences

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
            
            return suggestions[:8]  # Return top 8 suggestions
            
        except Exception as e:
            print(f"⚠️ Query suggestions error: {e}")
            # Return basic suggestions if data access fails
            return self.popular_queries[:6]

class VisualQueryBuilder:
    """Enhanced Visual Query Builder for creating charts through UI components"""
    
    def __init__(self, data_agent):
        self.data_agent = data_agent
        
    def build_query_from_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Build a query dictionary from visual components"""
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
            
            # Specific category
            specific_category = components.get('specific_category', 'All Categories')
            if specific_category != "All Categories":
                query_parts.append(f"for {specific_category.lower()}")
            
            # Amount filter
            min_amount = components.get('min_amount', 0)
            if min_amount > 0:
                query_parts.append(f"over ${min_amount}")
            
            # Build the final query
            generated_query = " ".join(query_parts)
            
            # Create analysis structure similar to NLP output
            analysis = {
                'primary_intent': self._map_focus_to_intent(chart_focus),
                'entities': {
                    'categories': [specific_category] if specific_category != "All Categories" else [],
                    'time_periods': [time_period] if time_period != "All Time" else []
                },
                'time_context': {
                    'range': self._map_time_period(time_period),
                    'comparison_periods': []
                },
                'data_focus': self._map_focus_to_data(chart_focus),
                'chart_preference': self._map_detail_to_chart(detail_level, chart_focus),
                'confidence_score': 1.0,  # Visual queries have highest confidence
                'reasoning': ['Query built from visual components'],
                'is_follow_up': False
            }
            
            return {
                'query': generated_query,
                'analysis': analysis,
                'components': components
            }
            
        except Exception as e:
            print(f"⚠️ Visual query building error: {e}")
            return {
                'query': "Show financial overview",
                'analysis': {
                    'primary_intent': 'comprehensive_dashboard',
                    'confidence_score': 0.5
                },
                'components': components
            }
    
    def _map_focus_to_intent(self, focus: str) -> str:
        """Map visual focus to NLP intent"""
        focus_map = {
            'Expenses': 'expense_breakdown',
            'Income': 'income_breakdown',
            'Savings': 'savings_trend',
            'Comparison': 'income_vs_expense'
        }
        return focus_map.get(focus, 'comprehensive_dashboard')
    
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
        """Map detail level to chart type"""
        if detail == "By Category":
            return 'pie' if focus in ['Expenses', 'Income'] else 'bar'
        elif detail == "Trend Over Time":
            return 'line'
        elif detail == "Detailed Breakdown":
            return 'bar'
        else:  # Summary
            return 'bar' if focus == 'Comparison' else 'pie'
    
    def get_available_categories(self) -> List[str]:
        """Get available categories from data"""
        try:
            df = self.data_agent.get_transactions_df()
            if df is not None and 'category' in df.columns:
                return sorted(df['category'].unique().tolist())
            return []
        except:
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

class EnhancedChartCreatorAgent:
    def __init__(self, data_agent):
        self.data_agent = data_agent
        self.visual_query_builder = VisualQueryBuilder(data_agent)
        self.anomaly_detector = AnomalyDetector()
        self.chart_recommender = ChartRecommender()
        self.performance_cache = PerformanceCache()
        self.query_assistant = QueryAssistant(data_agent)
        os.makedirs(CHART_DIR, exist_ok=True)
        self.chart_descriptions = {}
    
    def visual_components_to_chart(self, components: Dict[str, Any]):
        """Generate chart from visual query builder components"""
        try:
            print(f"🎨 Processing visual query: {components}")
            df = self.data_agent.get_transactions_df()
            
            if df is None or df.empty:
                return self._create_helpful_response(
                    "No transaction data available", 
                    "Please add some transactions first to generate charts.",
                    self.get_query_suggestions()
                )
            
            # Validate components
            is_valid, validation_msg = self.visual_query_builder.validate_components(components)
            if not is_valid:
                return self._create_helpful_response(
                    "Invalid Query Components",
                    validation_msg,
                    self.get_query_suggestions()
                )
            
            # Build query from components
            query_result = self.visual_query_builder.build_query_from_components(components)
            generated_query = query_result['query']
            analysis = query_result['analysis']
            
            print(f"✅ Visual query built: {generated_query}")
            print(f"✅ Analysis: {analysis['primary_intent']}")
            
            df_clean = self._clean_data(df)
            if df_clean.empty:
                return self._create_helpful_response(
                    "No valid transaction data", 
                    "The available transaction data appears to be invalid or corrupted.",
                    self.get_query_suggestions()
                )
            
            # Generate the chart based on analysis
            result = self._generate_chart_based_on_analysis(df_clean, analysis, generated_query)
            
            return result
            
        except Exception as e:
            print(f"❌ Visual chart generation error: {str(e)}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            
            return self._create_error_response(
                f"Chart generation error: {str(e)}",
                self.get_query_suggestions()
            )

    def _generate_chart_based_on_analysis(self, df: pd.DataFrame, analysis: Dict, original_query: str):
        """Generate chart based on analysis (shared with NLP)"""
        # Apply time filter
        filtered_df = self._apply_time_filter(df, analysis['time_context']['range'])
        
        if filtered_df.empty:
            return self._create_helpful_response(
                "No data for selected period",
                f"No transactions found for {analysis['time_context']['range']}.",
                self.get_query_suggestions()
            )
        
        # Add anomaly insights to regular charts if available
        insights = ""
        if analysis.get('primary_intent') in ['trend_analysis', 'comprehensive_dashboard']:
            anomaly_results = self.anomaly_detector.detect_spending_anomalies(filtered_df)
            if anomaly_results['insights'] and len(anomaly_results['anomalies']) > 0:
                insights = "Anomaly detected: " + "; ".join(anomaly_results['insights'][:2])
        
        # Route to appropriate chart handler based on intent
        intent_handlers = {
            'income_breakdown': self._create_income_breakdown,
            'expense_breakdown': self._create_expense_breakdown,
            'income_vs_expense': self._create_comparison_chart,
            'savings_trend': self._create_savings_trend,
            'category_analysis': self._create_category_analysis,
            'time_comparison': self._create_time_comparison,
            'trend_analysis': self._create_trend_analysis,
            'anomaly_detection': self._create_anomaly_chart
        }
        
        handler = intent_handlers.get(analysis['primary_intent'], self._create_comprehensive_dashboard)
        result = handler(filtered_df, analysis, original_query)
        
        # Add enhanced features to the result
        if isinstance(result, tuple) and len(result) > 1:
            chart, caption = result
            if insights:
                caption['enhanced_insights'] = insights
            caption['chart_recommendation'] = "AI-recommended based on data characteristics"
            return chart, caption
        
        return result

    # [Keep all the existing chart creation methods from your original code]
    # _create_income_breakdown, _create_expense_breakdown, _create_comparison_chart, etc.
    # ... (all your existing chart creation methods remain the same)

    def _create_income_breakdown(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create income breakdown chart"""
        income_df = df[df['type'] == 'income']
        
        if income_df.empty:
            return self._create_helpful_response(
                "No Income Data",
                "No income transactions found in the selected period."
            )
        
        # Filter by specific categories if requested
        requested_categories = analysis['entities']['categories']
        if requested_categories:
            income_df = income_df[income_df['category'].isin(requested_categories)]
        
        category_totals = income_df.groupby('category')['amount'].sum()
        
        if len(category_totals) <= 1:
            # Use bar chart for single category
            fig = px.bar(
                x=category_totals.index,
                y=category_totals.values,
                title="Income by Category",
                color_discrete_sequence=['#2ecc71']
            )
            chart_type = 'bar'
        else:
            # Use pie chart for multiple categories
            fig = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                title="Income Breakdown by Category",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            chart_type = 'pie'
        
        insights = self._generate_income_insights(category_totals)
        caption = self._generate_comprehensive_caption(query, analysis, insights, chart_type)
        
        return self._finalize_chart(fig, "income_breakdown", True, caption)
    
    def _create_expense_breakdown(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create expense breakdown chart"""
        expense_df = df[df['type'] == 'expense']
        
        if expense_df.empty:
            return self._create_helpful_response(
                "No Expense Data",
                "No expense transactions found in the selected period."
            )
        
        # Filter by specific categories if requested
        requested_categories = analysis['entities']['categories']
        if requested_categories:
            expense_df = expense_df[expense_df['category'].isin(requested_categories)]
        
        category_totals = expense_df.groupby('category')['amount'].sum()
        
        if len(category_totals) <= 1:
            # Use bar chart for single category
            fig = px.bar(
                x=category_totals.index,
                y=category_totals.values,
                title="Expenses by Category",
                color_discrete_sequence=['#e74c3c']
            )
            chart_type = 'bar'
        else:
            # Use pie chart for multiple categories
            fig = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                title="Expense Breakdown by Category",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            chart_type = 'pie'
        
        insights = self._generate_expense_insights(category_totals)
        caption = self._generate_comprehensive_caption(query, analysis, insights, chart_type)
        
        return self._finalize_chart(fig, "expense_breakdown", True, caption)
    
    def _create_comparison_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create income vs expense comparison"""
        # Prepare monthly data
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure()
        
        # Add income and expense bars
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
        
        # Add savings line
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
        
        insights = self._generate_comparison_insights(monthly_data)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'bar')
        
        return self._finalize_chart(fig, "income_vs_expense", True, caption)
    
    def _create_savings_trend(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create savings trend chart"""
        # Prepare monthly data
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
        
        insights = self._generate_savings_insights(savings, cumulative_savings)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'line')
        
        return self._finalize_chart(fig, "savings_trend", True, caption)

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
        
        # Filter for requested categories
        category_df = df[df['category'].isin(requested_categories)]
        
        if category_df.empty:
            available_cats = df['category'].unique().tolist()
            return self._create_helpful_response(
                "Category not found",
                f"'{requested_categories[0]}' not found in your data. Available categories: {', '.join(map(str, available_cats))}",
                [f"Show {cat} expenses" for cat in available_cats[:3]]
            )
        
        # Create appropriate visualization
        if analysis['time_context']['comparison_periods']:
            return self._create_category_comparison_chart(category_df, analysis, query)
        else:
            return self._create_single_category_chart(category_df, analysis, query)
    
    def _create_time_comparison(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create time period comparison chart"""
        comparison_periods = analysis['time_context']['comparison_periods']
        
        if not comparison_periods:
            # Default to this month vs last month
            comparison_periods = ['this_month', 'last_month']
        
        # Get data for each period
        period_data = {}
        for period in comparison_periods:
            period_df = self._apply_time_filter(df, period)
            period_expenses = period_df[period_df['type'] == 'expense']
            period_totals = period_expenses.groupby('category')['amount'].sum()
            period_data[period] = period_totals
        
        # Create comparison chart
        fig = go.Figure()
        
        for i, (period, data) in enumerate(period_data.items()):
            if not data.empty:
                fig.add_trace(go.Bar(
                    name=period.replace('_', ' ').title(),
                    x=data.index,
                    y=data.values,
                    marker_color=['#e74c3c', '#3498db'][i % 2]
                ))
        
        if not fig.data:  # No data added
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
        
        insights = self._generate_comparison_insights_time(period_data)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'bar')
        
        return self._finalize_chart(fig, "time_comparison", True, caption)

    def _create_trend_analysis(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create trend analysis chart"""
        data_focus = analysis['data_focus']
        
        # Prepare monthly data
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly_data = df_copy.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
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
        
        insights = self._generate_trend_insights(monthly_data, data_focus)
        caption = self._generate_comprehensive_caption(query, analysis, insights, 'line')
        
        return self._finalize_chart(fig, "trend_analysis", True, caption)
    
    def _create_anomaly_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create chart for anomaly detection"""
        anomaly_results = self.anomaly_detector.detect_spending_anomalies(df)
        category_anomalies = self.anomaly_detector.detect_category_anomalies(df)
        
        # Create anomaly visualization
        fig = go.Figure()
        
        # Add monthly spending line
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
        
        # Highlight anomalies
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
        
        # Combine insights
        all_insights = anomaly_results['insights'] + category_anomalies['insights']
        insights_text = "; ".join(all_insights[:3])  # Show top 3 insights
        
        caption = self._generate_comprehensive_caption(query, analysis, insights_text, 'line')
        caption['anomaly_summary'] = {
            'total_anomalies': len(anomaly_results.get('anomalies', [])),
            'category_anomalies': len(category_anomalies.get('category_anomalies', [])),
            'insights': all_insights
        }
        
        return self._finalize_chart(fig, "anomaly_detection", True, caption)
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
        
        # Add various visualizations
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
    def _create_category_comparison_chart(self, df: pd.DataFrame, analysis: Dict, query: str):
        """Create comparison chart for categories"""
        categories = analysis['entities']['categories']
        time_periods = analysis['time_context']['comparison_periods']
        
        # Prepare comparison data
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
        
        # Create comparison chart
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
        
        # Create monthly breakdown
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.strftime('%B %Y')
        
        monthly_totals = df_copy.groupby('month')['amount'].sum().reset_index()
        
        if len(monthly_totals) > 1:
            # Use bar chart for multiple months
            fig = px.bar(
                monthly_totals,
                x='month',
                y='amount',
                title=f"{category_name} Expenses Over Time",
                color_discrete_sequence=['#3498db']
            )
            chart_type = 'bar'
        else:
            # Use pie chart for single period
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

    def _add_dashboard_visualizations(self, fig, df: pd.DataFrame):
        """Add visualizations to dashboard"""
        # Expense pie chart
        expenses = df[df['type'] == 'expense']
        if not expenses.empty:
            exp_totals = expenses.groupby('category')['amount'].sum().nlargest(6)
            fig.add_trace(go.Pie(
                labels=exp_totals.index,
                values=exp_totals.values,
                name="Top Expenses",
                marker=dict(colors=px.colors.qualitative.Set3)
            ), 1, 1)
        
        # Monthly income vs expense
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
        
        # Monthly trends
        for col in ['income', 'expense']:
            if col in monthly_data.columns:
                fig.add_trace(go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data[col],
                    mode='lines',
                    name=f'{col.title()} Trend',
                    line=dict(width=2)
                ), 2, 1)
        
        # Top categories bar chart
        if not expenses.empty:
            top_categories = expenses.groupby('category')['amount'].sum().nlargest(8)
            fig.add_trace(go.Bar(
                x=top_categories.index,
                y=top_categories.values,
                name='Top Categories',
                marker_color='#3498db'
            ), 2, 2)

    # [Include all other chart creation methods...]
    # _create_expense_breakdown, _create_comparison_chart, _create_savings_trend, etc.

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
        
        df_clean = df_clean.dropna(subset=['date', 'amount', 'type'])
        df_clean = df_clean[df_clean['amount'] > 0]
        
        return df_clean

    def _apply_time_filter(self, df: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """Apply time filter to dataframe"""
        df_copy = df.copy()
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

    def _generate_comprehensive_caption(self, query: str, analysis: Dict, insights: str, chart_type: str) -> Dict:
        """Generate comprehensive caption"""
        # Safely get reasoning with default value
        reasoning = analysis.get('reasoning', ['Analysis completed'])
        
        return {
            'original_query': query,
            'understood_intent': analysis['primary_intent'],
            'confidence_level': f"{analysis['confidence_score']:.0%}",
            'time_period_analyzed': analysis['time_context']['range'],
            'categories_analyzed': analysis['entities']['categories'],
            'chart_type_used': chart_type,
            'key_insights': insights,
            'analysis_reasoning': reasoning,  # Use the safe get
            'summary': f"Showing {chart_type} chart for {analysis['data_focus']} data from {analysis['time_context']['range']}. {insights}",
            'query_type': 'visual_builder'
        }

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
            print(f"💾 Chart saved: {path}")
        
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

    # Insight generation methods
    def _generate_income_insights(self, category_totals) -> str:
        if len(category_totals) == 0:
            return "No income data available for insights"
        
        top_source = category_totals.idxmax()
        top_amount = category_totals.max()
        total_income = category_totals.sum()
        percentage = (top_amount / total_income) * 100
        
        return f"Primary income source: {top_source} (${top_amount:.2f}, {percentage:.1f}% of total)"

    def _generate_expense_insights(self, category_totals) -> str:
        if len(category_totals) == 0:
            return "No expense data available for insights"
        
        top_category = category_totals.idxmax()
        top_amount = category_totals.max()
        total_expenses = category_totals.sum()
        percentage = (top_amount / total_expenses) * 100
        
        return f"Largest expense: {top_category} (${top_amount:.2f}, {percentage:.1f}% of total)"

    def _generate_comparison_insights(self, monthly_data) -> str:
        insights = []
        
        if 'income' in monthly_data.columns and 'expense' in monthly_data.columns:
            total_income = monthly_data['income'].sum()
            total_expense = monthly_data['expense'].sum()
            net_savings = total_income - total_expense
            savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
            
            insights.append(f"Net savings: ${net_savings:.2f} ({savings_rate:.1f}% of income)")
            
            if len(monthly_data) > 1:
                income_growth = ((monthly_data['income'].iloc[-1] - monthly_data['income'].iloc[0]) / 
                               monthly_data['income'].iloc[0] * 100) if monthly_data['income'].iloc[0] > 0 else 0
                insights.append(f"Income growth: {income_growth:+.1f}%")
        
        return "; ".join(insights) if insights else "Comparison insights available"

    def _generate_savings_insights(self, monthly_savings, cumulative_savings) -> str:
        if len(monthly_savings) == 0:
            return "No savings data available"
        
        total_savings = cumulative_savings.iloc[-1] if len(cumulative_savings) > 0 else 0
        avg_monthly = monthly_savings.mean()
        positive_months = (monthly_savings > 0).sum()
        savings_ratio = (positive_months / len(monthly_savings)) * 100
        
        return f"Total savings: ${total_savings:.2f}; Average monthly: ${avg_monthly:.2f}; Positive months: {savings_ratio:.1f}%"

    def _generate_comparison_insights_time(self, period_data) -> str:
        if len(period_data) < 2:
            return "Insufficient data for comparison"
        
        periods = list(period_data.keys())
        data1 = period_data[periods[0]]
        data2 = period_data[periods[1]]
        
        total1 = data1.sum()
        total2 = data2.sum()
        change = ((total2 - total1) / total1 * 100) if total1 > 0 else 0
        
        return f"Total change: {change:+.1f}% ({periods[0]} to {periods[1]})"

    def _generate_trend_insights(self, monthly_data, data_focus) -> str:
        insights = []
        
        for col in ['income', 'expense']:
            if col in monthly_data.columns and (data_focus in [col, 'both']):
                if len(monthly_data) > 1:
                    growth = ((monthly_data[col].iloc[-1] - monthly_data[col].iloc[0]) / 
                            monthly_data[col].iloc[0] * 100) if monthly_data[col].iloc[0] > 0 else 0
                    insights.append(f"{col.title()} growth: {growth:+.1f}%")
        
        return "; ".join(insights) if insights else "Trend analysis completed"

    def _generate_category_insights(self, category_df, category_name) -> str:
        if category_df.empty:
            return f"No data available for {category_name}"
        
        total_spent = category_df['amount'].sum()
        avg_per_month = category_df.groupby(category_df['date'].dt.to_period('M'))['amount'].sum().mean()
        
        return f"Total spent on {category_name}: ${total_spent:.2f}; Monthly average: ${avg_per_month:.2f}"
    # [Include all other existing methods...]
    # predictive_charts, comparative_charts, line_savings_over_time, etc.

    def _get_data_time_range(self, df: pd.DataFrame) -> str:
        """Get data time range for context"""
        if df is None or df.empty or 'date' not in df.columns:
            return "No date data"
        
        try:
            dates = pd.to_datetime(df['date']).dropna()
            if len(dates) == 0:
                return "No valid dates"
            return f"{dates.min().strftime('%b %Y')} to {dates.max().strftime('%b %Y')}"
        except:
            return "Unknown time range"
        
    def natural_language_to_chart(self, query: str):
        """Main method to convert natural language to chart"""
        try:
            print(f"🧠 Processing with Gemini NLP: '{query}'")
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
        
        # For now, create a basic analysis structure since Gemini is removed
            analysis = {
                'primary_intent': 'comprehensive_dashboard',
                'entities': {'categories': []},
                'time_context': {'range': 'last_3_months'},
                'data_focus': 'both',
                'chart_preference': 'bar',
                'confidence_score': 0.8,
                'reasoning': ['Using visual query builder as fallback']
        }
        
        # Generate the chart based on analysis
            result = self._generate_chart_based_on_analysis(df_clean, analysis, query)
            return result
        
        except Exception as e:
            print(f"❌ Chart generation error: {str(e)}")
        return self._create_error_response(
            f"Chart generation error: {str(e)}",
            self.get_query_suggestions()
        )

    def pie_expenses_by_category(self, save=True, show=False, time_range=None):
        """Generate expenses pie chart"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")
    
        expenses = df[df['type'] == 'expense']
        if expenses.empty:
            return self._create_helpful_response("No Expenses", "No expense data available")
    
    # Apply time filter if specified
        if time_range:
            expenses = self._apply_time_filter(expenses, time_range)
    
        totals = expenses.groupby('category')['amount'].sum()
        fig = px.pie(values=totals.values, names=totals.index, title="Expenses by Category")
    
        caption = {
            'chart_type': 'pie',
            'summary': 'Expense distribution across categories'
    }
    
        return self._finalize_chart(fig, "expenses_pie", save, caption)

    def bar_income_vs_expense(self, save=True, show=False, time_range=None):
        """Generate income vs expense bar chart"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")

    # Apply time filter if specified
        if time_range:
            df = self._apply_time_filter(df, time_range)

        df_copy = df.copy()
    # Ensure 'date' is datetime before using .dt
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

        caption = {
            'chart_type': 'bar',
            'summary': 'Monthly income versus expenses comparison'
    }

        return self._finalize_chart(fig, "income_vs_expense", save, caption)

    def line_savings_over_time(self, time_range=None):
        """Generate savings over time chart"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")
        
        # Apply time filter if specified
        if time_range:
            df = self._apply_time_filter(df, time_range)
        
        # Use the savings trend method
        analysis = {
            'primary_intent': 'savings_trend',
            'entities': {'categories': []},
            'time_context': {'range': time_range or 'last_3_months'},
            'data_focus': 'savings',
            'chart_preference': 'line',
            'confidence_score': 1.0,
            'reasoning': ['Showing savings trend over time']  # ADD THIS LINE
        }
        
        return self._create_savings_trend(df, analysis, "Show savings over time")

    def create_interactive_dashboard(self, filters=None):
        """Create interactive dashboard"""
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_empty_chart("No data available for dashboard")

        # Apply filters if provided
        if filters:
            df = self._apply_filters(df, filters)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expenses by Category', 'Monthly Trends',
                        'Income vs Expense', 'Savings Progress'),
            specs=[[{"type": "pie"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]]
        )

        # Add visualizations
        self._add_dashboard_visualizations(fig, df)

        fig.update_layout(
            height=700,
            title_text="Interactive Financial Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig

    def comparative_charts(self, period1, period2):
        """Generate comparative charts"""
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
            'reasoning': [f'Comparing {period1} vs {period2}']
        }
    
        return self._create_time_comparison(df, analysis, f"Compare {period1} vs {period2}")

    def predictive_charts(self, months_to_predict=6):
        """Generate predictive financial charts"""
        # Create a simple predictive visualization
        df = self.data_agent.get_transactions_df()
        if df is None or df.empty:
            return self._create_helpful_response("No Data", "No transactions available")
        
        # Simple trend-based prediction
        expenses = df[df['type'] == 'expense']
        if expenses.empty:
            return self._create_helpful_response("No Expense Data", "No expense data for prediction")
        
        # Prepare monthly data
        expenses['date'] = pd.to_datetime(expenses['date'])
        expenses['month'] = expenses['date'].dt.to_period('M')
        monthly = expenses.groupby('month')['amount'].sum().tail(6)  # Last 6 months
        
        # REDUCED: Only need 2 months of data instead of 3
        if len(monthly) < 2:
            return self._create_helpful_response("Insufficient Data", "Need at least 2 months of data for prediction")
        
        # Simple prediction: average of available months
        available_months_avg = monthly.mean()
        predictions = [available_months_avg] * months_to_predict
        
        # Create chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=monthly.index.astype(str),
            y=monthly.values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        future_months = [f'Future {i+1}' for i in range(months_to_predict)]
        fig.add_trace(go.Scatter(
            x=future_months,
            y=predictions,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Expense Prediction - Next {months_to_predict} Months",
            xaxis_title="Month",
            yaxis_title="Amount ($)"
        )
        
        caption = {
            'chart_type': 'line',
            'summary': f'Simple expense prediction based on available months average',
            'prediction_note': f'Based on average of {len(monthly)} month(s) of spending data'
        }
        
        return self._finalize_chart(fig, "predictive_chart", False, caption)

    def get_context_aware_suggestions(self, current_query: str = "", user_history: List = None) -> List[str]:
        """Get context-aware query suggestions"""
        return self.query_assistant.get_smart_suggestions(current_query)

    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """Get query suggestions"""
        return self.query_assistant.get_smart_suggestions(partial_query)

    def _apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        df_copy = df.copy()

        if 'category' in filters and filters['category']:
            df_copy = df_copy[df_copy['category'].isin(filters['category'])]

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


class ChartCreatorAgent:
    """Main Chart Creator Agent class for external use"""
    def __init__(self, data_agent):
        self.enhanced_agent = EnhancedChartCreatorAgent(data_agent)
    
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

    # [Keep all other existing methods for backward compatibility...]
    # natural_language_to_chart, predictive_charts, comparative_charts, etc.
