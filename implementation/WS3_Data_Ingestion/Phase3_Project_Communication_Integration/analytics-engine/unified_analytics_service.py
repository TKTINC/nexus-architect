"""
Unified Analytics Service for Nexus Architect
Comprehensive analytics combining project management and communication data
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
API_LATENCY = Histogram('api_latency_seconds', 'API request latency', ['endpoint'])
ACTIVE_ANALYSES = Gauge('active_analyses', 'Number of active analyses')

@dataclass
class UnifiedInsight:
    """Unified insight combining project and communication data"""
    insight_id: str
    project_id: str
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_level: str
    data_sources: List[str]
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime

class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
    
    async def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                database=self.config.get('database', 'nexus_architect'),
                user=self.config.get('user', 'postgres'),
                password=self.config.get('password', 'password')
            )
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

class CacheManager:
    """Redis cache manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6379),
                db=self.config.get('db', 0),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache with TTL"""
        try:
            self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")

class UnifiedAnalyticsService:
    """Main unified analytics service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager = DatabaseManager(config.get('database', {}))
        self.cache_manager = CacheManager(config.get('cache', {}))
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
    
    async def initialize(self):
        """Initialize service components"""
        await self.db_manager.connect()
        await self.cache_manager.connect()
        logger.info("Unified Analytics Service initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'service': 'unified-analytics'
            })
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.route('/api/v1/projects/<project_id>/insights', methods=['GET'])
        def get_project_insights(project_id):
            """Get unified insights for a project"""
            start_time = datetime.now()
            
            try:
                # Get query parameters
                include_sources = request.args.getlist('sources')
                time_range = request.args.get('time_range', '30d')
                
                # Generate insights
                insights = asyncio.run(self._generate_unified_insights(
                    project_id, include_sources, time_range
                ))
                
                API_REQUESTS.labels(
                    endpoint='project_insights',
                    method='GET',
                    status='success'
                ).inc()
                
                return jsonify({
                    'project_id': project_id,
                    'insights': insights,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                })
            
            except Exception as e:
                logger.error(f"Project insights generation failed: {e}")
                API_REQUESTS.labels(
                    endpoint='project_insights',
                    method='GET',
                    status='error'
                ).inc()
                
                return jsonify({'error': str(e)}), 500
            
            finally:
                latency = (datetime.now() - start_time).total_seconds()
                API_LATENCY.labels(endpoint='project_insights').observe(latency)
        
        @self.app.route('/api/v1/teams/<team_id>/analytics', methods=['GET'])
        def get_team_analytics(team_id):
            """Get team analytics combining project and communication data"""
            start_time = datetime.now()
            
            try:
                analytics = asyncio.run(self._generate_team_analytics(team_id))
                
                API_REQUESTS.labels(
                    endpoint='team_analytics',
                    method='GET',
                    status='success'
                ).inc()
                
                return jsonify({
                    'team_id': team_id,
                    'analytics': analytics,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                })
            
            except Exception as e:
                logger.error(f"Team analytics generation failed: {e}")
                API_REQUESTS.labels(
                    endpoint='team_analytics',
                    method='GET',
                    status='error'
                ).inc()
                
                return jsonify({'error': str(e)}), 500
            
            finally:
                latency = (datetime.now() - start_time).total_seconds()
                API_LATENCY.labels(endpoint='team_analytics').observe(latency)
        
        @self.app.route('/api/v1/workflows/insights', methods=['GET'])
        def get_workflow_insights():
            """Get workflow efficiency insights"""
            start_time = datetime.now()
            
            try:
                insights = asyncio.run(self._generate_workflow_insights())
                
                API_REQUESTS.labels(
                    endpoint='workflow_insights',
                    method='GET',
                    status='success'
                ).inc()
                
                return jsonify({
                    'insights': insights,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                })
            
            except Exception as e:
                logger.error(f"Workflow insights generation failed: {e}")
                API_REQUESTS.labels(
                    endpoint='workflow_insights',
                    method='GET',
                    status='error'
                ).inc()
                
                return jsonify({'error': str(e)}), 500
            
            finally:
                latency = (datetime.now() - start_time).total_seconds()
                API_LATENCY.labels(endpoint='workflow_insights').observe(latency)
        
        @self.app.route('/api/v1/dashboard/summary', methods=['GET'])
        def get_dashboard_summary():
            """Get dashboard summary with key metrics"""
            start_time = datetime.now()
            
            try:
                summary = asyncio.run(self._generate_dashboard_summary())
                
                API_REQUESTS.labels(
                    endpoint='dashboard_summary',
                    method='GET',
                    status='success'
                ).inc()
                
                return jsonify(summary)
            
            except Exception as e:
                logger.error(f"Dashboard summary generation failed: {e}")
                API_REQUESTS.labels(
                    endpoint='dashboard_summary',
                    method='GET',
                    status='error'
                ).inc()
                
                return jsonify({'error': str(e)}), 500
            
            finally:
                latency = (datetime.now() - start_time).total_seconds()
                API_LATENCY.labels(endpoint='dashboard_summary').observe(latency)
    
    async def _generate_unified_insights(self, project_id: str, include_sources: List[str], 
                                       time_range: str) -> List[Dict[str, Any]]:
        """Generate unified insights combining multiple data sources"""
        ACTIVE_ANALYSES.inc()
        
        try:
            insights = []
            
            # Check cache first
            cache_key = f"insights:{project_id}:{':'.join(sorted(include_sources))}:{time_range}"
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Get project data from multiple sources
            project_data = await self._collect_project_data(project_id, include_sources, time_range)
            
            # Generate cross-source insights
            if 'project_management' in include_sources and 'communication' in include_sources:
                correlation_insights = await self._analyze_pm_communication_correlation(project_data)
                insights.extend(correlation_insights)
            
            if 'project_management' in include_sources:
                pm_insights = await self._analyze_project_management_data(project_data)
                insights.extend(pm_insights)
            
            if 'communication' in include_sources:
                comm_insights = await self._analyze_communication_data(project_data)
                insights.extend(comm_insights)
            
            # Convert insights to JSON format
            insights_json = [
                {
                    'insight_id': insight.insight_id,
                    'type': insight.insight_type,
                    'title': insight.title,
                    'description': insight.description,
                    'confidence': insight.confidence,
                    'impact_level': insight.impact_level,
                    'data_sources': insight.data_sources,
                    'recommendations': insight.recommendations,
                    'supporting_data': insight.supporting_data,
                    'generated_at': insight.generated_at.isoformat()
                }
                for insight in insights
            ]
            
            # Cache results
            await self.cache_manager.set(cache_key, json.dumps(insights_json), ttl=1800)  # 30 minutes
            
            return insights_json
        
        finally:
            ACTIVE_ANALYSES.dec()
    
    async def _collect_project_data(self, project_id: str, sources: List[str], 
                                  time_range: str) -> Dict[str, Any]:
        """Collect project data from multiple sources"""
        data = {'project_id': project_id}
        
        # Parse time range
        if time_range.endswith('d'):
            days = int(time_range[:-1])
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
        elif time_range.endswith('w'):
            weeks = int(time_range[:-1])
            start_date = datetime.now(timezone.utc) - timedelta(weeks=weeks)
        else:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Collect project management data
        if 'project_management' in sources:
            pm_query = """
                SELECT * FROM project_issues 
                WHERE project_id = %s AND created_at >= %s
                ORDER BY created_at DESC
            """
            data['issues'] = await self.db_manager.execute_query(pm_query, (project_id, start_date))
            
            # Get project info
            project_query = "SELECT * FROM projects WHERE project_id = %s"
            project_info = await self.db_manager.execute_query(project_query, (project_id,))
            data['project_info'] = project_info[0] if project_info else {}
        
        # Collect communication data
        if 'communication' in sources:
            comm_query = """
                SELECT * FROM communication_messages 
                WHERE project_id = %s AND timestamp >= %s
                ORDER BY timestamp DESC
            """
            data['messages'] = await self.db_manager.execute_query(comm_query, (project_id, start_date))
            
            # Get channels
            channels_query = """
                SELECT * FROM communication_channels 
                WHERE project_id = %s
            """
            data['channels'] = await self.db_manager.execute_query(channels_query, (project_id,))
        
        # Collect workflow data
        if 'workflows' in sources:
            workflow_query = """
                SELECT * FROM workflow_executions 
                WHERE project_id = %s AND started_at >= %s
                ORDER BY started_at DESC
            """
            data['workflow_executions'] = await self.db_manager.execute_query(workflow_query, (project_id, start_date))
        
        return data
    
    async def _analyze_pm_communication_correlation(self, project_data: Dict[str, Any]) -> List[UnifiedInsight]:
        """Analyze correlation between project management and communication data"""
        insights = []
        
        try:
            issues = project_data.get('issues', [])
            messages = project_data.get('messages', [])
            
            if not issues or not messages:
                return insights
            
            # Analyze communication around issue creation/completion
            issue_communication_correlation = self._calculate_issue_communication_correlation(issues, messages)
            
            if issue_communication_correlation['correlation'] > 0.7:
                insights.append(UnifiedInsight(
                    insight_id=str(uuid.uuid4()),
                    project_id=project_data['project_id'],
                    insight_type='correlation',
                    title='Strong Issue-Communication Correlation',
                    description=f'High correlation ({issue_communication_correlation["correlation"]:.2f}) between issue activity and team communication.',
                    confidence=0.85,
                    impact_level='medium',
                    data_sources=['project_management', 'communication'],
                    recommendations=[
                        'Leverage communication patterns to predict issue resolution',
                        'Encourage team discussion for complex issues',
                        'Monitor communication drops as early warning for delays'
                    ],
                    supporting_data=issue_communication_correlation,
                    generated_at=datetime.now(timezone.utc)
                ))
            
            # Analyze sentiment impact on productivity
            sentiment_productivity_correlation = self._analyze_sentiment_productivity_correlation(issues, messages)
            
            if sentiment_productivity_correlation['significant']:
                insights.append(UnifiedInsight(
                    insight_id=str(uuid.uuid4()),
                    project_id=project_data['project_id'],
                    insight_type='sentiment_impact',
                    title='Communication Sentiment Affects Productivity',
                    description='Team communication sentiment significantly correlates with issue resolution speed.',
                    confidence=0.75,
                    impact_level='high',
                    data_sources=['project_management', 'communication'],
                    recommendations=[
                        'Monitor team sentiment regularly',
                        'Address negative sentiment proactively',
                        'Celebrate positive achievements to maintain morale',
                        'Implement team wellness check-ins'
                    ],
                    supporting_data=sentiment_productivity_correlation,
                    generated_at=datetime.now(timezone.utc)
                ))
        
        except Exception as e:
            logger.error(f"PM-Communication correlation analysis failed: {e}")
        
        return insights
    
    def _calculate_issue_communication_correlation(self, issues: List[Dict], messages: List[Dict]) -> Dict[str, Any]:
        """Calculate correlation between issue activity and communication"""
        try:
            # Group by day
            daily_issues = {}
            daily_messages = {}
            
            for issue in issues:
                created_date = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00')).date()
                daily_issues[created_date] = daily_issues.get(created_date, 0) + 1
            
            for message in messages:
                message_date = datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00')).date()
                daily_messages[message_date] = daily_messages.get(message_date, 0) + 1
            
            # Calculate correlation
            common_dates = set(daily_issues.keys()) & set(daily_messages.keys())
            if len(common_dates) < 3:
                return {'correlation': 0, 'sample_size': len(common_dates)}
            
            issue_values = [daily_issues[date] for date in common_dates]
            message_values = [daily_messages[date] for date in common_dates]
            
            # Simple correlation calculation
            import numpy as np
            correlation = np.corrcoef(issue_values, message_values)[0, 1]
            
            return {
                'correlation': correlation if not np.isnan(correlation) else 0,
                'sample_size': len(common_dates),
                'avg_daily_issues': sum(issue_values) / len(issue_values),
                'avg_daily_messages': sum(message_values) / len(message_values)
            }
        
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return {'correlation': 0, 'error': str(e)}
    
    def _analyze_sentiment_productivity_correlation(self, issues: List[Dict], messages: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between sentiment and productivity"""
        try:
            # Calculate average sentiment by week
            weekly_sentiment = {}
            weekly_completions = {}
            
            for message in messages:
                if message.get('sentiment_score') is not None:
                    message_date = datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
                    week = message_date.isocalendar()[1]
                    
                    if week not in weekly_sentiment:
                        weekly_sentiment[week] = []
                    weekly_sentiment[week].append(message['sentiment_score'])
            
            for issue in issues:
                if issue.get('status') == 'done' and issue.get('completed_at'):
                    completed_date = datetime.fromisoformat(issue['completed_at'].replace('Z', '+00:00'))
                    week = completed_date.isocalendar()[1]
                    weekly_completions[week] = weekly_completions.get(week, 0) + 1
            
            # Calculate averages
            avg_sentiment_by_week = {
                week: sum(sentiments) / len(sentiments)
                for week, sentiments in weekly_sentiment.items()
                if sentiments
            }
            
            # Find correlation
            common_weeks = set(avg_sentiment_by_week.keys()) & set(weekly_completions.keys())
            if len(common_weeks) < 3:
                return {'significant': False, 'sample_size': len(common_weeks)}
            
            sentiment_values = [avg_sentiment_by_week[week] for week in common_weeks]
            completion_values = [weekly_completions[week] for week in common_weeks]
            
            import numpy as np
            correlation = np.corrcoef(sentiment_values, completion_values)[0, 1]
            
            return {
                'significant': abs(correlation) > 0.5 and not np.isnan(correlation),
                'correlation': correlation if not np.isnan(correlation) else 0,
                'sample_size': len(common_weeks),
                'avg_sentiment': sum(sentiment_values) / len(sentiment_values),
                'avg_completions': sum(completion_values) / len(completion_values)
            }
        
        except Exception as e:
            logger.error(f"Sentiment-productivity correlation analysis failed: {e}")
            return {'significant': False, 'error': str(e)}
    
    async def _analyze_project_management_data(self, project_data: Dict[str, Any]) -> List[UnifiedInsight]:
        """Analyze project management specific data"""
        insights = []
        
        try:
            issues = project_data.get('issues', [])
            if not issues:
                return insights
            
            # Analyze velocity trends
            velocity_insight = self._analyze_velocity_trends(issues, project_data['project_id'])
            if velocity_insight:
                insights.append(velocity_insight)
            
            # Analyze bottlenecks
            bottleneck_insight = self._analyze_workflow_bottlenecks(issues, project_data['project_id'])
            if bottleneck_insight:
                insights.append(bottleneck_insight)
        
        except Exception as e:
            logger.error(f"Project management analysis failed: {e}")
        
        return insights
    
    async def _analyze_communication_data(self, project_data: Dict[str, Any]) -> List[UnifiedInsight]:
        """Analyze communication specific data"""
        insights = []
        
        try:
            messages = project_data.get('messages', [])
            if not messages:
                return insights
            
            # Analyze communication patterns
            pattern_insight = self._analyze_communication_patterns(messages, project_data['project_id'])
            if pattern_insight:
                insights.append(pattern_insight)
            
            # Analyze team engagement
            engagement_insight = self._analyze_team_engagement(messages, project_data['project_id'])
            if engagement_insight:
                insights.append(engagement_insight)
        
        except Exception as e:
            logger.error(f"Communication analysis failed: {e}")
        
        return insights
    
    def _analyze_velocity_trends(self, issues: List[Dict], project_id: str) -> Optional[UnifiedInsight]:
        """Analyze issue velocity trends"""
        try:
            completed_issues = [i for i in issues if i.get('status') == 'done']
            if len(completed_issues) < 5:
                return None
            
            # Calculate weekly velocity
            weekly_velocity = {}
            for issue in completed_issues:
                if issue.get('completed_at'):
                    completed_date = datetime.fromisoformat(issue['completed_at'].replace('Z', '+00:00'))
                    week = completed_date.isocalendar()[1]
                    weekly_velocity[week] = weekly_velocity.get(week, 0) + 1
            
            if len(weekly_velocity) < 3:
                return None
            
            # Calculate trend
            weeks = sorted(weekly_velocity.keys())
            velocities = [weekly_velocity[week] for week in weeks]
            
            # Simple trend calculation
            recent_avg = sum(velocities[-2:]) / 2 if len(velocities) >= 2 else velocities[-1]
            overall_avg = sum(velocities) / len(velocities)
            trend = (recent_avg - overall_avg) / overall_avg if overall_avg > 0 else 0
            
            if trend < -0.3:
                return UnifiedInsight(
                    insight_id=str(uuid.uuid4()),
                    project_id=project_id,
                    insight_type='velocity_decline',
                    title='Declining Velocity Trend Detected',
                    description=f'Issue completion velocity has declined by {abs(trend):.1%} compared to average.',
                    confidence=0.80,
                    impact_level='high',
                    data_sources=['project_management'],
                    recommendations=[
                        'Investigate causes of velocity decline',
                        'Review team capacity and workload distribution',
                        'Address technical debt or process bottlenecks',
                        'Consider sprint planning adjustments'
                    ],
                    supporting_data={
                        'trend': trend,
                        'recent_avg_velocity': recent_avg,
                        'overall_avg_velocity': overall_avg,
                        'weeks_analyzed': len(weeks)
                    },
                    generated_at=datetime.now(timezone.utc)
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Velocity trend analysis failed: {e}")
            return None
    
    def _analyze_workflow_bottlenecks(self, issues: List[Dict], project_id: str) -> Optional[UnifiedInsight]:
        """Analyze workflow bottlenecks"""
        try:
            # Analyze time in each status
            status_times = {}
            for issue in issues:
                status = issue.get('status', 'unknown')
                created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                updated_at = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
                
                time_in_status = (updated_at - created_at).total_seconds() / 3600  # hours
                
                if status not in status_times:
                    status_times[status] = []
                status_times[status].append(time_in_status)
            
            # Find bottleneck status
            avg_times = {
                status: sum(times) / len(times)
                for status, times in status_times.items()
                if len(times) > 2
            }
            
            if not avg_times:
                return None
            
            bottleneck_status = max(avg_times, key=avg_times.get)
            bottleneck_time = avg_times[bottleneck_status]
            
            # Check if it's significantly higher than others
            other_times = [time for status, time in avg_times.items() if status != bottleneck_status]
            if other_times and bottleneck_time > sum(other_times) / len(other_times) * 2:
                return UnifiedInsight(
                    insight_id=str(uuid.uuid4()),
                    project_id=project_id,
                    insight_type='workflow_bottleneck',
                    title=f'Workflow Bottleneck in {bottleneck_status.title()} Status',
                    description=f'Issues spend an average of {bottleneck_time:.1f} hours in {bottleneck_status} status.',
                    confidence=0.75,
                    impact_level='medium',
                    data_sources=['project_management'],
                    recommendations=[
                        f'Review processes for {bottleneck_status} status',
                        'Identify and address resource constraints',
                        'Consider workflow automation opportunities',
                        'Implement status-specific metrics and monitoring'
                    ],
                    supporting_data={
                        'bottleneck_status': bottleneck_status,
                        'avg_time_hours': bottleneck_time,
                        'status_times': avg_times,
                        'issues_analyzed': len(issues)
                    },
                    generated_at=datetime.now(timezone.utc)
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Bottleneck analysis failed: {e}")
            return None
    
    def _analyze_communication_patterns(self, messages: List[Dict], project_id: str) -> Optional[UnifiedInsight]:
        """Analyze communication patterns"""
        try:
            if len(messages) < 10:
                return None
            
            # Analyze communication frequency by hour
            hourly_activity = {}
            for message in messages:
                timestamp = datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
                hour = timestamp.hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            
            # Find peak hours
            peak_hour = max(hourly_activity, key=hourly_activity.get)
            peak_activity = hourly_activity[peak_hour]
            total_messages = sum(hourly_activity.values())
            
            # Check for concentration
            if peak_activity / total_messages > 0.3:  # More than 30% in one hour
                return UnifiedInsight(
                    insight_id=str(uuid.uuid4()),
                    project_id=project_id,
                    insight_type='communication_pattern',
                    title='Communication Concentrated in Peak Hours',
                    description=f'{peak_activity/total_messages:.1%} of communication occurs around {peak_hour}:00.',
                    confidence=0.70,
                    impact_level='low',
                    data_sources=['communication'],
                    recommendations=[
                        'Consider asynchronous communication for global teams',
                        'Implement communication guidelines for different time zones',
                        'Use threaded discussions for complex topics',
                        'Schedule important discussions during peak hours'
                    ],
                    supporting_data={
                        'peak_hour': peak_hour,
                        'peak_percentage': peak_activity / total_messages,
                        'hourly_distribution': hourly_activity,
                        'total_messages': total_messages
                    },
                    generated_at=datetime.now(timezone.utc)
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Communication pattern analysis failed: {e}")
            return None
    
    def _analyze_team_engagement(self, messages: List[Dict], project_id: str) -> Optional[UnifiedInsight]:
        """Analyze team engagement levels"""
        try:
            # Analyze user participation
            user_activity = {}
            for message in messages:
                author = message.get('author')
                if author:
                    user_activity[author] = user_activity.get(author, 0) + 1
            
            if len(user_activity) < 2:
                return None
            
            # Calculate engagement distribution
            total_messages = sum(user_activity.values())
            sorted_activity = sorted(user_activity.values(), reverse=True)
            
            # Check for engagement imbalance
            top_contributor_percentage = sorted_activity[0] / total_messages
            
            if top_contributor_percentage > 0.6:  # One person dominates
                return UnifiedInsight(
                    insight_id=str(uuid.uuid4()),
                    project_id=project_id,
                    insight_type='engagement_imbalance',
                    title='Communication Dominated by Single Contributor',
                    description=f'One team member contributes {top_contributor_percentage:.1%} of all communication.',
                    confidence=0.85,
                    impact_level='medium',
                    data_sources=['communication'],
                    recommendations=[
                        'Encourage broader team participation',
                        'Rotate meeting facilitation and discussion leadership',
                        'Create inclusive communication practices',
                        'Implement structured discussion formats'
                    ],
                    supporting_data={
                        'top_contributor_percentage': top_contributor_percentage,
                        'active_contributors': len(user_activity),
                        'total_messages': total_messages,
                        'activity_distribution': sorted_activity[:5]  # Top 5
                    },
                    generated_at=datetime.now(timezone.utc)
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Team engagement analysis failed: {e}")
            return None
    
    async def _generate_team_analytics(self, team_id: str) -> Dict[str, Any]:
        """Generate comprehensive team analytics"""
        try:
            # Get team projects
            team_projects_query = "SELECT project_id FROM team_projects WHERE team_id = %s"
            team_projects = await self.db_manager.execute_query(team_projects_query, (team_id,))
            
            if not team_projects:
                return {'error': 'No projects found for team'}
            
            project_ids = [p['project_id'] for p in team_projects]
            
            # Aggregate analytics across all team projects
            analytics = {
                'team_id': team_id,
                'projects_count': len(project_ids),
                'productivity_metrics': {},
                'collaboration_metrics': {},
                'communication_metrics': {},
                'trends': {}
            }
            
            # Calculate productivity metrics
            total_issues = 0
            completed_issues = 0
            avg_resolution_time = 0
            
            for project_id in project_ids:
                project_data = await self._collect_project_data(
                    project_id, ['project_management'], '30d'
                )
                
                issues = project_data.get('issues', [])
                total_issues += len(issues)
                
                project_completed = [i for i in issues if i.get('status') == 'done']
                completed_issues += len(project_completed)
                
                # Calculate resolution times for this project
                resolution_times = []
                for issue in project_completed:
                    if issue.get('created_at') and issue.get('completed_at'):
                        created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                        completed = datetime.fromisoformat(issue['completed_at'].replace('Z', '+00:00'))
                        resolution_times.append((completed - created).total_seconds() / 3600)
                
                if resolution_times:
                    avg_resolution_time += sum(resolution_times) / len(resolution_times)
            
            analytics['productivity_metrics'] = {
                'total_issues': total_issues,
                'completed_issues': completed_issues,
                'completion_rate': completed_issues / total_issues if total_issues > 0 else 0,
                'avg_resolution_time_hours': avg_resolution_time / len(project_ids) if project_ids else 0
            }
            
            # Calculate communication metrics
            total_messages = 0
            total_participants = set()
            avg_sentiment = 0
            
            for project_id in project_ids:
                project_data = await self._collect_project_data(
                    project_id, ['communication'], '30d'
                )
                
                messages = project_data.get('messages', [])
                total_messages += len(messages)
                
                for message in messages:
                    if message.get('author'):
                        total_participants.add(message['author'])
                    
                    if message.get('sentiment_score') is not None:
                        avg_sentiment += message['sentiment_score']
            
            analytics['communication_metrics'] = {
                'total_messages': total_messages,
                'active_participants': len(total_participants),
                'avg_sentiment': avg_sentiment / total_messages if total_messages > 0 else 0,
                'messages_per_day': total_messages / 30
            }
            
            return analytics
        
        except Exception as e:
            logger.error(f"Team analytics generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_workflow_insights(self) -> List[Dict[str, Any]]:
        """Generate workflow efficiency insights"""
        try:
            # Get workflow execution data
            workflow_query = """
                SELECT workflow_id, status, started_at, completed_at, error_message
                FROM workflow_executions 
                WHERE started_at >= %s
                ORDER BY started_at DESC
            """
            
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            executions = await self.db_manager.execute_query(workflow_query, (thirty_days_ago,))
            
            if not executions:
                return []
            
            insights = []
            
            # Analyze success rates
            total_executions = len(executions)
            successful_executions = len([e for e in executions if e['status'] == 'completed'])
            success_rate = successful_executions / total_executions
            
            if success_rate < 0.8:
                insights.append({
                    'type': 'workflow_reliability',
                    'title': 'Low Workflow Success Rate',
                    'description': f'Workflow success rate is {success_rate:.1%}, below recommended 80%.',
                    'impact_level': 'high',
                    'recommendations': [
                        'Review and fix failing workflows',
                        'Improve error handling and retry logic',
                        'Monitor workflow dependencies',
                        'Implement better testing for workflow changes'
                    ],
                    'supporting_data': {
                        'success_rate': success_rate,
                        'total_executions': total_executions,
                        'failed_executions': total_executions - successful_executions
                    }
                })
            
            # Analyze execution times
            completed_executions = [
                e for e in executions 
                if e['status'] == 'completed' and e['completed_at']
            ]
            
            if completed_executions:
                execution_times = []
                for execution in completed_executions:
                    started = datetime.fromisoformat(execution['started_at'].replace('Z', '+00:00'))
                    completed = datetime.fromisoformat(execution['completed_at'].replace('Z', '+00:00'))
                    execution_times.append((completed - started).total_seconds())
                
                avg_execution_time = sum(execution_times) / len(execution_times)
                
                if avg_execution_time > 300:  # More than 5 minutes
                    insights.append({
                        'type': 'workflow_performance',
                        'title': 'Long Workflow Execution Times',
                        'description': f'Average workflow execution time is {avg_execution_time/60:.1f} minutes.',
                        'impact_level': 'medium',
                        'recommendations': [
                            'Optimize workflow steps and reduce complexity',
                            'Implement parallel execution where possible',
                            'Review external API call timeouts',
                            'Consider workflow decomposition'
                        ],
                        'supporting_data': {
                            'avg_execution_time_seconds': avg_execution_time,
                            'executions_analyzed': len(execution_times)
                        }
                    })
            
            return insights
        
        except Exception as e:
            logger.error(f"Workflow insights generation failed: {e}")
            return []
    
    async def _generate_dashboard_summary(self) -> Dict[str, Any]:
        """Generate dashboard summary with key metrics"""
        try:
            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'overview': {},
                'alerts': [],
                'trends': {}
            }
            
            # Get overall statistics
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            
            # Project statistics
            projects_query = "SELECT COUNT(*) as count FROM projects WHERE is_active = true"
            projects_result = await self.db_manager.execute_query(projects_query)
            active_projects = projects_result[0]['count'] if projects_result else 0
            
            # Issue statistics
            issues_query = """
                SELECT 
                    COUNT(*) as total_issues,
                    COUNT(CASE WHEN status = 'done' THEN 1 END) as completed_issues
                FROM project_issues 
                WHERE created_at >= %s
            """
            issues_result = await self.db_manager.execute_query(issues_query, (thirty_days_ago,))
            
            if issues_result:
                total_issues = issues_result[0]['total_issues']
                completed_issues = issues_result[0]['completed_issues']
                completion_rate = completed_issues / total_issues if total_issues > 0 else 0
            else:
                total_issues = completed_issues = completion_rate = 0
            
            # Communication statistics
            messages_query = "SELECT COUNT(*) as count FROM communication_messages WHERE timestamp >= %s"
            messages_result = await self.db_manager.execute_query(messages_query, (thirty_days_ago,))
            total_messages = messages_result[0]['count'] if messages_result else 0
            
            # Workflow statistics
            workflows_query = """
                SELECT 
                    COUNT(*) as total_executions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions
                FROM workflow_executions 
                WHERE started_at >= %s
            """
            workflows_result = await self.db_manager.execute_query(workflows_query, (thirty_days_ago,))
            
            if workflows_result:
                total_workflow_executions = workflows_result[0]['total_executions']
                successful_workflow_executions = workflows_result[0]['successful_executions']
                workflow_success_rate = successful_workflow_executions / total_workflow_executions if total_workflow_executions > 0 else 0
            else:
                total_workflow_executions = successful_workflow_executions = workflow_success_rate = 0
            
            summary['overview'] = {
                'active_projects': active_projects,
                'total_issues_30d': total_issues,
                'issue_completion_rate': completion_rate,
                'total_messages_30d': total_messages,
                'workflow_executions_30d': total_workflow_executions,
                'workflow_success_rate': workflow_success_rate
            }
            
            # Generate alerts
            if completion_rate < 0.5:
                summary['alerts'].append({
                    'type': 'warning',
                    'message': f'Low issue completion rate: {completion_rate:.1%}',
                    'action': 'Review project progress and remove blockers'
                })
            
            if workflow_success_rate < 0.8 and total_workflow_executions > 0:
                summary['alerts'].append({
                    'type': 'error',
                    'message': f'Low workflow success rate: {workflow_success_rate:.1%}',
                    'action': 'Review and fix failing workflows'
                })
            
            if total_messages < 100:  # Less than ~3 messages per day
                summary['alerts'].append({
                    'type': 'info',
                    'message': 'Low team communication activity',
                    'action': 'Encourage team collaboration and communication'
                })
            
            return summary
        
        except Exception as e:
            logger.error(f"Dashboard summary generation failed: {e}")
            return {'error': str(e)}
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Unified Analytics Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import asyncio
    
    # Example configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'nexus_architect',
            'user': 'postgres',
            'password': 'password'
        },
        'cache': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    }
    
    service = UnifiedAnalyticsService(config)
    
    # Initialize service
    asyncio.run(service.initialize())
    
    # Run the service
    service.run(port=8080, debug=True)

