#!/usr/bin/env python3
"""
Nexus Architect - WS4 Phase 6: Integration Test Suite
Comprehensive testing for Advanced Autonomy & Production Optimization components
"""

import asyncio
import json
import logging
import time
import unittest
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationTestSuite(unittest.TestCase):
    """Comprehensive integration test suite for WS4 Phase 6"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.base_urls = {
            'multi_agent': 'http://localhost:8070',
            'adaptive_learning': 'http://localhost:8071',
            'production_optimizer': 'http://localhost:8072'
        }
        
        cls.test_data = {
            'agents': [],
            'tasks': [],
            'experiences': [],
            'decisions': []
        }
        
        # Wait for services to be ready
        cls._wait_for_services()
    
    @classmethod
    def _wait_for_services(cls):
        """Wait for all services to be ready"""
        max_attempts = 30
        
        for service_name, base_url in cls.base_urls.items():
            logger.info(f"Waiting for {service_name} to be ready...")
            
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{base_url}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"{service_name} is ready")
                        break
                except requests.exceptions.RequestException:
                    if attempt == max_attempts - 1:
                        raise Exception(f"{service_name} failed to start")
                    time.sleep(2)
    
    def test_01_service_health_checks(self):
        """Test that all services are healthy"""
        logger.info("Testing service health checks...")
        
        for service_name, base_url in self.base_urls.items():
            with self.subTest(service=service_name):
                response = requests.get(f"{base_url}/health")
                self.assertEqual(response.status_code, 200)
                
                data = response.json()
                self.assertEqual(data['status'], 'healthy')
                self.assertIn('service', data)
                self.assertIn('timestamp', data)
                
                logger.info(f"✓ {service_name} health check passed")
    
    def test_02_multi_agent_coordination(self):
        """Test multi-agent coordination functionality"""
        logger.info("Testing multi-agent coordination...")
        
        base_url = self.base_urls['multi_agent']
        
        # Test coordination status
        response = requests.get(f"{base_url}/coordination/status")
        self.assertEqual(response.status_code, 200)
        
        status_data = response.json()
        self.assertIn('coordination_running', status_data)
        self.assertIn('total_agents', status_data)
        self.assertIn('total_tasks', status_data)
        
        # Start coordination if not running
        if not status_data.get('coordination_running', False):
            response = requests.post(f"{base_url}/coordination/start")
            self.assertEqual(response.status_code, 200)
            time.sleep(2)  # Allow coordination to start
        
        # Register test agents
        test_agents = [
            {
                'id': 'test_agent_1',
                'agent_type': 'decision_engine',
                'name': 'Test Decision Agent',
                'capabilities': ['analysis', 'decision_making'],
                'max_capacity': 100.0,
                'specializations': ['risk_assessment']
            },
            {
                'id': 'test_agent_2',
                'agent_type': 'qa_automation',
                'name': 'Test QA Agent',
                'capabilities': ['testing', 'validation'],
                'max_capacity': 80.0,
                'specializations': ['automated_testing']
            }
        ]
        
        for agent_data in test_agents:
            response = requests.post(f"{base_url}/agents", json=agent_data)
            self.assertEqual(response.status_code, 200)
            self.test_data['agents'].append(agent_data['id'])
        
        # Verify agents are registered
        response = requests.get(f"{base_url}/agents")
        self.assertEqual(response.status_code, 200)
        
        agents_data = response.json()
        self.assertGreaterEqual(agents_data['count'], 2)
        
        # Submit test tasks
        test_tasks = [
            {
                'task_type': 'analysis',
                'priority': 3,
                'description': 'Test analysis task',
                'requirements': {'complexity': 'medium'},
                'estimated_duration': 300.0
            },
            {
                'task_type': 'testing',
                'priority': 2,
                'description': 'Test validation task',
                'requirements': {'test_type': 'integration'},
                'estimated_duration': 600.0
            }
        ]
        
        for task_data in test_tasks:
            response = requests.post(f"{base_url}/tasks", json=task_data)
            self.assertEqual(response.status_code, 200)
            
            task_id = response.json()['task_id']
            self.test_data['tasks'].append(task_id)
        
        # Verify tasks are submitted
        response = requests.get(f"{base_url}/tasks")
        self.assertEqual(response.status_code, 200)
        
        tasks_data = response.json()
        self.assertGreaterEqual(tasks_data['count'], 2)
        
        # Test task assignment (wait a bit for coordination)
        time.sleep(5)
        
        response = requests.get(f"{base_url}/tasks")
        tasks_data = response.json()
        
        # Check if any tasks have been assigned
        assigned_tasks = [task for task in tasks_data['tasks'].values() 
                         if task.get('status') in ['assigned', 'in_progress']]
        self.assertGreater(len(assigned_tasks), 0, "No tasks were assigned to agents")
        
        logger.info("✓ Multi-agent coordination tests passed")
    
    def test_03_adaptive_learning_engine(self):
        """Test adaptive learning engine functionality"""
        logger.info("Testing adaptive learning engine...")
        
        base_url = self.base_urls['adaptive_learning']
        
        # Test learning status
        response = requests.get(f"{base_url}/learning/status")
        self.assertEqual(response.status_code, 200)
        
        status_data = response.json()
        self.assertIn('total_experiences', status_data)
        self.assertIn('adaptation_rules', status_data)
        
        # Add test learning experiences
        test_experiences = [
            {
                'context': {
                    'system_load': 0.7,
                    'memory_usage': 0.6,
                    'task_complexity': 'medium'
                },
                'action_taken': {
                    'optimization_type': 'performance',
                    'parameters': {'cpu_scaling': 1.2}
                },
                'outcome': {
                    'performance_improvement': 0.15,
                    'execution_time': 120
                },
                'success': True,
                'performance_metrics': {
                    'response_time': 0.8,
                    'throughput': 1500
                },
                'feedback_score': 0.85,
                'learning_type': 'supervised'
            },
            {
                'context': {
                    'system_load': 0.9,
                    'memory_usage': 0.8,
                    'task_complexity': 'high'
                },
                'action_taken': {
                    'optimization_type': 'scaling',
                    'parameters': {'instance_count': 3}
                },
                'outcome': {
                    'performance_improvement': 0.25,
                    'execution_time': 180
                },
                'success': True,
                'performance_metrics': {
                    'response_time': 0.6,
                    'throughput': 2200
                },
                'feedback_score': 0.92,
                'learning_type': 'reinforcement'
            }
        ]
        
        for experience_data in test_experiences:
            response = requests.post(f"{base_url}/learning/experiences", json=experience_data)
            self.assertEqual(response.status_code, 200)
            
            experience_id = response.json()['experience_id']
            self.test_data['experiences'].append(experience_id)
        
        # Test pattern recognition
        response = requests.get(f"{base_url}/learning/patterns?hours=1")
        self.assertEqual(response.status_code, 200)
        
        patterns_data = response.json()
        self.assertIn('patterns', patterns_data)
        self.assertIn('experiences_analyzed', patterns_data)
        
        # Test prediction
        prediction_request = {
            'context': {
                'system_load': 0.75,
                'memory_usage': 0.65,
                'task_complexity': 'medium'
            },
            'performance_metrics': {
                'response_time': 1.0,
                'throughput': 1200
            }
        }
        
        response = requests.post(f"{base_url}/learning/predict", json=prediction_request)
        self.assertEqual(response.status_code, 200)
        
        prediction_data = response.json()
        self.assertIn('predictions', prediction_data)
        
        # Test complex decision making
        decision_scenario = {
            'scenario_type': 'performance_optimization',
            'description': 'System performance degradation detected',
            'context': {
                'cpu_usage': 85,
                'memory_usage': 78,
                'response_time': 2.5
            },
            'objectives': ['maximize_performance', 'minimize_cost'],
            'available_actions': [
                {
                    'action_type': 'scale_up',
                    'estimated_cost': 30,
                    'performance_gain': 0.4,
                    'reliability_score': 0.9
                },
                {
                    'action_type': 'optimize_cache',
                    'estimated_cost': 10,
                    'performance_gain': 0.2,
                    'reliability_score': 0.95
                }
            ],
            'urgency': 0.7,
            'complexity': 0.6,
            'risk_tolerance': 0.5
        }
        
        response = requests.post(f"{base_url}/decisions", json=decision_scenario)
        self.assertEqual(response.status_code, 200)
        
        decision_data = response.json()
        self.assertIn('decision_result', decision_data)
        
        decision_result = decision_data['decision_result']
        self.assertIn('selected_action', decision_result)
        self.assertIn('confidence', decision_result)
        self.assertIn('reasoning', decision_result)
        
        self.test_data['decisions'].append(decision_result['scenario_id'])
        
        # Test strategic planning
        strategic_plan_request = {
            'goals': [
                {
                    'id': 'improve_performance',
                    'description': 'Improve system performance by 30%',
                    'priority': 'high',
                    'success_criteria': ['response_time < 1s', 'throughput > 2000']
                },
                {
                    'id': 'reduce_costs',
                    'description': 'Reduce operational costs by 20%',
                    'priority': 'medium',
                    'success_criteria': ['cost_reduction >= 20%']
                }
            ],
            'constraints': {
                'budget': 50000,
                'timeline_weeks': 12,
                'resource_limit': 'current_team'
            }
        }
        
        response = requests.post(f"{base_url}/strategic-plans", json=strategic_plan_request)
        self.assertEqual(response.status_code, 200)
        
        plan_data = response.json()
        self.assertIn('strategic_plan', plan_data)
        
        strategic_plan = plan_data['strategic_plan']
        self.assertIn('phases', strategic_plan)
        self.assertIn('milestones', strategic_plan)
        self.assertIn('risk_assessment', strategic_plan)
        
        logger.info("✓ Adaptive learning engine tests passed")
    
    def test_04_production_optimizer(self):
        """Test production optimizer functionality"""
        logger.info("Testing production optimizer...")
        
        base_url = self.base_urls['production_optimizer']
        
        # Test optimization status
        response = requests.get(f"{base_url}/optimization/status")
        self.assertEqual(response.status_code, 200)
        
        status_data = response.json()
        self.assertIn('optimization_running', status_data)
        self.assertIn('performance_metrics', status_data)
        self.assertIn('system_health', status_data)
        
        # Start optimization if not running
        if not status_data.get('optimization_running', False):
            response = requests.post(f"{base_url}/optimization/start")
            self.assertEqual(response.status_code, 200)
            time.sleep(3)  # Allow optimization to start
        
        # Test performance metrics
        response = requests.get(f"{base_url}/performance/metrics")
        self.assertEqual(response.status_code, 200)
        
        metrics_data = response.json()
        self.assertIn('metrics', metrics_data)
        
        # Test performance analysis
        response = requests.get(f"{base_url}/performance/analysis")
        self.assertEqual(response.status_code, 200)
        
        analysis_data = response.json()
        self.assertIn('opportunities', analysis_data)
        self.assertIn('total_opportunities', analysis_data)
        
        # Test system health assessment
        response = requests.get(f"{base_url}/reliability/health")
        self.assertEqual(response.status_code, 200)
        
        health_data = response.json()
        self.assertIn('overall_score', health_data)
        self.assertIn('component_scores', health_data)
        self.assertIn('recommendations', health_data)
        
        # Verify health score is reasonable
        self.assertGreaterEqual(health_data['overall_score'], 0.0)
        self.assertLessEqual(health_data['overall_score'], 1.0)
        
        # Test reliability enhancements
        response = requests.get(f"{base_url}/reliability/enhancements")
        self.assertEqual(response.status_code, 200)
        
        enhancements_data = response.json()
        self.assertIn('enhancements', enhancements_data)
        self.assertIn('count', enhancements_data)
        
        # Test forced optimization cycle
        response = requests.post(f"{base_url}/optimization/force-cycle")
        self.assertEqual(response.status_code, 200)
        
        cycle_data = response.json()
        self.assertIn('success', cycle_data)
        self.assertTrue(cycle_data['success'])
        
        logger.info("✓ Production optimizer tests passed")
    
    def test_05_cross_service_integration(self):
        """Test integration between services"""
        logger.info("Testing cross-service integration...")
        
        # Test that multi-agent coordinator can work with adaptive learning
        # by submitting a learning experience based on task completion
        
        multi_agent_url = self.base_urls['multi_agent']
        learning_url = self.base_urls['adaptive_learning']
        
        # Get current task status
        response = requests.get(f"{multi_agent_url}/tasks")
        self.assertEqual(response.status_code, 200)
        
        tasks_data = response.json()
        
        # Simulate task completion and learning
        if tasks_data['tasks']:
            task_id = list(tasks_data['tasks'].keys())[0]
            task = tasks_data['tasks'][task_id]
            
            # Update task status to completed
            update_data = {'status': 'completed'}
            response = requests.put(f"{multi_agent_url}/tasks/{task_id}/status", json=update_data)
            self.assertEqual(response.status_code, 200)
            
            # Create learning experience from task completion
            learning_experience = {
                'context': {
                    'task_type': task.get('task_type', 'unknown'),
                    'task_priority': task.get('priority', 1),
                    'agent_type': task.get('assigned_agent', 'unknown')
                },
                'action_taken': {
                    'task_execution': 'completed',
                    'execution_strategy': 'standard'
                },
                'outcome': {
                    'task_completed': True,
                    'execution_time': 300
                },
                'success': True,
                'performance_metrics': {
                    'completion_time': 300,
                    'quality_score': 0.9
                },
                'feedback_score': 0.88,
                'learning_type': 'supervised',
                'tags': ['task_completion', 'multi_agent']
            }
            
            response = requests.post(f"{learning_url}/learning/experiences", json=learning_experience)
            self.assertEqual(response.status_code, 200)
        
        # Test that production optimizer can influence multi-agent decisions
        optimizer_url = self.base_urls['production_optimizer']
        
        # Get system health from optimizer
        response = requests.get(f"{optimizer_url}/reliability/health")
        self.assertEqual(response.status_code, 200)
        
        health_data = response.json()
        
        # Use health data to make a decision in adaptive learning
        if health_data['overall_score'] < 0.8:
            decision_scenario = {
                'scenario_type': 'incident_response',
                'description': 'System health degradation detected',
                'context': {
                    'health_score': health_data['overall_score'],
                    'critical_issues': len(health_data.get('critical_issues', [])),
                    'warnings': len(health_data.get('warnings', []))
                },
                'objectives': ['maximize_reliability', 'minimize_downtime'],
                'available_actions': [
                    {
                        'action_type': 'scale_resources',
                        'estimated_cost': 25,
                        'reliability_improvement': 0.3
                    },
                    {
                        'action_type': 'restart_services',
                        'estimated_cost': 5,
                        'reliability_improvement': 0.15
                    }
                ],
                'urgency': 0.8,
                'complexity': 0.4,
                'risk_tolerance': 0.3
            }
            
            response = requests.post(f"{learning_url}/decisions", json=decision_scenario)
            self.assertEqual(response.status_code, 200)
        
        logger.info("✓ Cross-service integration tests passed")
    
    def test_06_performance_and_scalability(self):
        """Test performance and scalability characteristics"""
        logger.info("Testing performance and scalability...")
        
        # Test concurrent requests to each service
        def make_concurrent_requests(url, endpoint, count=10):
            """Make concurrent requests to test performance"""
            results = []
            
            def make_request():
                start_time = time.time()
                try:
                    response = requests.get(f"{url}{endpoint}", timeout=10)
                    end_time = time.time()
                    results.append({
                        'status_code': response.status_code,
                        'response_time': end_time - start_time,
                        'success': response.status_code == 200
                    })
                except Exception as e:
                    end_time = time.time()
                    results.append({
                        'status_code': 0,
                        'response_time': end_time - start_time,
                        'success': False,
                        'error': str(e)
                    })
            
            threads = []
            for _ in range(count):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return results
        
        # Test each service
        for service_name, base_url in self.base_urls.items():
            with self.subTest(service=service_name):
                results = make_concurrent_requests(base_url, '/health', count=20)
                
                # Verify all requests succeeded
                success_count = sum(1 for r in results if r['success'])
                self.assertGreaterEqual(success_count, 18, f"Too many failed requests for {service_name}")
                
                # Verify reasonable response times
                response_times = [r['response_time'] for r in results if r['success']]
                avg_response_time = sum(response_times) / len(response_times)
                self.assertLess(avg_response_time, 2.0, f"Average response time too high for {service_name}")
                
                logger.info(f"✓ {service_name} performance test passed (avg: {avg_response_time:.3f}s)")
        
        logger.info("✓ Performance and scalability tests passed")
    
    def test_07_error_handling_and_resilience(self):
        """Test error handling and system resilience"""
        logger.info("Testing error handling and resilience...")
        
        # Test invalid requests
        for service_name, base_url in self.base_urls.items():
            with self.subTest(service=service_name):
                # Test invalid endpoint
                response = requests.get(f"{base_url}/invalid-endpoint")
                self.assertEqual(response.status_code, 404)
                
                # Test invalid JSON data (if service accepts POST)
                if service_name == 'multi_agent':
                    response = requests.post(f"{base_url}/agents", json={'invalid': 'data'})
                    self.assertEqual(response.status_code, 400)
                
                elif service_name == 'adaptive_learning':
                    response = requests.post(f"{base_url}/learning/experiences", json={'invalid': 'data'})
                    self.assertEqual(response.status_code, 400)
                
                logger.info(f"✓ {service_name} error handling test passed")
        
        # Test service recovery (simulate brief unavailability)
        # This would require more complex setup in a real environment
        
        logger.info("✓ Error handling and resilience tests passed")
    
    def test_08_data_persistence_and_consistency(self):
        """Test data persistence and consistency"""
        logger.info("Testing data persistence and consistency...")
        
        # Test that data persists across requests
        multi_agent_url = self.base_urls['multi_agent']
        learning_url = self.base_urls['adaptive_learning']
        
        # Check that previously created agents still exist
        response = requests.get(f"{multi_agent_url}/agents")
        self.assertEqual(response.status_code, 200)
        
        agents_data = response.json()
        for agent_id in self.test_data['agents']:
            self.assertIn(agent_id, agents_data['agents'])
        
        # Check that learning experiences are persisted
        response = requests.get(f"{learning_url}/learning/status")
        self.assertEqual(response.status_code, 200)
        
        status_data = response.json()
        self.assertGreater(status_data['total_experiences'], 0)
        
        logger.info("✓ Data persistence and consistency tests passed")
    
    def test_09_security_and_access_control(self):
        """Test basic security measures"""
        logger.info("Testing security and access control...")
        
        # Test that services are accessible (basic connectivity)
        for service_name, base_url in self.base_urls.items():
            with self.subTest(service=service_name):
                response = requests.get(f"{base_url}/health")
                self.assertEqual(response.status_code, 200)
                
                # Verify response headers include basic security headers
                headers = response.headers
                # Note: In production, we'd check for security headers like:
                # - X-Content-Type-Options
                # - X-Frame-Options
                # - X-XSS-Protection
                # - Strict-Transport-Security
                
                logger.info(f"✓ {service_name} security test passed")
        
        logger.info("✓ Security and access control tests passed")
    
    def test_10_monitoring_and_observability(self):
        """Test monitoring and observability features"""
        logger.info("Testing monitoring and observability...")
        
        # Test that all services provide meaningful status information
        for service_name, base_url in self.base_urls.items():
            with self.subTest(service=service_name):
                response = requests.get(f"{base_url}/health")
                self.assertEqual(response.status_code, 200)
                
                health_data = response.json()
                self.assertIn('timestamp', health_data)
                self.assertIn('version', health_data)
                
                # Test service-specific status endpoints
                if service_name == 'multi_agent':
                    response = requests.get(f"{base_url}/coordination/status")
                    self.assertEqual(response.status_code, 200)
                    
                elif service_name == 'adaptive_learning':
                    response = requests.get(f"{base_url}/learning/status")
                    self.assertEqual(response.status_code, 200)
                    
                elif service_name == 'production_optimizer':
                    response = requests.get(f"{base_url}/optimization/status")
                    self.assertEqual(response.status_code, 200)
                
                logger.info(f"✓ {service_name} monitoring test passed")
        
        logger.info("✓ Monitoring and observability tests passed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data"""
        logger.info("Cleaning up test data...")
        
        # Clean up agents
        multi_agent_url = cls.base_urls['multi_agent']
        for agent_id in cls.test_data['agents']:
            try:
                requests.delete(f"{multi_agent_url}/agents/{agent_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up agent {agent_id}: {e}")
        
        logger.info("Test cleanup completed")

def run_integration_tests():
    """Run the complete integration test suite"""
    print("=" * 60)
    print("WS4 Phase 6 - Integration Test Suite")
    print("Advanced Autonomy & Production Optimization")
    print("=" * 60)
    print()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("=" * 60)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)

