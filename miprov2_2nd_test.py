"""
MIPROv2 Test Program - Agent Management Task with Real LLM Calls
Tests optimization of a multi-agent customer support system
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
from openai import OpenAI
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
client = OpenAI(api_key=OPENAI_API_KEY)

# Cost tracking (GPT-3.5-turbo pricing as of 2024)
COST_PER_1K_INPUT_TOKENS = 0.0005
COST_PER_1K_OUTPUT_TOKENS = 0.0015

# ==================== Dataset ====================

CUSTOMER_SUPPORT_SCENARIOS = [
    {
        "id": 1,
        "type": "technical",
        "query": "My application keeps crashing when I try to export PDFs. Error code: 0x80004005",
        "complexity": "high",
        "expected_skills": ["technical", "debugging"],
        "ground_truth": "This is a Windows COM error. Check permissions, reinstall PDF components, or update the application."
    },
    {
        "id": 2,
        "type": "billing",
        "query": "I was charged twice for my subscription last month. Order #A1234 and #A1235",
        "complexity": "medium",
        "expected_skills": ["billing", "empathy"],
        "ground_truth": "Apologize for the inconvenience, verify the duplicate charge, and process a refund for one charge."
    },
    {
        "id": 3,
        "type": "product",
        "query": "What's the difference between the Pro and Enterprise plans?",
        "complexity": "low",
        "expected_skills": ["product", "sales"],
        "ground_truth": "Pro plan includes X features for small teams, Enterprise adds Y features plus dedicated support and custom contracts."
    },
    {
        "id": 4,
        "type": "technical",
        "query": "How do I integrate your API with my Python application? I need to sync user data.",
        "complexity": "medium",
        "expected_skills": ["technical", "api"],
        "ground_truth": "Use our Python SDK, authenticate with API key, and use the sync_users() method. See documentation link."
    },
    {
        "id": 5,
        "type": "complaint",
        "query": "I've been waiting 3 days for support! This is unacceptable for a paid customer!",
        "complexity": "high",
        "expected_skills": ["empathy", "escalation"],
        "ground_truth": "Sincerely apologize, acknowledge frustration, escalate to senior support, and provide direct contact."
    }
]

# ==================== Agent System ====================

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    role: str
    expertise: List[str]
    response_style: str
    escalation_threshold: float
    max_tokens: int = 150

@dataclass
class AgentMetrics:
    """Metrics for agent performance"""
    agent_id: str
    total_queries: int = 0
    successful_resolutions: int = 0
    total_latency_ms: float = 0.0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    accuracy_scores: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_resolutions / self.total_queries
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries
    
    @property
    def avg_accuracy(self) -> float:
        if not self.accuracy_scores:
            return 0.0
        return np.mean(self.accuracy_scores)

class SupportAgent:
    """Individual support agent with LLM backend"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.metrics = AgentMetrics(agent_id)
        
    def create_prompt(self, query: str, instruction_quality: float) -> str:
        """Create prompt based on instruction quality parameter"""
        
        # Base prompt
        base_prompt = f"You are a {self.config.role} support agent."
        
        # Enhanced prompt based on instruction quality
        if instruction_quality > 0.7:
            enhanced_prompt = f"""You are an expert {self.config.role} support agent with expertise in {', '.join(self.config.expertise)}.

Your response style: {self.config.response_style}

Guidelines:
- Provide clear, actionable solutions
- Match the technical level to the query
- Be concise but complete
- Show empathy when appropriate
- Escalate if confidence < {self.config.escalation_threshold}

Customer Query: {query}

Response:"""
        elif instruction_quality > 0.4:
            enhanced_prompt = f"""{base_prompt}
Expertise: {', '.join(self.config.expertise)}
Style: {self.config.response_style}

Query: {query}

Response:"""
        else:
            enhanced_prompt = f"""{base_prompt}

Query: {query}

Response:"""
        
        return enhanced_prompt
    
    def handle_query(self, query: str, instruction_quality: float) -> Tuple[str, Dict[str, Any]]:
        """Handle a customer query with LLM"""
        
        start_time = time.time()
        metrics = {}
        
        try:
            # Create prompt
            prompt = self.create_prompt(query, instruction_quality)
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support agent."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.7
            )
            
            # Extract response
            agent_response = response.choices[0].message.content
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = response.usage.total_tokens
            cost = (
                response.usage.prompt_tokens * COST_PER_1K_INPUT_TOKENS / 1000 +
                response.usage.completion_tokens * COST_PER_1K_OUTPUT_TOKENS / 1000
            )
            
            metrics = {
                'latency_ms': latency_ms,
                'tokens_used': tokens_used,
                'cost': cost,
                'success': True
            }
            
            # Update agent metrics
            self.metrics.total_queries += 1
            self.metrics.total_latency_ms += latency_ms
            self.metrics.total_tokens_used += tokens_used
            self.metrics.total_cost += cost
            
            return agent_response, metrics
            
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id}: {e}")
            metrics = {
                'latency_ms': (time.time() - start_time) * 1000,
                'tokens_used': 0,
                'cost': 0,
                'success': False,
                'error': str(e)
            }
            return f"Error: {e}", metrics

class MultiAgentSystem:
    """Multi-agent customer support system"""
    
    def __init__(self, agents: List[SupportAgent]):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.routing_metrics = defaultdict(list)
        
    def route_query(self, scenario: Dict, routing_quality: float) -> str:
        """Route query to appropriate agent based on routing quality"""
        
        # Simple routing based on quality parameter
        if routing_quality > 0.7:
            # Smart routing based on expertise match
            best_agent = None
            best_score = -1
            
            for agent_id, agent in self.agents.items():
                # Calculate expertise match
                match_score = len(set(agent.config.expertise) & set(scenario.get('expected_skills', [])))
                if match_score > best_score:
                    best_score = match_score
                    best_agent = agent_id
            
            return best_agent or list(self.agents.keys())[0]
        else:
            # Random or round-robin routing
            return list(self.agents.keys())[hash(scenario['id']) % len(self.agents)]
    
    def process_scenario(
        self,
        scenario: Dict,
        instruction_quality: float,
        routing_quality: float
    ) -> Dict[str, Any]:
        """Process a support scenario"""
        
        # Route to agent
        selected_agent_id = self.route_query(scenario, routing_quality)
        agent = self.agents[selected_agent_id]
        
        # Handle query
        response, metrics = agent.handle_query(scenario['query'], instruction_quality)
        
        # Evaluate response quality (simplified - in reality would use more sophisticated evaluation)
        accuracy = self.evaluate_response(response, scenario.get('ground_truth', ''))
        agent.metrics.accuracy_scores.append(accuracy)
        
        if accuracy > 0.6:
            agent.metrics.successful_resolutions += 1
        
        return {
            'scenario_id': scenario['id'],
            'agent_id': selected_agent_id,
            'response': response,
            'accuracy': accuracy,
            'metrics': metrics
        }
    
    def evaluate_response(self, response: str, ground_truth: str) -> float:
        """Simple evaluation of response quality"""
        if not ground_truth:
            return 0.5
        
        # Check for key concepts (simplified evaluation)
        key_concepts = ground_truth.lower().split()
        response_lower = response.lower()
        
        matches = sum(1 for concept in key_concepts if concept in response_lower)
        accuracy = matches / len(key_concepts) if key_concepts else 0.5
        
        return min(accuracy, 1.0)

# ==================== MIPROv2 Optimizer ====================

class MIPROv2AgentOptimizer:
    """MIPROv2 optimizer for agent management"""
    
    def __init__(
        self,
        system: MultiAgentSystem,
        scenarios: List[Dict],
        total_budget: int = 100
    ):
        self.system = system
        self.scenarios = scenarios
        self.total_budget = total_budget
        self.used_budget = 0
        self.optimization_history = []
        
        # Parameters to optimize
        self.current_params = {
            'instruction_quality': 0.5,
            'routing_quality': 0.5,
            'agent_specialization': 0.5
        }
        
        self.best_params = self.current_params.copy()
        self.best_performance = 0.0
        
    def run_trial(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Run a single optimization trial"""
        
        trial_start = time.time()
        results = []
        
        # Test on subset of scenarios to save API calls
        test_scenarios = np.random.choice(self.scenarios, size=min(3, len(self.scenarios)), replace=False)
        
        for scenario in test_scenarios:
            result = self.system.process_scenario(
                scenario,
                params['instruction_quality'],
                params['routing_quality']
            )
            results.append(result)
        
        # Calculate aggregate metrics
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_latency = np.mean([r['metrics']['latency_ms'] for r in results])
        total_cost = sum(r['metrics']['cost'] for r in results)
        success_rate = np.mean([r['metrics']['success'] for r in results])
        
        # Combined performance score
        performance = (
            0.4 * avg_accuracy +
            0.2 * success_rate +
            0.2 * (1 - min(avg_latency / 1000, 1)) +  # Normalize latency
            0.2 * (1 - min(total_cost, 1))  # Normalize cost
        )
        
        trial_time = time.time() - trial_start
        
        return {
            'params': params,
            'performance': performance,
            'accuracy': avg_accuracy,
            'latency_ms': avg_latency,
            'cost': total_cost,
            'success_rate': success_rate,
            'trial_time': trial_time,
            'results': results
        }
    
    def optimize(self) -> Dict[str, Any]:
        """Run MIPROv2 optimization"""
        
        logger.info("Starting MIPROv2 Agent Optimization")
        logger.info(f"Budget: {self.total_budget} trials")
        
        start_time = time.time()
        
        # Exploration phase (30% of budget)
        exploration_budget = int(self.total_budget * 0.3)
        logger.info(f"\nPhase 1: Exploration ({exploration_budget} trials)")
        
        for i in range(exploration_budget):
            # Random sampling
            params = {
                'instruction_quality': np.random.uniform(0.2, 1.0),
                'routing_quality': np.random.uniform(0.2, 1.0),
                'agent_specialization': np.random.uniform(0.2, 1.0)
            }
            
            trial_result = self.run_trial(params)
            self.optimization_history.append(trial_result)
            
            if trial_result['performance'] > self.best_performance:
                self.best_performance = trial_result['performance']
                self.best_params = params.copy()
                logger.info(f"  Trial {i+1}: New best! Performance: {self.best_performance:.3f}")
            
            self.used_budget += 1
        
        # Exploitation phase (70% of budget)
        exploitation_budget = self.total_budget - self.used_budget
        logger.info(f"\nPhase 2: Exploitation ({exploitation_budget} trials)")
        
        for i in range(exploitation_budget):
            # Gaussian perturbation around best params
            params = {}
            for key, value in self.best_params.items():
                perturbation = np.random.normal(0, 0.1)
                params[key] = np.clip(value + perturbation, 0.2, 1.0)
            
            trial_result = self.run_trial(params)
            self.optimization_history.append(trial_result)
            
            if trial_result['performance'] > self.best_performance:
                self.best_performance = trial_result['performance']
                self.best_params = params.copy()
                logger.info(f"  Trial {self.used_budget+1}: Improved! Performance: {self.best_performance:.3f}")
            
            self.used_budget += 1
            
            # Progress logging
            if (i + 1) % 10 == 0:
                recent_avg = np.mean([h['performance'] for h in self.optimization_history[-10:]])
                logger.info(f"  Progress: {self.used_budget}/{self.total_budget}, Recent avg: {recent_avg:.3f}")
        
        # Final evaluation with best params
        logger.info("\nPhase 3: Final Evaluation")
        final_results = []
        for scenario in self.scenarios:
            result = self.system.process_scenario(
                scenario,
                self.best_params['instruction_quality'],
                self.best_params['routing_quality']
            )
            final_results.append(result)
        
        # Compile results
        total_time = time.time() - start_time
        
        return self.compile_results(final_results, total_time)
    
    def compile_results(self, final_results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Compile optimization results"""
        
        # Calculate final metrics
        final_accuracy = np.mean([r['accuracy'] for r in final_results])
        final_latency = np.mean([r['metrics']['latency_ms'] for r in final_results])
        total_cost = sum(h['cost'] for h in self.optimization_history)
        
        # Agent performance summary
        agent_summary = {}
        for agent_id, agent in self.system.agents.items():
            agent_summary[agent_id] = {
                'success_rate': agent.metrics.success_rate,
                'avg_accuracy': agent.metrics.avg_accuracy,
                'avg_latency_ms': agent.metrics.avg_latency_ms,
                'total_cost': agent.metrics.total_cost,
                'queries_handled': agent.metrics.total_queries
            }
        
        return {
            'optimization_method': 'MIPROv2',
            'best_params': self.best_params,
            'best_performance': self.best_performance,
            'final_accuracy': final_accuracy,
            'final_latency_ms': final_latency,
            'total_cost': total_cost,
            'total_time_seconds': total_time,
            'trials_used': self.used_budget,
            'agent_performance': agent_summary,
            'optimization_history': self.optimization_history[-10:],  # Last 10 trials
            'convergence_data': {
                'performances': [h['performance'] for h in self.optimization_history],
                'costs': [h['cost'] for h in self.optimization_history],
                'latencies': [h['latency_ms'] for h in self.optimization_history]
            }
        }

# ==================== Main Execution ====================

def create_agent_system() -> MultiAgentSystem:
    """Create a multi-agent support system"""
    
    agents = [
        SupportAgent("tech_agent", AgentConfig(
            role="Technical Support Specialist",
            expertise=["technical", "debugging", "api"],
            response_style="detailed and technical",
            escalation_threshold=0.3,
            max_tokens=150
        )),
        SupportAgent("billing_agent", AgentConfig(
            role="Billing Support Representative",
            expertise=["billing", "refunds", "accounts"],
            response_style="empathetic and clear",
            escalation_threshold=0.4,
            max_tokens=120
        )),
        SupportAgent("general_agent", AgentConfig(
            role="General Support Representative",
            expertise=["product", "sales", "general"],
            response_style="friendly and informative",
            escalation_threshold=0.5,
            max_tokens=130
        ))
    ]
    
    return MultiAgentSystem(agents)

def main():
    """Main execution"""
    
    # Check API key
    if OPENAI_API_KEY == "your-api-key-here":
        logger.error("Please set your OpenAI API key in the OPENAI_API_KEY environment variable")
        return
    
    logger.info("=" * 60)
    logger.info("MIPROv2 Agent Management Test")
    logger.info("=" * 60)
    
    # Create system
    system = create_agent_system()
    
    # Create optimizer
    optimizer = MIPROv2AgentOptimizer(
        system=system,
        scenarios=CUSTOMER_SUPPORT_SCENARIOS,
        total_budget=30  # Reduced for cost management
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Print results
    print("\n" + "=" * 60)
    print("MIPROV2 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nBest Parameters Found:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value:.3f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Best Performance Score: {results['best_performance']:.3f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.3f}")
    print(f"  Average Latency: {results['final_latency_ms']:.1f}ms")
    print(f"  Total Cost: ${results['total_cost']:.4f}")
    print(f"  Total Time: {results['total_time_seconds']:.1f}s")
    
    print(f"\nAgent Performance:")
    for agent_id, metrics in results['agent_performance'].items():
        print(f"\n  {agent_id}:")
        print(f"    Success Rate: {metrics['success_rate']:.2%}")
        print(f"    Accuracy: {metrics['avg_accuracy']:.3f}")
        print(f"    Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"    Queries: {metrics['queries_handled']}")
    
    # Save results
    with open('miprov2_agent_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("\nResults saved to miprov2_agent_results.json")

if __name__ == "__main__":
    main()