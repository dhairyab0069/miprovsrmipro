"""
r-MIPRO Test Program - Agent Management Task with Real LLM Calls
Tests resource-adaptive optimization of a multi-agent customer support system
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
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

# Cost tracking (GPT-4o-mini pricing as of 2024)
# GPT-4o-mini is actually cheaper than GPT-3.5-turbo
COST_PER_1K_INPUT_TOKENS = 0.00015  # $0.15 per 1M input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.0006  # $0.60 per 1M output tokens

# ==================== Dataset (Same as MIPROv2) ====================

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

# ==================== Enhanced Agent System for r-MIPRO ====================

@dataclass
class ResourceConstraints:
    """Resource constraints for production deployment"""
    max_latency_ms: float = 2000.0
    max_cost_per_query: float = 0.01
    max_total_cost: float = 1.0
    min_accuracy: float = 0.6
    max_api_calls: int = 200

@dataclass
class EnhancedAgentMetrics:
    """Enhanced metrics with information theory components"""
    agent_id: str
    module_type: str
    
    # Performance tracking
    performance_history: deque = field(default_factory=lambda: deque(maxlen=50))
    accuracy_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Resource tracking
    latency_history: deque = field(default_factory=lambda: deque(maxlen=50))
    cost_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Statistical metrics
    mean_performance: float = 0.0
    variance: float = 1.0
    entropy: float = 1.0
    
    # Optimization metrics
    improvement_velocity: float = 0.0
    time_since_improvement: int = 0
    bottleneck_score: float = 0.5
    
    # Cumulative metrics
    total_queries: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    best_performance: float = 0.0
    
    def update(self, performance: float, accuracy: float, latency_ms: float, cost: float):
        """Update metrics with new observation"""
        self.performance_history.append(performance)
        self.accuracy_history.append(accuracy)
        self.latency_history.append(latency_ms)
        self.cost_history.append(cost)
        
        self.total_queries += 1
        self.total_cost += cost
        self.total_latency_ms += latency_ms
        
        # Update statistical metrics
        if len(self.performance_history) > 2:
            self.mean_performance = np.mean(self.performance_history)
            self.variance = np.var(self.performance_history)
            
            # Calculate entropy
            if self.variance > 0:
                self.entropy = 0.5 * np.log(2 * np.pi * np.e * self.variance)
            
            # Calculate improvement velocity
            recent_improvements = []
            for i in range(1, min(5, len(self.performance_history))):
                recent_improvements.append(
                    self.performance_history[-i] - self.performance_history[-i-1]
                )
            if recent_improvements:
                self.improvement_velocity = np.mean(recent_improvements)
        
        # Track best performance
        if performance > self.best_performance:
            self.best_performance = performance
            self.time_since_improvement = 0
        else:
            self.time_since_improvement += 1

class AdaptiveAgent:
    """Agent with adaptive optimization capabilities"""
    
    def __init__(self, agent_id: str, module_type: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.module_type = module_type
        self.config = config
        self.metrics = EnhancedAgentMetrics(agent_id, module_type)
        
        # Optimization state
        self.current_params = {
            'instruction_detail': 0.5,
            'few_shot_examples': 0,
            'reasoning_depth': 0.5,
            'response_structure': 0.5
        }
        self.best_params = self.current_params.copy()
    
    def generate_adaptive_prompt(self, query: str, params: Dict[str, float]) -> str:
        """Generate prompt based on optimization parameters"""
        
        instruction_detail = params.get('instruction_detail', 0.5)
        few_shot_examples = int(params.get('few_shot_examples', 0))
        reasoning_depth = params.get('reasoning_depth', 0.5)
        response_structure = params.get('response_structure', 0.5)
        
        # Base instruction
        prompt_parts = [f"You are a {self.config['role']} with expertise in {', '.join(self.config['expertise'])}."]
        
        # Add detail based on instruction quality
        if instruction_detail > 0.7:
            prompt_parts.append(f"""
Your approach:
1. Analyze the customer's issue carefully
2. Consider the emotional context
3. Provide a solution that addresses both technical and emotional needs
4. Be {self.config['response_style']}
5. Escalate if confidence is below {self.config.get('escalation_threshold', 0.5)}
""")
        
        # Add few-shot examples if specified
        if few_shot_examples > 0:
            prompt_parts.append("\nExamples:")
            example_scenarios = [
                ("How do I reset my password?", 
                 "I'll help you reset your password. Please go to the login page, click 'Forgot Password', and follow the email instructions."),
                ("Your service is terrible!", 
                 "I sincerely apologize for your frustration. Let me immediately look into this issue and find a solution for you.")
            ]
            for i in range(min(few_shot_examples, len(example_scenarios))):
                q, a = example_scenarios[i]
                prompt_parts.append(f"Q: {q}\nA: {a}")
        
        # Add reasoning instruction based on depth
        if reasoning_depth > 0.6:
            prompt_parts.append("\nThink step-by-step about the best solution before responding.")
        
        # Add structure requirements
        if response_structure > 0.7:
            prompt_parts.append("\nStructure your response with: 1) Acknowledgment 2) Solution 3) Next steps")
        
        # Add the actual query
        prompt_parts.append(f"\nCustomer Query: {query}\n\nResponse:")
        
        return "\n".join(prompt_parts)
    
    def process_query(self, query: str, params: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
        """Process query with adaptive parameters"""
        
        start_time = time.time()
        
        try:
            # Generate adaptive prompt
            prompt = self.generate_adaptive_prompt(query, params)
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini which is cheaper and better than gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are a professional customer support agent."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            # Extract response and metrics
            agent_response = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = response.usage.total_tokens
            cost = (
                response.usage.prompt_tokens * COST_PER_1K_INPUT_TOKENS / 1000 +
                response.usage.completion_tokens * COST_PER_1K_OUTPUT_TOKENS / 1000
            )
            
            return agent_response, {
                'latency_ms': latency_ms,
                'tokens_used': tokens_used,
                'cost': cost,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id}: {e}")
            return f"Error: {e}", {
                'latency_ms': (time.time() - start_time) * 1000,
                'tokens_used': 0,
                'cost': 0,
                'success': False,
                'error': str(e)
            }

# ==================== r-MIPRO Optimizer ====================

class AdaptiveScheduler:
    """Intelligent scheduler for resource allocation"""
    
    def __init__(self, exploration_weight: float = 0.3):
        self.exploration_weight = exploration_weight
        self.allocation_history = []
        
    def compute_allocation_scores(
        self,
        agent_metrics: Dict[str, EnhancedAgentMetrics],
        constraints: ResourceConstraints
    ) -> Dict[str, float]:
        """Compute allocation scores using multiple strategies"""
        
        scores = {}
        
        for agent_id, metrics in agent_metrics.items():
            # Entropy-based exploration (high uncertainty = more exploration)
            entropy_score = metrics.entropy
            
            # Performance gradient (improving agents get more resources)
            gradient_score = max(0, metrics.improvement_velocity)
            
            # Bottleneck score (underperforming agents need help)
            performances = [m.mean_performance for m in agent_metrics.values()]
            if performances:
                min_perf = min(performances)
                max_perf = max(performances)
                if max_perf > min_perf:
                    bottleneck = 1 - (metrics.mean_performance - min_perf) / (max_perf - min_perf)
                else:
                    bottleneck = 0.5
                metrics.bottleneck_score = bottleneck
            else:
                bottleneck = 0.5
            
            # Convergence penalty (converged agents get fewer resources)
            convergence_penalty = 1 / (1 + metrics.time_since_improvement * 0.1)
            
            # Resource efficiency (penalize high-cost agents)
            if metrics.total_queries > 0:
                avg_cost = metrics.total_cost / metrics.total_queries
                cost_penalty = 1 - min(avg_cost / constraints.max_cost_per_query, 1)
            else:
                cost_penalty = 1.0
            
            # Combined score
            scores[agent_id] = max(0, (  # Ensure non-negative
                0.2 * entropy_score +
                0.2 * gradient_score +
                0.3 * bottleneck +
                0.2 * convergence_penalty +
                0.1 * cost_penalty
            ))
        
        return scores
    
    def select_agent(
        self,
        agent_metrics: Dict[str, EnhancedAgentMetrics],
        constraints: ResourceConstraints
    ) -> Optional[str]:
        """Select next agent to optimize"""
        
        # Check resource constraints
        total_cost = sum(m.total_cost for m in agent_metrics.values())
        if total_cost >= constraints.max_total_cost:
            return None
        
        # Compute scores
        scores = self.compute_allocation_scores(agent_metrics, constraints)
        
        # Filter out converged agents
        active_agents = {
            agent_id: score
            for agent_id, score in scores.items()
            if agent_metrics[agent_id].time_since_improvement < 20
        }
        
        if not active_agents:
            return None
        
        # Weighted selection
        agents = list(active_agents.keys())
        weights = np.array(list(active_agents.values()))
        
        # Ensure weights are non-negative
        weights = np.maximum(weights, 0)
        
        # Handle case where all weights are zero
        if weights.sum() == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / weights.sum()
        
        selected = np.random.choice(agents, p=weights)
        
        self.allocation_history.append({
            'selected': selected,
            'scores': scores,
            'timestamp': time.time()
        })
        
        return selected

class RMIPROAgentOptimizer:
    """r-MIPRO optimizer for agent management"""
    
    def __init__(
        self,
        agents: List[AdaptiveAgent],
        scenarios: List[Dict],
        total_budget: int = 100,
        constraints: ResourceConstraints = None
    ):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.scenarios = scenarios
        self.total_budget = total_budget
        self.constraints = constraints or ResourceConstraints()
        self.scheduler = AdaptiveScheduler()
        
        self.used_budget = 0
        self.optimization_history = []
        
    def generate_optimized_params(self, agent: AdaptiveAgent) -> Dict[str, float]:
        """Generate optimized parameters for an agent"""
        
        # Use agent's performance history to guide parameter generation
        if len(agent.metrics.performance_history) > 5:
            # Exploit best known parameters with small perturbations
            base_params = agent.best_params.copy()
            params = {}
            for key, value in base_params.items():
                if key == 'few_shot_examples':
                    # Integer parameter
                    params[key] = max(0, min(3, value + np.random.randint(-1, 2)))
                else:
                    # Continuous parameter
                    perturbation = np.random.normal(0, 0.1)
                    params[key] = np.clip(value + perturbation, 0.0, 1.0)
        else:
            # Explore with random parameters
            params = {
                'instruction_detail': np.random.uniform(0.3, 1.0),
                'few_shot_examples': np.random.randint(0, 3),
                'reasoning_depth': np.random.uniform(0.2, 1.0),
                'response_structure': np.random.uniform(0.2, 1.0)
            }
        
        return params
    
    def evaluate_agent_performance(
        self,
        agent: AdaptiveAgent,
        params: Dict[str, float],
        scenario: Dict
    ) -> Dict[str, Any]:
        """Evaluate agent with given parameters on a scenario"""
        
        # Process query
        response, metrics = agent.process_query(scenario['query'], params)
        
        # Evaluate accuracy
        ground_truth = scenario.get('ground_truth', '')
        key_concepts = ground_truth.lower().split()
        response_lower = response.lower()
        
        matches = sum(1 for concept in key_concepts if concept in response_lower)
        accuracy = matches / len(key_concepts) if key_concepts else 0.5
        
        # Calculate performance score
        performance = (
            0.5 * accuracy +
            0.2 * metrics['success'] +
            0.15 * (1 - min(metrics['latency_ms'] / 2000, 1)) +
            0.15 * (1 - min(metrics['cost'] / 0.01, 1))
        )
        
        return {
            'agent_id': agent.agent_id,
            'scenario_id': scenario['id'],
            'params': params,
            'response': response,
            'accuracy': accuracy,
            'performance': performance,
            'metrics': metrics
        }
    
    def optimization_step(self) -> bool:
        """Single optimization step with adaptive resource allocation"""
        
        if self.used_budget >= self.total_budget:
            return False
        
        # Select agent to optimize
        agent_metrics = {
            agent_id: agent.metrics
            for agent_id, agent in self.agents.items()
        }
        
        selected_agent_id = self.scheduler.select_agent(agent_metrics, self.constraints)
        
        if not selected_agent_id:
            return False
        
        agent = self.agents[selected_agent_id]
        
        # Generate optimized parameters
        params = self.generate_optimized_params(agent)
        
        # Select scenario (prefer challenging ones)
        if agent.metrics.total_queries > 0:
            # Select scenarios where agent performs poorly
            scenario_weights = []
            for scenario in self.scenarios:
                # Higher weight for complex scenarios
                weight = 1.0 if scenario['complexity'] == 'high' else 0.5
                scenario_weights.append(weight)
            scenario_weights = np.array(scenario_weights) / sum(scenario_weights)
            scenario = np.random.choice(self.scenarios, p=scenario_weights)
        else:
            scenario = np.random.choice(self.scenarios)
        
        # Evaluate
        result = self.evaluate_agent_performance(agent, params, scenario)
        
        # Update agent metrics
        agent.metrics.update(
            result['performance'],
            result['accuracy'],
            result['metrics']['latency_ms'],
            result['metrics']['cost']
        )
        
        # Update best params if improved
        if result['performance'] > agent.metrics.best_performance:
            agent.best_params = params.copy()
            logger.info(
                f"  Agent {selected_agent_id} improved: {agent.metrics.best_performance:.3f} "
                f"(accuracy: {result['accuracy']:.2f}, latency: {result['metrics']['latency_ms']:.0f}ms)"
            )
        
        # Log optimization step
        self.optimization_history.append(result)
        self.used_budget += 1
        
        return True
    
    def run_exploration_phase(self):
        """Smart exploration phase"""
        exploration_budget = int(self.total_budget * 0.2)
        logger.info(f"Phase 1: Intelligent Exploration ({exploration_budget} trials)")
        
        for _ in range(exploration_budget):
            if self.used_budget >= exploration_budget:
                break
            
            # Round-robin through agents for initial exploration
            for agent in self.agents.values():
                if self.used_budget >= exploration_budget:
                    break
                
                params = self.generate_optimized_params(agent)
                scenario = np.random.choice(self.scenarios)
                result = self.evaluate_agent_performance(agent, params, scenario)
                
                agent.metrics.update(
                    result['performance'],
                    result['accuracy'],
                    result['metrics']['latency_ms'],
                    result['metrics']['cost']
                )
                
                self.optimization_history.append(result)
                self.used_budget += 1
    
    def optimize(self) -> Dict[str, Any]:
        """Run r-MIPRO optimization"""
        
        logger.info("=" * 60)
        logger.info("Starting r-MIPRO Agent Optimization")
        logger.info(f"Budget: {self.total_budget} trials")
        logger.info(f"Constraints: max_cost=${self.constraints.max_total_cost}, "
                   f"max_latency={self.constraints.max_latency_ms}ms")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Exploration phase
        self.run_exploration_phase()
        
        # Adaptive optimization phase
        logger.info(f"\nPhase 2: Adaptive Optimization ({self.total_budget - self.used_budget} trials)")
        
        while self.used_budget < self.total_budget:
            if not self.optimization_step():
                logger.info("Stopping: constraints reached or all agents converged")
                break
            
            # Progress logging
            if self.used_budget % 10 == 0:
                total_cost = sum(agent.metrics.total_cost for agent in self.agents.values())
                avg_performance = np.mean([
                    agent.metrics.mean_performance 
                    for agent in self.agents.values()
                ])
                logger.info(
                    f"  Progress: {self.used_budget}/{self.total_budget}, "
                    f"Avg Performance: {avg_performance:.3f}, "
                    f"Total Cost: ${total_cost:.3f}"
                )
        
        # Final evaluation
        logger.info("\nPhase 3: Final Evaluation")
        final_results = []
        for scenario in self.scenarios:
            # Use best agent for each scenario type
            best_agent = self.select_best_agent_for_scenario(scenario)
            result = self.evaluate_agent_performance(
                best_agent,
                best_agent.best_params,
                scenario
            )
            final_results.append(result)
        
        # Compile results
        total_time = time.time() - start_time
        return self.compile_results(final_results, total_time)
    
    def select_best_agent_for_scenario(self, scenario: Dict) -> AdaptiveAgent:
        """Select best agent for a scenario based on expertise match"""
        best_agent = None
        best_score = -1
        
        for agent in self.agents.values():
            # Calculate expertise match
            expertise_match = len(
                set(agent.config['expertise']) & set(scenario.get('expected_skills', []))
            )
            # Weight by agent performance
            score = expertise_match * agent.metrics.mean_performance
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent or list(self.agents.values())[0]
    
    def compile_results(self, final_results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Compile optimization results"""
        
        # Calculate final metrics
        final_accuracy = np.mean([r['accuracy'] for r in final_results])
        final_latency = np.mean([r['metrics']['latency_ms'] for r in final_results])
        total_cost = sum(agent.metrics.total_cost for agent in self.agents.values())
        
        # Agent performance summary
        agent_summary = {}
        for agent_id, agent in self.agents.items():
            if agent.metrics.total_queries > 0:
                agent_summary[agent_id] = {
                    'best_performance': agent.metrics.best_performance,
                    'mean_performance': agent.metrics.mean_performance,
                    'variance': agent.metrics.variance,
                    'entropy': agent.metrics.entropy,
                    'bottleneck_score': agent.metrics.bottleneck_score,
                    'improvement_velocity': agent.metrics.improvement_velocity,
                    'total_queries': agent.metrics.total_queries,
                    'total_cost': agent.metrics.total_cost,
                    'avg_latency_ms': agent.metrics.total_latency_ms / agent.metrics.total_queries,
                    'best_params': agent.best_params
                }
        
        # Resource allocation analysis
        allocation_summary = {}
        for record in self.scheduler.allocation_history:
            agent = record['selected']
            allocation_summary[agent] = allocation_summary.get(agent, 0) + 1
        
        return {
            'optimization_method': 'r-MIPRO',
            'final_accuracy': final_accuracy,
            'final_latency_ms': final_latency,
            'total_cost': total_cost,
            'total_time_seconds': total_time,
            'trials_used': self.used_budget,
            'agent_performance': agent_summary,
            'resource_allocation': allocation_summary,
            'optimization_history': self.optimization_history[-10:],
            'convergence_data': {
                'performances': [h['performance'] for h in self.optimization_history],
                'accuracies': [h['accuracy'] for h in self.optimization_history],
                'costs': [h['metrics']['cost'] for h in self.optimization_history],
                'latencies': [h['metrics']['latency_ms'] for h in self.optimization_history]
            }
        }

# ==================== Main Execution ====================

def create_adaptive_agents() -> List[AdaptiveAgent]:
    """Create adaptive agents for r-MIPRO"""
    
    agents = [
        AdaptiveAgent("tech_specialist", "technical", {
            'role': "Senior Technical Support Engineer",
            'expertise': ["technical", "debugging", "api", "integration"],
            'response_style': "precise and solution-focused",
            'escalation_threshold': 0.3
        }),
        AdaptiveAgent("billing_expert", "billing", {
            'role': "Billing and Account Specialist",
            'expertise': ["billing", "refunds", "accounts", "empathy"],
            'response_style': "empathetic and clear",
            'escalation_threshold': 0.4
        }),
        AdaptiveAgent("product_advisor", "product", {
            'role': "Product Knowledge Expert",
            'expertise': ["product", "sales", "features", "comparison"],
            'response_style': "informative and consultative",
            'escalation_threshold': 0.5
        }),
        AdaptiveAgent("escalation_handler", "escalation", {
            'role': "Senior Support Manager",
            'expertise': ["empathy", "escalation", "resolution", "retention"],
            'response_style': "professional and solution-oriented",
            'escalation_threshold': 0.2
        })
    ]
    
    return agents

def main():
    """Main execution"""
    
    # Check API key
    if OPENAI_API_KEY == "your-api-key-here":
        logger.error("Please set your OpenAI API key in the OPENAI_API_KEY environment variable")
        return
    
    logger.info("=" * 60)
    logger.info("r-MIPRO Agent Management Test")
    logger.info("=" * 60)
    
    # Create adaptive agents
    agents = create_adaptive_agents()
    
    # Create optimizer with constraints
    optimizer = RMIPROAgentOptimizer(
        agents=agents,
        scenarios=CUSTOMER_SUPPORT_SCENARIOS,
        total_budget=30,  # Reduced for cost management
        constraints=ResourceConstraints(
            max_latency_ms=2000,
            max_cost_per_query=0.01,
            max_total_cost=1.0,
            min_accuracy=0.6
        )
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Print results
    print("\n" + "=" * 60)
    print("R-MIPRO OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nPerformance Metrics:")
    print(f"  Final Accuracy: {results['final_accuracy']:.3f}")
    print(f"  Average Latency: {results['final_latency_ms']:.1f}ms")
    print(f"  Total Cost: ${results['total_cost']:.4f}")
    print(f"  Total Time: {results['total_time_seconds']:.1f}s")
    print(f"  Trials Used: {results['trials_used']}")
    
    print(f"\nAgent Performance & Resource Allocation:")
    for agent_id, metrics in results['agent_performance'].items():
        allocations = results['resource_allocation'].get(agent_id, 0)
        print(f"\n  {agent_id}:")
        print(f"    Best Performance: {metrics['best_performance']:.3f}")
        print(f"    Mean Performance: {metrics['mean_performance']:.3f}")
        print(f"    Entropy: {metrics['entropy']:.3f}")
        print(f"    Bottleneck Score: {metrics['bottleneck_score']:.3f}")
        print(f"    Queries Handled: {metrics['total_queries']}")
        print(f"    Resource Allocations: {allocations} ({allocations/results['trials_used']*100:.1f}%)")
        print(f"    Best Parameters:")
        for param, value in metrics['best_params'].items():
            print(f"      {param}: {value:.3f}")
    
    # Save results
    with open('r_mipro_agent_results.json', 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, deque):
                return list(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("\nResults saved to r_mipro_agent_results.json")
    
    # Print comparison insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("\n1. Resource Allocation:")
    print("   r-MIPRO dynamically allocated resources based on agent performance,")
    print("   bottleneck scores, and improvement potential.")
    
    print("\n2. Parameter Optimization:")
    print("   Each agent developed unique optimal parameters based on their role")
    print("   and the types of queries they handled.")
    
    print("\n3. Cost Efficiency:")
    print(f"   Total optimization cost: ${results['total_cost']:.4f}")
    print(f"   Average cost per trial: ${results['total_cost']/results['trials_used']:.4f}")

if __name__ == "__main__":
    main()