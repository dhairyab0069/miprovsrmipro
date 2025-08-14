"""
r-MIPRO: Resource-Adaptive MIPRO Implementation
Advanced resource-adaptive optimization for industrial AI pipelines
Based on theoretical framework combining bandit algorithms, information theory, and bounded rationality
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import math
from abc import ABC, abstractmethod
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Theoretical Foundation ====================

class OptimizationPolicy(Enum):
    """Optimization policies with theoretical grounding"""
    ENTROPY_GUIDED = "entropy_guided"  # Information theory based
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian approach
    EXP3 = "exp3"  # Adversarial bandit
    GRADIENT_BANDIT = "gradient_bandit"  # Gradient-based exploration
    HYBRID_ADAPTIVE = "hybrid_adaptive"  # Combines multiple strategies

@dataclass
class InformationMetrics:
    """Information-theoretic metrics for optimization decisions"""
    entropy: float = 1.0
    mutual_information: float = 0.0
    kl_divergence: float = 0.0
    fisher_information: float = 0.0
    
    def compute_information_gain(self, new_observation: float, history: List[float]) -> float:
        """Compute information gain from new observation"""
        if len(history) < 2:
            return 1.0
        
        # Estimate entropy reduction
        old_variance = np.var(history) if history else 1.0
        new_history = history + [new_observation]
        new_variance = np.var(new_history)
        
        # Information gain as entropy reduction
        old_entropy = 0.5 * np.log(2 * np.pi * np.e * max(old_variance, 1e-6))
        new_entropy = 0.5 * np.log(2 * np.pi * np.e * max(new_variance, 1e-6))
        
        return max(0, old_entropy - new_entropy)

@dataclass
class ResourceConstraints:
    """Industrial deployment constraints"""
    max_latency_ms: float = 1000.0
    max_memory_mb: float = 512.0
    max_api_calls: int = 10000
    cost_per_call: float = 0.001
    max_cost: float = 10.0
    parallel_capacity: int = 4
    
    def check_constraints(self, current_usage: Dict[str, float]) -> bool:
        """Check if constraints are satisfied"""
        if current_usage.get('latency_ms', 0) > self.max_latency_ms:
            return False
        if current_usage.get('memory_mb', 0) > self.max_memory_mb:
            return False
        if current_usage.get('api_calls', 0) > self.max_api_calls:
            return False
        if current_usage.get('total_cost', 0) > self.max_cost:
            return False
        return True

# ==================== Advanced Module Metrics ====================

@dataclass
class AdvancedModuleMetrics:
    """Enhanced metrics with theoretical foundations"""
    module_id: str
    
    # Performance metrics
    current_performance: float = 0.0
    best_performance: float = 0.0
    expected_performance: float = 0.0
    
    # Statistical metrics
    mean: float = 0.0
    variance: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    # Information metrics
    info_metrics: InformationMetrics = field(default_factory=InformationMetrics)
    
    # Resource usage
    total_resources_used: int = 0
    average_latency_ms: float = 0.0
    average_memory_mb: float = 0.0
    
    # Optimization state
    improvement_velocity: float = 0.0
    improvement_acceleration: float = 0.0
    time_since_improvement: int = 0
    exploration_bonus: float = 1.0
    
    # History tracking
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    gradient_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def update_metrics(self, performance: float, resources_used: Dict[str, float]):
        """Update all metrics with new observation"""
        self.performance_history.append(performance)
        self.current_performance = performance
        self.best_performance = max(self.best_performance, performance)
        self.total_resources_used += 1
        
        # Update statistical metrics
        if len(self.performance_history) > 1:
            self.mean = np.mean(self.performance_history)
            self.variance = np.var(self.performance_history)
            
            # Confidence interval (95%)
            std_error = np.sqrt(self.variance / len(self.performance_history))
            self.confidence_interval = (
                self.mean - 1.96 * std_error,
                self.mean + 1.96 * std_error
            )
            
            # Gradient calculation
            if len(self.performance_history) > 2:
                recent_gradient = performance - self.performance_history[-2]
                self.gradient_history.append(recent_gradient)
                
                # Velocity and acceleration
                self.improvement_velocity = np.mean(list(self.gradient_history)[-5:])
                if len(self.gradient_history) > 2:
                    self.improvement_acceleration = (
                        self.gradient_history[-1] - self.gradient_history[-2]
                    )
        
        # Update resource metrics
        self.average_latency_ms = (
            self.average_latency_ms * (self.total_resources_used - 1) + 
            resources_used.get('latency_ms', 0)
        ) / self.total_resources_used
        
        self.average_memory_mb = (
            self.average_memory_mb * (self.total_resources_used - 1) + 
            resources_used.get('memory_mb', 0)
        ) / self.total_resources_used
        
        # Update improvement tracking
        if performance > self.best_performance - 0.001:
            self.time_since_improvement = 0
        else:
            self.time_since_improvement += 1
        
        # Information gain
        if len(self.performance_history) > 1:
            self.info_metrics.entropy = -np.sum([
                p * np.log(p + 1e-10) for p in 
                np.histogram(self.performance_history, bins=10, density=True)[0] / 10
                if p > 0
            ])

# ==================== Intelligent Resource Scheduler ====================

class IntelligentScheduler:
    """Advanced scheduler with multiple theoretical approaches"""
    
    def __init__(
        self,
        policy: OptimizationPolicy = OptimizationPolicy.HYBRID_ADAPTIVE,
        exploration_decay: float = 0.995,
        temperature: float = 1.0
    ):
        self.policy = policy
        self.exploration_decay = exploration_decay
        self.temperature = temperature
        self.allocation_history = []
        self.policy_weights = {
            'entropy': 0.3,
            'thompson': 0.2,
            'gradient': 0.2,
            'bottleneck': 0.3
        }
        
        # Thompson sampling parameters
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)
        
        # EXP3 parameters
        self.exp3_weights = defaultdict(lambda: 1.0)
        self.gamma = 0.1
        
    def compute_priority_scores(
        self,
        metrics: Dict[str, AdvancedModuleMetrics],
        constraints: ResourceConstraints
    ) -> Dict[str, float]:
        """Compute priority scores using selected policy"""
        
        if self.policy == OptimizationPolicy.ENTROPY_GUIDED:
            return self._entropy_guided_scores(metrics)
        elif self.policy == OptimizationPolicy.THOMPSON_SAMPLING:
            return self._thompson_sampling_scores(metrics)
        elif self.policy == OptimizationPolicy.EXP3:
            return self._exp3_scores(metrics)
        elif self.policy == OptimizationPolicy.GRADIENT_BANDIT:
            return self._gradient_bandit_scores(metrics)
        elif self.policy == OptimizationPolicy.HYBRID_ADAPTIVE:
            return self._hybrid_adaptive_scores(metrics, constraints)
        else:
            return {m_id: 1.0 for m_id in metrics}
    
    def _entropy_guided_scores(self, metrics: Dict[str, AdvancedModuleMetrics]) -> Dict[str, float]:
        """Information theory based scoring"""
        scores = {}
        
        for m_id, m in metrics.items():
            # High entropy = high uncertainty = more exploration needed
            entropy_score = m.info_metrics.entropy
            
            # Potential for information gain
            variance_score = np.sqrt(m.variance)
            
            # Time-based exploration bonus
            staleness = np.log1p(m.time_since_improvement)
            
            scores[m_id] = (
                0.5 * entropy_score +
                0.3 * variance_score +
                0.2 * staleness
            ) * m.exploration_bonus
        
        return scores
    
    def _thompson_sampling_scores(self, metrics: Dict[str, AdvancedModuleMetrics]) -> Dict[str, float]:
        """Bayesian approach with Beta distribution"""
        scores = {}
        
        for m_id, m in metrics.items():
            # Update Beta parameters based on performance
            if m.total_resources_used > 0:
                successes = sum(1 for p in m.performance_history if p > m.mean)
                failures = len(m.performance_history) - successes
                self.alpha[m_id] = 1 + successes
                self.beta[m_id] = 1 + failures
            
            # Sample from Beta distribution
            scores[m_id] = np.random.beta(self.alpha[m_id], self.beta[m_id])
        
        return scores
    
    def _exp3_scores(self, metrics: Dict[str, AdvancedModuleMetrics]) -> Dict[str, float]:
        """Adversarial bandit approach"""
        scores = {}
        total_weight = sum(self.exp3_weights.values())
        
        for m_id, m in metrics.items():
            # Compute probability with exploration
            prob = (1 - self.gamma) * (self.exp3_weights[m_id] / total_weight) + \
                   self.gamma / len(metrics)
            scores[m_id] = prob
            
            # Update weights based on performance
            if m.total_resources_used > 0:
                reward = m.current_performance
                estimated_reward = reward / prob
                self.exp3_weights[m_id] *= np.exp(self.gamma * estimated_reward / len(metrics))
        
        return scores
    
    def _gradient_bandit_scores(self, metrics: Dict[str, AdvancedModuleMetrics]) -> Dict[str, float]:
        """Gradient-based exploration"""
        scores = {}
        
        for m_id, m in metrics.items():
            # Base score from gradient
            gradient_score = max(0, m.improvement_velocity)
            
            # Acceleration bonus for improving modules
            accel_bonus = max(0, m.improvement_acceleration)
            
            # Exploration based on confidence interval width
            ci_width = m.confidence_interval[1] - m.confidence_interval[0]
            exploration = ci_width * self.temperature
            
            scores[m_id] = gradient_score + 0.5 * accel_bonus + exploration
        
        return scores
    
    def _hybrid_adaptive_scores(
        self,
        metrics: Dict[str, AdvancedModuleMetrics],
        constraints: ResourceConstraints
    ) -> Dict[str, float]:
        """Sophisticated hybrid approach combining multiple strategies"""
        
        # Get scores from different policies
        entropy_scores = self._entropy_guided_scores(metrics)
        thompson_scores = self._thompson_sampling_scores(metrics)
        gradient_scores = self._gradient_bandit_scores(metrics)
        
        # Compute bottleneck scores
        bottleneck_scores = self._compute_bottleneck_scores(metrics)
        
        # Combine scores with adaptive weighting
        combined_scores = {}
        for m_id in metrics:
            combined_scores[m_id] = (
                self.policy_weights['entropy'] * entropy_scores.get(m_id, 0) +
                self.policy_weights['thompson'] * thompson_scores.get(m_id, 0) +
                self.policy_weights['gradient'] * gradient_scores.get(m_id, 0) +
                self.policy_weights['bottleneck'] * bottleneck_scores.get(m_id, 0)
            )
            
            # Adjust for resource constraints
            m = metrics[m_id]
            if m.average_latency_ms > constraints.max_latency_ms * 0.8:
                combined_scores[m_id] *= 0.5  # Penalize high-latency modules
            if m.average_memory_mb > constraints.max_memory_mb * 0.8:
                combined_scores[m_id] *= 0.7  # Penalize high-memory modules
        
        # Adapt policy weights based on performance
        self._adapt_policy_weights(metrics)
        
        return combined_scores
    
    def _compute_bottleneck_scores(self, metrics: Dict[str, AdvancedModuleMetrics]) -> Dict[str, float]:
        """Identify and score bottleneck modules"""
        scores = {}
        
        if not metrics:
            return scores
        
        # Find performance range
        performances = [m.current_performance for m in metrics.values()]
        min_perf = min(performances)
        max_perf = max(performances)
        
        for m_id, m in metrics.items():
            if max_perf > min_perf:
                # Lower performance = higher bottleneck score
                bottleneck = 1 - (m.current_performance - min_perf) / (max_perf - min_perf)
            else:
                bottleneck = 0.5
            
            # Adjust for convergence
            convergence_penalty = 1 / (1 + m.time_since_improvement)
            scores[m_id] = bottleneck * convergence_penalty
        
        return scores
    
    def _adapt_policy_weights(self, metrics: Dict[str, AdvancedModuleMetrics]):
        """Dynamically adapt policy weights based on performance"""
        # Calculate which policy would have made the best decisions
        if len(self.allocation_history) > 10:
            recent_improvements = defaultdict(list)
            
            for m_id, m in metrics.items():
                if len(m.performance_history) > 1:
                    improvement = m.performance_history[-1] - m.performance_history[-2]
                    recent_improvements[m_id].append(improvement)
            
            # Slightly adjust weights based on recent success
            # (This is simplified - could use more sophisticated meta-learning)
            total_improvement = sum(sum(imps) for imps in recent_improvements.values())
            if total_improvement > 0:
                # Increase weight of successful strategies
                self.policy_weights['gradient'] = min(0.4, self.policy_weights['gradient'] + 0.01)
                self.policy_weights['entropy'] = max(0.2, self.policy_weights['entropy'] - 0.01)
    
    def select_module(
        self,
        metrics: Dict[str, AdvancedModuleMetrics],
        remaining_budget: int,
        constraints: ResourceConstraints
    ) -> Optional[str]:
        """Select next module to optimize"""
        
        if remaining_budget <= 0:
            return None
        
        # Compute priority scores
        scores = self.compute_priority_scores(metrics, constraints)
        
        # Filter out converged modules
        active_scores = {}
        for m_id, score in scores.items():
            m = metrics[m_id]
            
            # Check convergence criteria
            if m.time_since_improvement < 50 and m.variance > 1e-6:
                # Check resource constraints
                if constraints.check_constraints({
                    'latency_ms': m.average_latency_ms,
                    'memory_mb': m.average_memory_mb,
                    'api_calls': m.total_resources_used + 1,
                    'total_cost': (m.total_resources_used + 1) * constraints.cost_per_call
                }):
                    active_scores[m_id] = score
        
        if not active_scores:
            return None
        
        # Weighted selection with temperature
        modules = list(active_scores.keys())
        weights = np.array(list(active_scores.values()))
        
        # FIX: Ensure weights are positive before power operation
        # Add small epsilon to avoid zero weights
        weights = np.maximum(weights, 1e-10)
        
        # Apply temperature for exploration control
        # Use safe power operation
        if self.temperature > 0:
            weights = np.power(weights, 1 / max(self.temperature, 0.01))
        
        # Normalize weights
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback to uniform distribution if all weights are zero
            weights = np.ones(len(weights)) / len(weights)
        
        # Additional safety check for NaN
        if np.any(np.isnan(weights)):
            logger.warning("NaN weights detected, using uniform distribution")
            weights = np.ones(len(weights)) / len(weights)
        
        selected = np.random.choice(modules, p=weights)
        
        # Update temperature (exploration decay)
        self.temperature *= self.exploration_decay
        
        # Record selection
        self.allocation_history.append({
            'module': selected,
            'scores': dict(scores),
            'timestamp': time.time()
        })
        
        return selected

# ==================== Industrial Pipeline Components ====================

class IndustrialPipelineModule(ABC):
    """Enhanced pipeline module for industrial applications"""
    
    def __init__(
        self,
        module_id: str,
        module_type: str,
        complexity: float = 1.0,
        resource_profile: Dict[str, float] = None
    ):
        self.module_id = module_id
        self.module_type = module_type
        self.complexity = complexity
        self.resource_profile = resource_profile or {
            'base_latency_ms': 100,
            'base_memory_mb': 50,
            'latency_variance': 20,
            'memory_variance': 10
        }
        self.optimization_state = {}
    
    @abstractmethod
    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate module and return (performance, resource_usage)"""
        pass
    
    @abstractmethod
    def generate_candidate(self, guided: bool = False) -> Dict[str, Any]:
        """Generate candidate configuration"""
        pass
    
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate resource usage"""
        # FIX: Ensure resource values are always positive
        latency = max(0, self.resource_profile['base_latency_ms'] + 
                      np.random.normal(0, self.resource_profile['latency_variance']))
        memory = max(0, self.resource_profile['base_memory_mb'] + 
                     np.random.normal(0, self.resource_profile['memory_variance']))
        
        return {
            'latency_ms': latency,
            'memory_mb': memory
        }

class TextGraphExtractionModule(IndustrialPipelineModule):
    """Combined text and graph extraction module"""
    
    def __init__(self, module_id: str):
        super().__init__(
            module_id=module_id,
            module_type="text_graph_extraction",
            complexity=1.2,
            resource_profile={
                'base_latency_ms': 150,
                'base_memory_mb': 100,
                'latency_variance': 30,
                'memory_variance': 20
            }
        )
    
    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate extraction performance"""
        # Simulate complex extraction task
        entity_precision = config.get('entity_precision', 0.5)
        relation_recall = config.get('relation_recall', 0.5)
        graph_coherence = config.get('graph_coherence', 0.5)
        semantic_depth = config.get('semantic_depth', 0.5)
        
        # Performance model
        performance = (
            0.3 * entity_precision +
            0.3 * relation_recall +
            0.2 * graph_coherence +
            0.2 * semantic_depth
        ) * self.complexity
        
        # Add realistic noise
        performance += np.random.normal(0, 0.05)
        performance = np.clip(performance, 0, 1)
        
        # Resource usage
        resources = self.estimate_resources()
        resources['latency_ms'] *= (1 + 0.5 * semantic_depth)  # Deeper analysis takes longer
        
        return performance, resources
    
    def generate_candidate(self, guided: bool = False) -> Dict[str, Any]:
        """Generate extraction configuration"""
        if guided and self.optimization_state:
            # Use previous best as starting point
            base = self.optimization_state.get('best_config', {})
            return {
                'entity_precision': np.clip(
                    base.get('entity_precision', 0.5) + np.random.normal(0, 0.1), 0, 1
                ),
                'relation_recall': np.clip(
                    base.get('relation_recall', 0.5) + np.random.normal(0, 0.1), 0, 1
                ),
                'graph_coherence': np.clip(
                    base.get('graph_coherence', 0.5) + np.random.normal(0, 0.1), 0, 1
                ),
                'semantic_depth': np.clip(
                    base.get('semantic_depth', 0.5) + np.random.normal(0, 0.1), 0, 1
                ),
                'extraction_strategy': np.random.choice(['hybrid', 'semantic', 'syntactic']),
                'chunking_size': np.random.randint(100, 1000)
            }
        else:
            return {
                'entity_precision': np.random.uniform(0, 1),
                'relation_recall': np.random.uniform(0, 1),
                'graph_coherence': np.random.uniform(0, 1),
                'semantic_depth': np.random.uniform(0, 1),
                'extraction_strategy': np.random.choice(['hybrid', 'semantic', 'syntactic']),
                'chunking_size': np.random.randint(100, 1000)
            }

class IndustrialCodeGenerationModule(IndustrialPipelineModule):
    """Industrial-strength code generation module"""
    
    def __init__(self, module_id: str):
        super().__init__(
            module_id=module_id,
            module_type="code_generation",
            complexity=1.5,
            resource_profile={
                'base_latency_ms': 200,
                'base_memory_mb': 150,
                'latency_variance': 50,
                'memory_variance': 30
            }
        )
    
    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate code generation quality"""
        syntax_correctness = config.get('syntax_correctness', 0.5)
        test_coverage = config.get('test_coverage', 0.5)
        code_efficiency = config.get('code_efficiency', 0.5)
        documentation = config.get('documentation', 0.5)
        
        # Complex performance model
        performance = (
            0.4 * syntax_correctness +
            0.3 * test_coverage +
            0.2 * code_efficiency +
            0.1 * documentation
        )
        
        # Complexity penalty
        performance *= (1 - 0.2 * (self.complexity - 1))
        
        # Add noise
        performance += np.random.normal(0, 0.08)
        performance = np.clip(performance, 0, 1)
        
        # Resource usage scales with code complexity
        resources = self.estimate_resources()
        resources['latency_ms'] *= (1 + 0.3 * test_coverage)
        resources['memory_mb'] *= (1 + 0.2 * code_efficiency)
        
        return performance, resources
    
    def generate_candidate(self, guided: bool = False) -> Dict[str, Any]:
        """Generate code generation configuration"""
        return {
            'syntax_correctness': np.random.uniform(0, 1),
            'test_coverage': np.random.uniform(0, 1),
            'code_efficiency': np.random.uniform(0, 1),
            'documentation': np.random.uniform(0, 1),
            'language': np.random.choice(['python', 'javascript', 'java']),
            'paradigm': np.random.choice(['functional', 'object-oriented', 'procedural']),
            'optimization_level': np.random.randint(0, 4)
        }

class MultiAgentCoordinationModule(IndustrialPipelineModule):
    """Multi-agent coordination module"""
    
    def __init__(self, module_id: str, num_agents: int = 3):
        super().__init__(
            module_id=module_id,
            module_type="multi_agent",
            complexity=1.3,
            resource_profile={
                'base_latency_ms': 100 * num_agents,
                'base_memory_mb': 50 * num_agents,
                'latency_variance': 20 * num_agents,
                'memory_variance': 10 * num_agents
            }
        )
        self.num_agents = num_agents
    
    def evaluate(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate multi-agent coordination"""
        coordination = config.get('coordination', 0.5)
        consensus = config.get('consensus', 0.5)
        task_allocation = config.get('task_allocation', 0.5)
        communication_efficiency = config.get('communication_efficiency', 0.5)
        
        # Agent interaction effects
        interaction_penalty = 0.1 * (self.num_agents - 1) / self.num_agents
        
        performance = (
            0.3 * coordination +
            0.3 * consensus +
            0.2 * task_allocation +
            0.2 * communication_efficiency
        ) * (1 - interaction_penalty)
        
        # Add noise
        performance += np.random.normal(0, 0.06)
        performance = np.clip(performance, 0, 1)
        
        # Resource usage scales with agent count
        resources = self.estimate_resources()
        resources['latency_ms'] *= (1 + 0.1 * (self.num_agents - 1))
        
        return performance, resources
    
    def generate_candidate(self, guided: bool = False) -> Dict[str, Any]:
        """Generate agent coordination configuration"""
        return {
            'coordination': np.random.uniform(0, 1),
            'consensus': np.random.uniform(0, 1),
            'task_allocation': np.random.uniform(0, 1),
            'communication_efficiency': np.random.uniform(0, 1),
            'coordination_strategy': np.random.choice(['centralized', 'distributed', 'hierarchical']),
            'consensus_algorithm': np.random.choice(['voting', 'averaging', 'leader-based'])
        }

# ==================== r-MIPRO Main Optimizer ====================

class RMIPRO:
    """Resource-Adaptive MIPRO for Industrial AI Pipelines"""
    
    def __init__(
        self,
        modules: List[IndustrialPipelineModule],
        total_budget: int = 1000,
        policy: OptimizationPolicy = OptimizationPolicy.HYBRID_ADAPTIVE,
        constraints: ResourceConstraints = None,
        exploration_ratio: float = 0.15,
        parallel_trials: int = 1
    ):
        self.modules = {m.module_id: m for m in modules}
        self.total_budget = total_budget
        self.used_budget = 0
        self.policy = policy
        self.constraints = constraints or ResourceConstraints()
        self.exploration_ratio = exploration_ratio
        self.parallel_trials = parallel_trials
        
        # Initialize scheduler
        self.scheduler = IntelligentScheduler(policy=policy)
        
        # Initialize metrics
        self.metrics = {
            m.module_id: AdvancedModuleMetrics(m.module_id)
            for m in modules
        }
        
        # Optimization log
        self.optimization_log = []
        self.resource_usage_log = []
        
    def run_exploration_phase(self) -> Dict[str, Any]:
        """Smart exploration phase with early stopping"""
        exploration_budget = int(self.total_budget * self.exploration_ratio)
        logger.info(f"Starting intelligent exploration with budget: {exploration_budget}")
        
        exploration_results = {}
        
        # Adaptive exploration - allocate more to complex modules
        complexity_scores = {
            m_id: m.complexity for m_id, m in self.modules.items()
        }
        total_complexity = sum(complexity_scores.values())
        
        for module_id, module in self.modules.items():
            # Weighted exploration based on complexity
            module_budget = int(
                exploration_budget * complexity_scores[module_id] / total_complexity
            )
            
            performances = []
            for _ in range(max(3, module_budget)):  # At least 3 trials per module
                if self.used_budget >= exploration_budget:
                    break
                
                config = module.generate_candidate()
                performance, resources = module.evaluate(config)
                
                self.metrics[module_id].update_metrics(performance, resources)
                performances.append(performance)
                self.used_budget += 1
                
                # Early stopping if module seems easy
                if len(performances) > 5 and np.var(performances) < 0.01:
                    break
            
            exploration_results[module_id] = {
                'mean': np.mean(performances) if performances else 0,
                'variance': np.var(performances) if performances else 1,
                'best': max(performances) if performances else 0
            }
        
        logger.info(f"Exploration complete. Used budget: {self.used_budget}")
        return exploration_results
    
    def optimize_step(self) -> bool:
        """Single optimization step with parallel trials support"""
        
        if self.used_budget >= self.total_budget:
            return False
        
        # Select modules for parallel trials
        selected_modules = []
        for _ in range(min(self.parallel_trials, self.total_budget - self.used_budget)):
            module_id = self.scheduler.select_module(
                self.metrics,
                self.total_budget - self.used_budget,
                self.constraints
            )
            
            if module_id:
                selected_modules.append(module_id)
        
        if not selected_modules:
            return False
        
        # Execute trials in parallel (simulated)
        for module_id in selected_modules:
            module = self.modules[module_id]
            
            # Generate candidate (guided after exploration)
            config = module.generate_candidate(guided=self.used_budget > self.total_budget * 0.2)
            
            # Evaluate
            performance, resources = module.evaluate(config)
            
            # Update metrics
            old_best = self.metrics[module_id].best_performance
            self.metrics[module_id].update_metrics(performance, resources)
            
            # Track improvements
            if performance > old_best:
                module.optimization_state['best_config'] = config
                logger.info(
                    f"[{self.used_budget}/{self.total_budget}] "
                    f"Module {module_id} improved: {old_best:.3f} → {performance:.3f}"
                )
            
            # Log step
            self.optimization_log.append({
                'step': self.used_budget,
                'module': module_id,
                'performance': performance,
                'resources': resources,
                'policy': self.policy.value
            })
            
            self.resource_usage_log.append(resources)
            self.used_budget += 1
        
        return True
    
    def optimize(self) -> Dict[str, Any]:
        """Run complete r-MIPRO optimization"""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info(f"Starting r-MIPRO Optimization")
        logger.info(f"Policy: {self.policy.value}")
        logger.info(f"Modules: {list(self.modules.keys())}")
        logger.info(f"Budget: {self.total_budget}")
        logger.info("=" * 60)
        
        # Phase 1: Intelligent Exploration
        exploration_results = self.run_exploration_phase()
        
        # Phase 2: Adaptive Optimization
        logger.info("\nStarting adaptive optimization phase")
        
        optimization_rounds = 0
        while self.used_budget < self.total_budget:
            if not self.optimize_step():
                break
            
            optimization_rounds += 1
            
            # Periodic logging
            if optimization_rounds % 50 == 0:
                self._log_progress()
            
            # Adaptive policy weight adjustment
            if optimization_rounds % 100 == 0 and self.policy == OptimizationPolicy.HYBRID_ADAPTIVE:
                self.scheduler._adapt_policy_weights(self.metrics)
        
        # Phase 3: Final Convergence Check
        self._check_convergence()
        
        # Compile results
        results = self._compile_results(time.time() - start_time)
        
        logger.info("\n" + "=" * 60)
        logger.info("r-MIPRO Optimization Complete")
        logger.info("=" * 60)
        
        return results
    
    def _check_convergence(self):
        """Check and report convergence status"""
        converged_modules = []
        active_modules = []
        
        for m_id, metrics in self.metrics.items():
            if metrics.time_since_improvement > 30 or metrics.variance < 1e-4:
                converged_modules.append(m_id)
            else:
                active_modules.append(m_id)
        
        logger.info(f"\nConvergence Status:")
        logger.info(f"  Converged: {converged_modules}")
        logger.info(f"  Active: {active_modules}")
    
    def _log_progress(self):
        """Log optimization progress"""
        logger.info(f"\n--- Progress Report [{self.used_budget}/{self.total_budget}] ---")
        
        for m_id, metrics in self.metrics.items():
            logger.info(
                f"{m_id}: "
                f"best={metrics.best_performance:.3f}, "
                f"current={metrics.current_performance:.3f}, "
                f"trials={metrics.total_resources_used}, "
                f"velocity={metrics.improvement_velocity:.4f}"
            )
        
        # Resource usage summary
        if self.resource_usage_log:
            avg_latency = np.mean([r['latency_ms'] for r in self.resource_usage_log[-50:]])
            avg_memory = np.mean([r['memory_mb'] for r in self.resource_usage_log[-50:]])
            logger.info(f"\nResource Usage (last 50 trials):")
            logger.info(f"  Avg Latency: {avg_latency:.1f}ms")
            logger.info(f"  Avg Memory: {avg_memory:.1f}MB")
    
    def _compile_results(self, elapsed_time: float) -> Dict[str, Any]:
        """Compile comprehensive results"""
        
        # Module performance summary
        module_summary = {}
        for m_id, metrics in self.metrics.items():
            module_summary[m_id] = {
                'best_performance': metrics.best_performance,
                'final_performance': metrics.current_performance,
                'improvement': metrics.best_performance - (
                    metrics.performance_history[0] if metrics.performance_history else 0
                ),
                'trials': metrics.total_resources_used,
                'convergence': metrics.time_since_improvement > 30,
                'avg_latency_ms': metrics.average_latency_ms,
                'avg_memory_mb': metrics.average_memory_mb,
                'confidence_interval': metrics.confidence_interval,
                'final_variance': metrics.variance
            }
        
        # Resource allocation analysis
        total_trials = sum(m.total_resources_used for m in self.metrics.values())
        allocation_efficiency = {
            m_id: m.total_resources_used / total_trials if total_trials > 0 else 0
            for m_id, m in self.metrics.items()
        }
        
        # Overall performance
        overall_performance = np.mean([m['best_performance'] for m in module_summary.values()])
        
        # Resource usage statistics
        if self.resource_usage_log:
            resource_stats = {
                'total_latency_ms': sum(r['latency_ms'] for r in self.resource_usage_log),
                'total_memory_mb': sum(r['memory_mb'] for r in self.resource_usage_log),
                'avg_latency_ms': np.mean([r['latency_ms'] for r in self.resource_usage_log]),
                'avg_memory_mb': np.mean([r['memory_mb'] for r in self.resource_usage_log]),
                'max_latency_ms': max(r['latency_ms'] for r in self.resource_usage_log),
                'max_memory_mb': max(r['memory_mb'] for r in self.resource_usage_log)
            }
        else:
            resource_stats = {}
        
        return {
            'summary': {
                'overall_performance': overall_performance,
                'total_budget_used': self.used_budget,
                'optimization_time_seconds': elapsed_time,
                'policy': self.policy.value,
                'modules_optimized': len(self.modules)
            },
            'module_results': module_summary,
            'allocation_efficiency': allocation_efficiency,
            'resource_statistics': resource_stats,
            'optimization_log': self.optimization_log[-100:],  # Last 100 entries
            'scheduler_history': self.scheduler.allocation_history[-50:]  # Last 50 decisions
        }

# ==================== Comparison Framework ====================

class RMIPROComparison:
    """Compare r-MIPRO with standard MIPROv2"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def create_industrial_pipeline(self) -> List[IndustrialPipelineModule]:
        """Create a realistic industrial pipeline"""
        return [
            TextGraphExtractionModule("text_graph_extractor"),
            IndustrialCodeGenerationModule("code_generator"),
            MultiAgentCoordinationModule("agent_coordinator", num_agents=3)
        ]
    
    def run_comparison(
        self,
        budget: int = 500,
        runs: int = 5,
        policies_to_test: List[OptimizationPolicy] = None
    ) -> Dict[str, Any]:
        """Run comprehensive comparison"""
        
        if policies_to_test is None:
            policies_to_test = [
                OptimizationPolicy.HYBRID_ADAPTIVE,
                OptimizationPolicy.ENTROPY_GUIDED,
                OptimizationPolicy.THOMPSON_SAMPLING
            ]
        
        logger.info("\n" + "=" * 60)
        logger.info("Starting r-MIPRO Comparison Study")
        logger.info("=" * 60)
        
        results = {}
        
        for policy in policies_to_test:
            policy_results = []
            
            for run in range(runs):
                logger.info(f"\n--- Testing {policy.value} - Run {run+1}/{runs} ---")
                
                # Create fresh pipeline
                modules = self.create_industrial_pipeline()
                
                # Create optimizer
                optimizer = RMIPRO(
                    modules=modules,
                    total_budget=budget,
                    policy=policy,
                    constraints=ResourceConstraints(
                        max_latency_ms=500,
                        max_memory_mb=512,
                        max_cost=5.0
                    ),
                    parallel_trials=2
                )
                
                # Run optimization
                result = optimizer.optimize()
                policy_results.append(result)
            
            results[policy.value] = policy_results
        
        # Analyze results
        analysis = self._analyze_comparison(results)
        
        self.comparison_results = {
            'raw_results': results,
            'analysis': analysis
        }
        
        return self.comparison_results
    
    def _analyze_comparison(self, results: Dict) -> Dict[str, Any]:
        """Analyze comparison results"""
        analysis = {}
        
        for policy, runs in results.items():
            performances = []
            improvements = []
            resource_efficiencies = []
            convergence_speeds = []
            
            for run in runs:
                # Overall performance
                perf = run['summary']['overall_performance']
                performances.append(perf)
                
                # Average improvement
                module_improvements = [
                    m['improvement'] for m in run['module_results'].values()
                ]
                improvements.append(np.mean(module_improvements))
                
                # Resource efficiency (performance per unit resource)
                if run['resource_statistics']:
                    efficiency = perf / (run['resource_statistics']['avg_latency_ms'] / 100)
                    resource_efficiencies.append(efficiency)
                
                # Convergence speed (trials to best performance)
                convergence_speeds.append(run['summary']['total_budget_used'])
            
            analysis[policy] = {
                'mean_performance': np.mean(performances),
                'std_performance': np.std(performances),
                'mean_improvement': np.mean(improvements),
                'mean_resource_efficiency': np.mean(resource_efficiencies) if resource_efficiencies else 0,
                'mean_convergence_speed': np.mean(convergence_speeds),
                'best_run_performance': max(performances),
                'worst_run_performance': min(performances)
            }
        
        return analysis
    
    def print_comparison_report(self):
        """Print formatted comparison report"""
        if not self.comparison_results:
            logger.error("No comparison results available. Run comparison first.")
            return
        
        analysis = self.comparison_results['analysis']
        
        print("\n" + "=" * 70)
        print("r-MIPRO COMPARISON REPORT")
        print("=" * 70)
        
        # Create comparison table
        print("\n📊 PERFORMANCE COMPARISON")
        print("-" * 70)
        print(f"{'Policy':<25} {'Mean Perf':<12} {'Std Dev':<12} {'Best Run':<12}")
        print("-" * 70)
        
        for policy, metrics in analysis.items():
            print(
                f"{policy:<25} "
                f"{metrics['mean_performance']:.4f}      "
                f"{metrics['std_performance']:.4f}      "
                f"{metrics['best_run_performance']:.4f}"
            )
        
        print("\n📈 IMPROVEMENT & EFFICIENCY")
        print("-" * 70)
        print(f"{'Policy':<25} {'Avg Improve':<12} {'Resource Eff':<12} {'Conv Speed':<12}")
        print("-" * 70)
        
        for policy, metrics in analysis.items():
            print(
                f"{policy:<25} "
                f"{metrics['mean_improvement']:.4f}      "
                f"{metrics['mean_resource_efficiency']:.4f}      "
                f"{metrics['mean_convergence_speed']:.0f}"
            )
        
        # Find best policy
        best_policy = max(
            analysis.items(),
            key=lambda x: x[1]['mean_performance']
        )[0]
        
        print("\n" + "=" * 70)
        print(f"🏆 BEST POLICY: {best_policy}")
        print("=" * 70)

# ==================== Main Execution ====================

def main():
    """Main execution for r-MIPRO testing"""
    
    # Set random seed
    np.random.seed(42)
    
    # Run comparison study
    comparison = RMIPROComparison()
    results = comparison.run_comparison(
        budget=300,
        runs=3,
        policies_to_test=[
            OptimizationPolicy.HYBRID_ADAPTIVE,
            OptimizationPolicy.ENTROPY_GUIDED,
            OptimizationPolicy.THOMPSON_SAMPLING,
            OptimizationPolicy.GRADIENT_BANDIT
        ]
    )
    
    # Print report
    comparison.print_comparison_report()
    
    # Save results
    with open('r_mipro_comparison_results.json', 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("\n✅ Results saved to r_mipro_comparison_results.json")

if __name__ == "__main__":
    main()