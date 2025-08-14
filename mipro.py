"""
MIPROv2 Testing Framework with Resource-Adaptive Optimization
Based on r-MIPRO concepts for pipeline optimization
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import optuna
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Core Data Structures ====================

class OptimizationStrategy(Enum):
    """Resource allocation strategies"""
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    BANDIT = "bandit"
    UNCERTAINTY = "uncertainty"
    BOTTLENECK = "bottleneck"

@dataclass
class ModuleMetrics:
    """Metrics for a single module"""
    module_id: str
    current_performance: float = 0.0
    best_performance: float = 0.0
    improvement_rate: float = 0.0
    uncertainty: float = 1.0
    trials_allocated: int = 0
    convergence_score: float = 0.0
    bottleneck_score: float = 0.5
    history: List[float] = field(default_factory=list)
    
    def update(self, new_performance: float):
        """Update metrics with new performance data"""
        self.history.append(new_performance)
        self.current_performance = new_performance
        self.best_performance = max(self.best_performance, new_performance)
        
        # Calculate improvement rate
        if len(self.history) > 1:
            self.improvement_rate = new_performance - self.history[-2]
        
        # Update uncertainty
        if len(self.history) > 2:
            self.uncertainty = np.std(self.history[-5:])
        
        # Calculate convergence score
        if len(self.history) > 5:
            recent_variance = np.var(self.history[-5:])
            self.convergence_score = 1 / (1 + recent_variance)

@dataclass
class ResourceBudget:
    """Resource budget management"""
    total_budget: int
    used_budget: int = 0
    module_allocations: Dict[str, int] = field(default_factory=dict)
    
    @property
    def remaining_budget(self) -> int:
        return self.total_budget - self.used_budget
    
    def allocate(self, module_id: str, amount: int = 1):
        """Allocate resources to a module"""
        if self.remaining_budget >= amount:
            self.used_budget += amount
            self.module_allocations[module_id] = \
                self.module_allocations.get(module_id, 0) + amount
            return True
        return False

# ==================== Pipeline Components ====================

class PipelineModule(ABC):
    """Abstract base class for pipeline modules"""
    
    def __init__(self, module_id: str, complexity: float = 1.0):
        self.module_id = module_id
        self.complexity = complexity
        self.current_prompt = None
        self.best_prompt = None
        
    @abstractmethod
    def evaluate(self, prompt_config: Dict[str, Any]) -> float:
        """Evaluate module with given configuration"""
        pass
    
    @abstractmethod
    def generate_candidate(self) -> Dict[str, Any]:
        """Generate a candidate prompt configuration"""
        pass

class TextExtractionModule(PipelineModule):
    """Text extraction pipeline module"""
    
    def evaluate(self, prompt_config: Dict[str, Any]) -> float:
        """Simulate text extraction performance"""
        # Simulate evaluation with some randomness
        base_score = 0.6
        instruction_quality = prompt_config.get('instruction_quality', 0.5)
        few_shot_count = prompt_config.get('few_shot_count', 0)
        
        score = base_score + 0.2 * instruction_quality + 0.05 * min(few_shot_count, 5)
        score += np.random.normal(0, 0.05)  # Add noise
        
        return min(max(score, 0), 1)
    
    def generate_candidate(self) -> Dict[str, Any]:
        """Generate candidate configuration for text extraction"""
        return {
            'instruction_quality': np.random.uniform(0, 1),
            'few_shot_count': np.random.randint(0, 8),
            'temperature': np.random.uniform(0, 1),
            'extraction_strategy': np.random.choice(['entity', 'keyword', 'semantic'])
        }

class CodeGenerationModule(PipelineModule):
    """Code generation pipeline module"""
    
    def evaluate(self, prompt_config: Dict[str, Any]) -> float:
        """Simulate code generation performance"""
        base_score = 0.5
        syntax_awareness = prompt_config.get('syntax_awareness', 0.5)
        test_driven = prompt_config.get('test_driven', False)
        
        score = base_score + 0.3 * syntax_awareness + (0.1 if test_driven else 0)
        score *= self.complexity
        score += np.random.normal(0, 0.08)
        
        return min(max(score, 0), 1)
    
    def generate_candidate(self) -> Dict[str, Any]:
        """Generate candidate configuration for code generation"""
        return {
            'syntax_awareness': np.random.uniform(0, 1),
            'test_driven': np.random.choice([True, False]),
            'language_specific': np.random.choice([True, False]),
            'complexity_handling': np.random.uniform(0, 1)
        }

class AgentModule(PipelineModule):
    """Agent reasoning pipeline module"""
    
    def evaluate(self, prompt_config: Dict[str, Any]) -> float:
        """Simulate agent reasoning performance"""
        base_score = 0.4
        reasoning_depth = prompt_config.get('reasoning_depth', 0.5)
        chain_of_thought = prompt_config.get('chain_of_thought', False)
        
        score = base_score + 0.4 * reasoning_depth + (0.15 if chain_of_thought else 0)
        score += np.random.normal(0, 0.1)
        
        return min(max(score, 0), 1)
    
    def generate_candidate(self) -> Dict[str, Any]:
        """Generate candidate configuration for agent reasoning"""
        return {
            'reasoning_depth': np.random.uniform(0, 1),
            'chain_of_thought': np.random.choice([True, False]),
            'step_by_step': np.random.choice([True, False]),
            'validation_loops': np.random.randint(0, 5)
        }

# ==================== Resource Allocation Scheduler ====================

class AdaptiveScheduler:
    """Adaptive resource allocation scheduler for r-MIPRO"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.allocation_history = []
        
    def compute_allocation_scores(self, metrics: Dict[str, ModuleMetrics]) -> Dict[str, float]:
        """Compute allocation scores for each module"""
        scores = {}
        
        if self.strategy == OptimizationStrategy.UNIFORM:
            # Equal allocation
            for module_id in metrics:
                scores[module_id] = 1.0
                
        elif self.strategy == OptimizationStrategy.ADAPTIVE:
            # Balanced approach considering multiple factors
            for module_id, m in metrics.items():
                improvement_factor = max(0.1, m.improvement_rate)
                uncertainty_factor = m.uncertainty
                convergence_penalty = 1 - m.convergence_score
                bottleneck_factor = m.bottleneck_score
                
                scores[module_id] = (
                    0.3 * improvement_factor +
                    0.2 * uncertainty_factor +
                    0.2 * convergence_penalty +
                    0.3 * bottleneck_factor
                )
                
        elif self.strategy == OptimizationStrategy.BANDIT:
            # Multi-armed bandit approach (UCB-like)
            total_trials = sum(m.trials_allocated for m in metrics.values())
            for module_id, m in metrics.items():
                exploitation = m.current_performance
                exploration = np.sqrt(2 * np.log(max(1, total_trials)) / max(1, m.trials_allocated))
                scores[module_id] = exploitation + exploration
                
        elif self.strategy == OptimizationStrategy.UNCERTAINTY:
            # Focus on high uncertainty modules
            for module_id, m in metrics.items():
                scores[module_id] = m.uncertainty * (1 - m.convergence_score)
                
        elif self.strategy == OptimizationStrategy.BOTTLENECK:
            # Focus on bottleneck modules
            for module_id, m in metrics.items():
                scores[module_id] = m.bottleneck_score * (1 - m.convergence_score)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v/total_score for k, v in scores.items()}
        
        return scores
    
    def select_module(self, metrics: Dict[str, ModuleMetrics], budget: ResourceBudget) -> Optional[str]:
        """Select next module to optimize"""
        if budget.remaining_budget <= 0:
            return None
        
        scores = self.compute_allocation_scores(metrics)
        
        # Filter out converged modules
        active_modules = {
            k: v for k, v in scores.items() 
            if metrics[k].convergence_score < 0.95
        }
        
        if not active_modules:
            return None
        
        # Weighted random selection
        modules = list(active_modules.keys())
        weights = list(active_modules.values())
        
        selected = np.random.choice(modules, p=weights/np.sum(weights))
        self.allocation_history.append(selected)
        
        return selected

# ==================== MIPROv2 Optimizer ====================

class MIPROv2Optimizer:
    """Main MIPROv2 optimizer with resource-adaptive capabilities"""
    
    def __init__(
        self,
        modules: List[PipelineModule],
        total_budget: int = 1000,
        strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
        exploration_ratio: float = 0.2
    ):
        self.modules = {m.module_id: m for m in modules}
        self.budget = ResourceBudget(total_budget)
        self.scheduler = AdaptiveScheduler(strategy)
        self.exploration_ratio = exploration_ratio
        self.metrics = {
            m.module_id: ModuleMetrics(m.module_id) 
            for m in modules
        }
        self.optimization_log = []
        
    def run_exploration_phase(self):
        """Initial exploration phase for baseline performance"""
        exploration_budget = int(self.budget.total_budget * self.exploration_ratio)
        logger.info(f"Starting exploration phase with budget: {exploration_budget}")
        
        trials_per_module = exploration_budget // len(self.modules)
        
        for module_id, module in self.modules.items():
            for _ in range(trials_per_module):
                if not self.budget.allocate(module_id):
                    break
                    
                candidate = module.generate_candidate()
                score = module.evaluate(candidate)
                self.metrics[module_id].update(score)
                
                if score > self.metrics[module_id].best_performance:
                    module.best_prompt = candidate
        
        # Calculate initial bottleneck scores
        self._update_bottleneck_scores()
    
    def _update_bottleneck_scores(self):
        """Update bottleneck scores based on relative performance"""
        performances = [m.current_performance for m in self.metrics.values()]
        min_perf = min(performances)
        max_perf = max(performances)
        
        if max_perf > min_perf:
            for metric in self.metrics.values():
                # Lower performance = higher bottleneck score
                metric.bottleneck_score = 1 - (metric.current_performance - min_perf) / (max_perf - min_perf)
    
    def optimize_step(self) -> bool:
        """Single optimization step"""
        # Select module to optimize
        selected_module_id = self.scheduler.select_module(self.metrics, self.budget)
        
        if selected_module_id is None:
            return False
        
        # Allocate resource
        if not self.budget.allocate(selected_module_id):
            return False
        
        # Generate and evaluate candidate
        module = self.modules[selected_module_id]
        candidate = module.generate_candidate()
        score = module.evaluate(candidate)
        
        # Update metrics
        self.metrics[selected_module_id].update(score)
        self.metrics[selected_module_id].trials_allocated += 1
        
        # Update best configuration if improved
        if score > self.metrics[selected_module_id].best_performance:
            module.best_prompt = candidate
            logger.info(f"Module {selected_module_id} improved: {score:.3f}")
        
        # Log optimization step
        self.optimization_log.append({
            'step': self.budget.used_budget,
            'module': selected_module_id,
            'score': score,
            'best_score': self.metrics[selected_module_id].best_performance
        })
        
        return True
    
    def optimize(self) -> Dict[str, Any]:
        """Run full optimization process"""
        logger.info(f"Starting MIPROv2 optimization with strategy: {self.scheduler.strategy.value}")
        
        # Exploration phase
        self.run_exploration_phase()
        
        # Adaptive optimization phase
        logger.info("Starting adaptive optimization phase")
        
        while self.budget.remaining_budget > 0:
            # Update bottleneck scores periodically
            if self.budget.used_budget % 50 == 0:
                self._update_bottleneck_scores()
            
            if not self.optimize_step():
                break
            
            # Log progress
            if self.budget.used_budget % 100 == 0:
                self._log_progress()
        
        # Final results
        results = self._compile_results()
        logger.info("Optimization completed")
        
        return results
    
    def _log_progress(self):
        """Log current optimization progress"""
        logger.info(f"Progress: {self.budget.used_budget}/{self.budget.total_budget}")
        for module_id, metric in self.metrics.items():
            logger.info(
                f"  {module_id}: best={metric.best_performance:.3f}, "
                f"convergence={metric.convergence_score:.3f}, "
                f"trials={metric.trials_allocated}"
            )
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile optimization results"""
        return {
            'final_metrics': {
                module_id: {
                    'best_performance': m.best_performance,
                    'trials_allocated': m.trials_allocated,
                    'convergence_score': m.convergence_score,
                    'improvement': m.best_performance - (m.history[0] if m.history else 0)
                }
                for module_id, m in self.metrics.items()
            },
            'best_configurations': {
                module_id: module.best_prompt
                for module_id, module in self.modules.items()
            },
            'resource_allocation': dict(self.budget.module_allocations),
            'total_budget_used': self.budget.used_budget,
            'strategy': self.scheduler.strategy.value,
            'optimization_log': self.optimization_log
        }

# ==================== Testing Framework ====================

class MIPROv2TestSuite:
    """Comprehensive test suite for MIPROv2"""
    
    def __init__(self):
        self.test_results = []
    
    def create_test_pipeline(self, complexity_profile: str = "balanced") -> List[PipelineModule]:
        """Create a test pipeline with different complexity profiles"""
        if complexity_profile == "balanced":
            return [
                TextExtractionModule("text_extractor", complexity=1.0),
                CodeGenerationModule("code_generator", complexity=1.0),
                AgentModule("agent_reasoner", complexity=1.0)
            ]
        elif complexity_profile == "bottleneck":
            return [
                TextExtractionModule("text_extractor", complexity=0.8),
                CodeGenerationModule("code_generator", complexity=1.5),  # Bottleneck
                AgentModule("agent_reasoner", complexity=0.9)
            ]
        elif complexity_profile == "complex":
            return [
                TextExtractionModule("text_extractor", complexity=1.2),
                CodeGenerationModule("code_generator", complexity=1.3),
                AgentModule("agent_reasoner", complexity=1.4)
            ]
        else:
            raise ValueError(f"Unknown complexity profile: {complexity_profile}")
    
    def test_strategy_comparison(self, budget: int = 500, runs: int = 3):
        """Compare different optimization strategies"""
        strategies = [
            OptimizationStrategy.UNIFORM,
            OptimizationStrategy.ADAPTIVE,
            OptimizationStrategy.BANDIT,
            OptimizationStrategy.UNCERTAINTY,
            OptimizationStrategy.BOTTLENECK
        ]
        
        results = {}
        
        for strategy in strategies:
            strategy_results = []
            
            for run in range(runs):
                logger.info(f"\nTesting {strategy.value} - Run {run+1}/{runs}")
                
                # Create fresh pipeline
                modules = self.create_test_pipeline("bottleneck")
                
                # Run optimization
                optimizer = MIPROv2Optimizer(
                    modules=modules,
                    total_budget=budget,
                    strategy=strategy
                )
                
                result = optimizer.optimize()
                strategy_results.append(result)
            
            results[strategy.value] = strategy_results
        
        self.test_results.append({
            'test': 'strategy_comparison',
            'results': results
        })
        
        return self._analyze_strategy_comparison(results)
    
    def test_budget_scaling(self, budgets: List[int] = None):
        """Test performance with different budget sizes"""
        if budgets is None:
            budgets = [100, 250, 500, 1000, 2000]
        
        results = {}
        
        for budget in budgets:
            logger.info(f"\nTesting with budget: {budget}")
            
            modules = self.create_test_pipeline("balanced")
            optimizer = MIPROv2Optimizer(
                modules=modules,
                total_budget=budget,
                strategy=OptimizationStrategy.ADAPTIVE
            )
            
            result = optimizer.optimize()
            results[budget] = result
        
        self.test_results.append({
            'test': 'budget_scaling',
            'results': results
        })
        
        return self._analyze_budget_scaling(results)
    
    def test_pipeline_complexity(self):
        """Test different pipeline complexity profiles"""
        profiles = ["balanced", "bottleneck", "complex"]
        results = {}
        
        for profile in profiles:
            logger.info(f"\nTesting {profile} pipeline")
            
            modules = self.create_test_pipeline(profile)
            optimizer = MIPROv2Optimizer(
                modules=modules,
                total_budget=1000,
                strategy=OptimizationStrategy.ADAPTIVE
            )
            
            result = optimizer.optimize()
            results[profile] = result
        
        self.test_results.append({
            'test': 'pipeline_complexity',
            'results': results
        })
        
        return results
    
    def _analyze_strategy_comparison(self, results: Dict) -> Dict:
        """Analyze strategy comparison results"""
        analysis = {}
        
        for strategy, runs in results.items():
            performances = []
            allocations = []
            
            for run in runs:
                avg_perf = np.mean([
                    m['best_performance'] 
                    for m in run['final_metrics'].values()
                ])
                performances.append(avg_perf)
                
                # Calculate allocation efficiency (std dev of allocations)
                allocs = list(run['resource_allocation'].values())
                if allocs:
                    allocations.append(np.std(allocs))
            
            analysis[strategy] = {
                'mean_performance': np.mean(performances),
                'std_performance': np.std(performances),
                'allocation_variance': np.mean(allocations) if allocations else 0
            }
        
        return analysis
    
    def _analyze_budget_scaling(self, results: Dict) -> Dict:
        """Analyze budget scaling results"""
        analysis = {}
        
        for budget, result in results.items():
            avg_performance = np.mean([
                m['best_performance'] 
                for m in result['final_metrics'].values()
            ])
            
            analysis[budget] = {
                'average_performance': avg_performance,
                'performance_per_budget': avg_performance / budget,
                'modules': result['final_metrics']
            }
        
        return analysis
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("=" * 60)
        logger.info("Starting MIPROv2 Comprehensive Test Suite")
        logger.info("=" * 60)
        
        # Test 1: Strategy Comparison
        logger.info("\n[TEST 1] Strategy Comparison")
        strategy_analysis = self.test_strategy_comparison(budget=500, runs=3)
        self._print_strategy_results(strategy_analysis)
        
        # Test 2: Budget Scaling
        logger.info("\n[TEST 2] Budget Scaling")
        budget_analysis = self.test_budget_scaling()
        self._print_budget_results(budget_analysis)
        
        # Test 3: Pipeline Complexity
        logger.info("\n[TEST 3] Pipeline Complexity")
        complexity_results = self.test_pipeline_complexity()
        self._print_complexity_results(complexity_results)
        
        logger.info("\n" + "=" * 60)
        logger.info("Test Suite Completed")
        logger.info("=" * 60)
        
        return self.test_results
    
    def _print_strategy_results(self, analysis: Dict):
        """Print strategy comparison results"""
        print("\nStrategy Performance Summary:")
        print("-" * 40)
        for strategy, metrics in analysis.items():
            print(f"{strategy:15} | Avg: {metrics['mean_performance']:.3f} "
                  f"| Std: {metrics['std_performance']:.3f}")
    
    def _print_budget_results(self, analysis: Dict):
        """Print budget scaling results"""
        print("\nBudget Scaling Summary:")
        print("-" * 40)
        for budget, metrics in analysis.items():
            print(f"Budget {budget:4} | Performance: {metrics['average_performance']:.3f} "
                  f"| Efficiency: {metrics['performance_per_budget']:.5f}")
    
    def _print_complexity_results(self, results: Dict):
        """Print pipeline complexity results"""
        print("\nPipeline Complexity Summary:")
        print("-" * 40)
        for profile, result in results.items():
            avg_perf = np.mean([
                m['best_performance'] 
                for m in result['final_metrics'].values()
            ])
            print(f"{profile:10} | Avg Performance: {avg_perf:.3f}")

# ==================== Main Execution ====================

def main():
    """Main execution function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create and run test suite
    test_suite = MIPROv2TestSuite()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Save results
    with open('miprov2_test_results.json', 'w') as f:
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
    
    logger.info("\nResults saved to miprov2_test_results.json")
    
    # Demonstration: Run a single optimization with detailed output
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION: Single Optimization Run")
    logger.info("=" * 60)
    
    demo_modules = [
        TextExtractionModule("text_extractor", complexity=0.8),
        CodeGenerationModule("code_generator", complexity=1.5),
        AgentModule("agent_reasoner", complexity=1.0)
    ]
    
    demo_optimizer = MIPROv2Optimizer(
        modules=demo_modules,
        total_budget=200,
        strategy=OptimizationStrategy.ADAPTIVE,
        exploration_ratio=0.25
    )
    
    demo_results = demo_optimizer.optimize()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION RESULTS")
    print("=" * 60)
    print("\nFinal Module Performance:")
    for module_id, metrics in demo_results['final_metrics'].items():
        print(f"\n{module_id}:")
        print(f"  Best Performance: {metrics['best_performance']:.3f}")
        print(f"  Improvement: {metrics['improvement']:.3f}")
        print(f"  Trials Allocated: {metrics['trials_allocated']}")
        print(f"  Convergence: {metrics['convergence_score']:.3f}")
    
    print("\nResource Allocation:")
    total_trials = sum(demo_results['resource_allocation'].values())
    for module_id, trials in demo_results['resource_allocation'].items():
        percentage = (trials / total_trials) * 100 if total_trials > 0 else 0
        print(f"  {module_id}: {trials} trials ({percentage:.1f}%)")
    
    print(f"\nTotal Budget Used: {demo_results['total_budget_used']}")
    print(f"Optimization Strategy: {demo_results['strategy']}")

if __name__ == "__main__":
    main()