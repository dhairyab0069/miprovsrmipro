"""
MIPROv2 Implementation with OpenAI Integration
Based on Stanford NLP's MIPROv2 optimizer with real LLM calls
Enhanced with resource-adaptive features for comparison with r-MIPRO
"""

import os
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import random
import optuna
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
client = OpenAI(api_key=OPENAI_API_KEY)

# Cost tracking (GPT-4o-mini pricing)
COST_PER_1K_INPUT_TOKENS = 0.00015  # $0.15 per 1M input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.0006  # $0.60 per 1M output tokens

# ==================== Core MIPROv2 Components ====================

class BootstrapMode(Enum):
    """Bootstrap modes for MIPROv2"""
    RANDOM = "random"
    LABELED = "labeled"
    MIXED = "mixed"
    TELEPROMPTER = "teleprompter"

class TaskModel(Enum):
    """Task models for optimization"""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    REASONING = "reasoning"
    EXTRACTION = "extraction"

@dataclass
class MIPROConfig:
    """Configuration for MIPROv2 optimizer"""
    num_candidates: int = 10
    init_temperature: float = 1.0
    verbose: bool = True
    track_stats: bool = True
    view_data_batch_size: int = 10
    minibatch_size: int = 25
    minibatch_full_eval_steps: int = 10
    minibatch_candidates_buffer_size: int = 0
    max_bootstrapped_demos: int = 8
    max_labeled_demos: int = 16
    max_errors: int = 5
    
    # MIPROv2 specific
    num_trials: int = 100
    minibatch: bool = True
    requires_permission_to_run: bool = False
    
    # Hyperparameter optimization
    optimize_temperature: bool = True
    optimize_demo_count: bool = True
    optimize_instruction_detail: bool = True
    
    # LLM settings
    model: str = "gpt-4o-mini"
    max_tokens: int = 150
    use_real_llm: bool = True
    max_api_cost: float = 1.0  # Maximum cost in dollars

@dataclass
class LLMMetrics:
    """Metrics for LLM usage"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def add_call(self, input_tokens: int, output_tokens: int, latency_ms: float):
        """Add metrics from an API call"""
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms
        self.total_cost += (
            input_tokens * COST_PER_1K_INPUT_TOKENS / 1000 +
            output_tokens * COST_PER_1K_OUTPUT_TOKENS / 1000
        )
    
    def get_average_latency(self) -> float:
        """Get average latency per call"""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

@dataclass
class ProposalScore:
    """Score for a proposal/candidate"""
    prompt_id: str
    score: float
    metrics: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    llm_response: str = ""
    
@dataclass
class PromptCandidate:
    """Candidate prompt configuration"""
    candidate_id: str
    instruction: str
    few_shot_examples: List[Any] = field(default_factory=list)
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    evaluation_count: int = 0
    llm_metrics: LLMMetrics = field(default_factory=LLMMetrics)

# ==================== LLM Integration ====================

class LLMInterface:
    """Interface for LLM interactions"""
    
    def __init__(self, config: MIPROConfig):
        self.config = config
        self.client = client
        self.total_metrics = LLMMetrics()
        
    def create_prompt_from_candidate(
        self,
        candidate: PromptCandidate,
        query: str,
        task_model: TaskModel
    ) -> str:
        """Create a complete prompt from candidate configuration"""
        
        prompt_parts = []
        
        # Add instruction
        prompt_parts.append(candidate.instruction)
        
        # Add few-shot examples if available
        if candidate.few_shot_examples:
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(candidate.few_shot_examples, 1):
                if isinstance(example, dict):
                    prompt_parts.append(f"\nExample {i}:")
                    if 'input' in example:
                        prompt_parts.append(f"Input: {example['input']}")
                    if 'output' in example:
                        prompt_parts.append(f"Output: {example['output']}")
                else:
                    prompt_parts.append(f"\nExample {i}: {example}")
        
        # Add reasoning instructions based on metadata
        if candidate.metadata.get('reasoning_depth', 0) > 0.6:
            prompt_parts.append("\nApproach this step-by-step and show your reasoning.")
        
        # Add the actual query
        prompt_parts.append(f"\nQuery: {query}")
        
        # Add response format instruction
        response_format = candidate.metadata.get('response_format', 'standard')
        if response_format == 'concise':
            prompt_parts.append("\nProvide a brief, direct response.")
        elif response_format == 'detailed':
            prompt_parts.append("\nProvide a comprehensive, detailed response.")
        
        prompt_parts.append("\nResponse:")
        
        return "\n".join(prompt_parts)
    
    def call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = None
    ) -> Tuple[str, LLMMetrics]:
        """Call the LLM and return response with metrics"""
        
        if not self.config.use_real_llm or not self.client:
            # Simulate LLM response for testing
            return self._simulate_llm_response(prompt, temperature)
        
        # Check cost limit
        if self.total_metrics.total_cost >= self.config.max_api_cost:
            logger.warning(f"API cost limit reached: ${self.total_metrics.total_cost:.4f}")
            return self._simulate_llm_response(prompt, temperature)
        
        start_time = time.time()
        metrics = LLMMetrics()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            # Extract response and metrics
            llm_response = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            metrics.add_call(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                latency_ms
            )
            
            # Update total metrics
            self.total_metrics.add_call(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                latency_ms
            )
            
            return llm_response, metrics
            
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            metrics.errors.append(str(e))
            return f"Error: {e}", metrics
    
    def _simulate_llm_response(self, prompt: str, temperature: float) -> Tuple[str, LLMMetrics]:
        """Simulate LLM response for testing without API calls"""
        
        # Simulate based on prompt length and temperature
        prompt_length = len(prompt)
        
        # Simulate response
        responses = [
            "Based on the analysis, the solution involves multiple steps.",
            "The key insight here is to approach this systematically.",
            "After careful consideration, the optimal approach is clear.",
            "This requires a balanced perspective on the problem.",
            "The evidence suggests a comprehensive solution."
        ]
        
        # Add some randomness based on temperature
        if temperature > 0.8:
            response = random.choice(responses) + " " + random.choice([
                "Further exploration reveals interesting patterns.",
                "Additional factors should be considered.",
                "The implications are significant."
            ])
        else:
            response = responses[hash(prompt) % len(responses)]
        
        # Simulate metrics
        metrics = LLMMetrics()
        metrics.add_call(
            input_tokens=prompt_length // 4,  # Rough token estimate
            output_tokens=len(response) // 4,
            latency_ms=random.uniform(100, 500)
        )
        
        return response, metrics

# ==================== Bayesian Optimization Components ====================

class BayesianSignatureOptimizer:
    """Bayesian optimization for prompt signatures"""
    
    def __init__(self, config: MIPROConfig):
        self.config = config
        self.study = None
        self.best_score = float('-inf')
        self.best_candidate = None
        self.trial_history = []
        
    def initialize_study(self, seed: int = None):
        """Initialize Optuna study for Bayesian optimization"""
        if seed:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler()
            
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization"""
        params = {}
        
        if self.config.optimize_temperature:
            params['temperature'] = trial.suggest_float('temperature', 0.1, 2.0)
            
        if self.config.optimize_demo_count:
            params['num_demos'] = trial.suggest_int('num_demos', 0, self.config.max_bootstrapped_demos)
            
        if self.config.optimize_instruction_detail:
            params['instruction_detail'] = trial.suggest_categorical(
                'instruction_detail', 
                ['minimal', 'moderate', 'detailed']
            )
            
        # Additional MIPROv2 hyperparameters
        params['reasoning_depth'] = trial.suggest_float('reasoning_depth', 0.0, 1.0)
        params['response_format'] = trial.suggest_categorical(
            'response_format',
            ['concise', 'standard', 'detailed']
        )
        
        return params
    
    def update_score(self, candidate: PromptCandidate, score: float):
        """Update candidate score"""
        candidate.score = score
        candidate.evaluation_count += 1
        
        if score > self.best_score:
            self.best_score = score
            self.best_candidate = candidate

# ==================== Instruction & Example Generation ====================

class InstructionGenerator:
    """Generate and optimize instructions"""
    
    def __init__(self, task_model: TaskModel):
        self.task_model = task_model
        self.instruction_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load instruction templates based on task model"""
        templates = {
            TaskModel.CLASSIFICATION: [
                "Classify the following into appropriate categories",
                "Determine the correct category for each input",
                "Analyze and categorize the given information accurately"
            ],
            TaskModel.GENERATION: [
                "Generate a high-quality response based on the input",
                "Create appropriate output following the given context",
                "Produce a suitable and accurate response"
            ],
            TaskModel.REASONING: [
                "Apply logical reasoning to solve this problem",
                "Think step-by-step to reach the correct conclusion",
                "Use systematic analysis to find the solution"
            ],
            TaskModel.EXTRACTION: [
                "Extract all relevant information from the input",
                "Identify and extract the key elements",
                "Find and extract the specified information accurately"
            ]
        }
        return templates
    
    def generate_instruction(self, detail_level: str = 'moderate') -> str:
        """Generate instruction based on detail level"""
        base_instruction = random.choice(self.instruction_templates[self.task_model])
        
        if detail_level == 'minimal':
            return base_instruction + "."
        elif detail_level == 'moderate':
            return f"{base_instruction}. Be clear, accurate, and provide relevant details."
        else:  # detailed
            return f"{base_instruction}. Provide comprehensive analysis with clear reasoning, ensure accuracy, and include all relevant details in your response."

class ExampleSelector:
    """Select and manage few-shot examples"""
    
    def __init__(self, max_examples: int = 8):
        self.max_examples = max_examples
        self.example_pool = []
        self.selected_examples = []
        
    def add_to_pool(self, example: Dict[str, Any]):
        """Add example to pool"""
        self.example_pool.append(example)
        
    def select_examples(self, n: int, strategy: str = 'random') -> List[Dict]:
        """Select examples using specified strategy"""
        if not self.example_pool or n == 0:
            return []
            
        if strategy == 'random':
            return random.sample(self.example_pool, min(n, len(self.example_pool)))
        elif strategy == 'diverse':
            return self._select_diverse_examples(n)
        elif strategy == 'similar':
            return self._select_similar_examples(n)
        else:
            return self.example_pool[:n]
    
    def _select_diverse_examples(self, n: int) -> List[Dict]:
        """Select diverse examples"""
        if len(self.example_pool) <= n:
            return self.example_pool
        
        # Simple diversity: select evenly spaced examples
        step = len(self.example_pool) // n
        return [self.example_pool[i * step] for i in range(n)]
    
    def _select_similar_examples(self, n: int) -> List[Dict]:
        """Select similar examples"""
        return self.example_pool[:n]

# ==================== Evaluation Datasets ====================

class EvaluationDataset:
    """Dataset for evaluation"""
    
    def __init__(self, task_model: TaskModel):
        self.task_model = task_model
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load evaluation data based on task model"""
        
        if self.task_model == TaskModel.CLASSIFICATION:
            return [
                {"input": "This movie was fantastic!", "output": "positive", "category": "sentiment"},
                {"input": "The service was terrible.", "output": "negative", "category": "sentiment"},
                {"input": "It's okay, nothing special.", "output": "neutral", "category": "sentiment"}
            ]
        elif self.task_model == TaskModel.GENERATION:
            return [
                {"input": "Write a brief product description for a smartwatch", 
                 "output": "Track fitness, receive notifications, and monitor health metrics"},
                {"input": "Explain machine learning in simple terms",
                 "output": "Computers learning patterns from data to make predictions"}
            ]
        elif self.task_model == TaskModel.REASONING:
            return [
                {"input": "If all A are B, and all B are C, what can we say about A and C?",
                 "output": "All A are C (transitive property)"},
                {"input": "A train travels 60 miles in 1 hour. How far in 2.5 hours?",
                 "output": "150 miles (60 × 2.5)"}
            ]
        elif self.task_model == TaskModel.EXTRACTION:
            return [
                {"input": "John Smith, age 30, lives in New York and works as an engineer.",
                 "output": {"name": "John Smith", "age": 30, "location": "New York", "job": "engineer"}},
                {"input": "The meeting is scheduled for March 15, 2024 at 3 PM in Room 201.",
                 "output": {"date": "March 15, 2024", "time": "3 PM", "location": "Room 201"}}
            ]
        else:
            return []
    
    def get_samples(self, n: int = None) -> List[Dict]:
        """Get n samples from dataset"""
        if n is None or n >= len(self.data):
            return self.data
        return random.sample(self.data, n)

# ==================== Main MIPROv2 Optimizer ====================

class MIPROv2Optimizer:
    """Main MIPROv2 optimizer with LLM integration"""
    
    def __init__(
        self,
        task_model: TaskModel = TaskModel.GENERATION,
        config: MIPROConfig = None,
        seed: int = None
    ):
        self.config = config or MIPROConfig()
        self.task_model = task_model
        self.seed = seed
        
        # Initialize components
        self.llm_interface = LLMInterface(self.config)
        self.bayesian_optimizer = BayesianSignatureOptimizer(self.config)
        self.instruction_generator = InstructionGenerator(task_model)
        self.example_selector = ExampleSelector(self.config.max_bootstrapped_demos)
        self.dataset = EvaluationDataset(task_model)
        
        # Add examples to pool
        for example in self.dataset.get_samples():
            self.example_selector.add_to_pool(example)
        
        # Tracking
        self.candidates = []
        self.evaluation_history = []
        self.best_candidate = None
        
        # Initialize Bayesian optimization
        self.bayesian_optimizer.initialize_study(seed)
        
    def generate_candidate(self, trial: optuna.Trial = None) -> PromptCandidate:
        """Generate a prompt candidate"""
        
        # Get hyperparameters from Bayesian optimization
        if trial:
            params = self.bayesian_optimizer.suggest_hyperparameters(trial)
        else:
            params = {
                'temperature': 0.7,
                'num_demos': 3,
                'instruction_detail': 'moderate',
                'reasoning_depth': 0.5,
                'response_format': 'standard'
            }
        
        # Generate instruction
        instruction = self.instruction_generator.generate_instruction(
            params.get('instruction_detail', 'moderate')
        )
        
        # Select examples
        num_demos = params.get('num_demos', 3)
        examples = self.example_selector.select_examples(num_demos, strategy='diverse')
        
        # Create candidate
        candidate = PromptCandidate(
            candidate_id=f"candidate_{len(self.candidates)}",
            instruction=instruction,
            few_shot_examples=examples,
            temperature=params.get('temperature', 0.7),
            metadata=params
        )
        
        self.candidates.append(candidate)
        return candidate
    
    def evaluate_candidate(self, candidate: PromptCandidate, test_samples: List[Dict] = None) -> float:
        """Evaluate a candidate prompt with LLM"""
        
        if test_samples is None:
            test_samples = self.dataset.get_samples(3)  # Use 3 samples for evaluation
        
        scores = []
        
        for sample in test_samples:
            # Create prompt
            prompt = self.llm_interface.create_prompt_from_candidate(
                candidate,
                sample['input'],
                self.task_model
            )
            
            # Call LLM
            response, metrics = self.llm_interface.call_llm(
                prompt,
                temperature=candidate.temperature
            )
            
            # Update candidate metrics
            candidate.llm_metrics.total_calls += metrics.total_calls
            candidate.llm_metrics.total_cost += metrics.total_cost
            candidate.llm_metrics.total_latency_ms += metrics.total_latency_ms
            
            # Evaluate response
            score = self.evaluate_response(response, sample.get('output', ''))
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def evaluate_response(self, response: str, expected_output: Any) -> float:
        """Evaluate LLM response against expected output"""
        
        if not expected_output:
            # If no expected output, check if response is reasonable
            return 0.5 if len(response) > 10 else 0.0
        
        # Convert expected output to string for comparison
        if isinstance(expected_output, dict):
            expected_str = json.dumps(expected_output, sort_keys=True).lower()
        else:
            expected_str = str(expected_output).lower()
        
        response_lower = response.lower()
        
        # Check for key components
        if self.task_model == TaskModel.EXTRACTION:
            # For extraction, check if key values are present
            if isinstance(expected_output, dict):
                matches = sum(1 for v in expected_output.values() 
                             if str(v).lower() in response_lower)
                return matches / len(expected_output) if expected_output else 0.0
        
        # General similarity check
        expected_tokens = expected_str.split()
        matches = sum(1 for token in expected_tokens if token in response_lower)
        accuracy = matches / len(expected_tokens) if expected_tokens else 0.0
        
        # Bonus for response quality
        if len(response) > 20 and len(response) < 500:
            accuracy += 0.1
        
        return min(accuracy, 1.0)
    
    def optimize(self, n_trials: int = None) -> Dict[str, Any]:
        """Run MIPROv2 optimization with LLM"""
        
        n_trials = n_trials or self.config.num_trials
        
        logger.info("=" * 60)
        logger.info(f"Starting MIPROv2 Optimization")
        logger.info(f"Task Model: {self.task_model.value}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Using LLM: {self.config.use_real_llm}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        def objective(trial):
            # Generate candidate
            candidate = self.generate_candidate(trial)
            
            # Evaluate candidate
            score = self.evaluate_candidate(candidate)
            
            # Update optimizer
            self.bayesian_optimizer.update_score(candidate, score)
            
            # Track evaluation
            self.evaluation_history.append({
                'trial': trial.number,
                'score': score,
                'params': candidate.metadata,
                'cost': candidate.llm_metrics.total_cost,
                'latency_ms': candidate.llm_metrics.total_latency_ms
            })
            
            # Log progress
            if trial.number % 10 == 0:
                logger.info(
                    f"Trial {trial.number}: Score={score:.3f}, "
                    f"Cost=${candidate.llm_metrics.total_cost:.4f}"
                )
            
            return score
        
        # Run optimization
        self.bayesian_optimizer.study.optimize(objective, n_trials=n_trials)
        
        # Get best candidate
        self.best_candidate = self.bayesian_optimizer.best_candidate
        
        # Compile results
        results = self._compile_results(time.time() - start_time)
        
        logger.info("\n" + "=" * 60)
        logger.info("MIPROv2 Optimization Complete")
        logger.info("=" * 60)
        
        return results
    
    def _compile_results(self, elapsed_time: float) -> Dict[str, Any]:
        """Compile optimization results"""
        
        # Calculate statistics
        all_scores = [h['score'] for h in self.evaluation_history]
        all_costs = [h['cost'] for h in self.evaluation_history]
        all_latencies = [h['latency_ms'] for h in self.evaluation_history]
        
        best_result = {
            'score': self.best_candidate.score if self.best_candidate else 0,
            'instruction': self.best_candidate.instruction if self.best_candidate else "",
            'num_examples': len(self.best_candidate.few_shot_examples) if self.best_candidate else 0,
            'temperature': self.best_candidate.temperature if self.best_candidate else 0.7,
            'metadata': self.best_candidate.metadata if self.best_candidate else {}
        }
        
        return {
            'method': 'MIPROv2',
            'task_model': self.task_model.value,
            'elapsed_time': elapsed_time,
            'total_trials': len(self.evaluation_history),
            'best_result': best_result,
            'statistics': {
                'mean_score': np.mean(all_scores) if all_scores else 0,
                'best_score': max(all_scores) if all_scores else 0,
                'std_score': np.std(all_scores) if all_scores else 0,
                'total_cost': sum(all_costs),
                'mean_latency_ms': np.mean(all_latencies) if all_latencies else 0
            },
            'llm_metrics': {
                'total_calls': self.llm_interface.total_metrics.total_calls,
                'total_cost': self.llm_interface.total_metrics.total_cost,
                'total_tokens': (self.llm_interface.total_metrics.total_input_tokens + 
                               self.llm_interface.total_metrics.total_output_tokens),
                'avg_latency_ms': self.llm_interface.total_metrics.get_average_latency()
            },
            'config': {
                'model': self.config.model,
                'max_tokens': self.config.max_tokens,
                'num_trials': self.config.num_trials,
                'max_bootstrapped_demos': self.config.max_bootstrapped_demos
            },
            'evaluation_history': self.evaluation_history[-20:]  # Last 20 evaluations
        }

# ==================== Testing Framework ====================

class MIPROv2TestSuite:
    """Test suite for MIPROv2 with different task models"""
    
    def __init__(self, use_real_llm: bool = False):
        self.use_real_llm = use_real_llm
        self.results = {}
        
    def run_task_comparison(self, n_trials: int = 30) -> Dict[str, Any]:
        """Compare performance across different task models"""
        
        logger.info("\n" + "=" * 60)
        logger.info("MIPROv2 Task Model Comparison")
        logger.info("=" * 60)
        
        task_models = [
            TaskModel.CLASSIFICATION,
            TaskModel.GENERATION,
            TaskModel.REASONING,
            TaskModel.EXTRACTION
        ]
        
        for task_model in task_models:
            logger.info(f"\nTesting {task_model.value}...")
            
            # Configure optimizer
            config = MIPROConfig(
                num_trials=n_trials,
                use_real_llm=self.use_real_llm,
                model="gpt-4o-mini",
                max_api_cost=0.5,
                optimize_temperature=True,
                optimize_demo_count=True,
                optimize_instruction_detail=True
            )
            
            # Run optimization
            optimizer = MIPROv2Optimizer(
                task_model=task_model,
                config=config,
                seed=42
            )
            
            result = optimizer.optimize(n_trials)
            self.results[task_model.value] = result
        
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze results across task models"""
        
        summary = {}
        for task_model, result in self.results.items():
            summary[task_model] = {
                'best_score': result['statistics']['best_score'],
                'mean_score': result['statistics']['mean_score'],
                'total_cost': result['statistics']['total_cost'],
                'best_instruction': result['best_result']['instruction'][:100] + "..."
            }
        
        return {
            'task_results': self.results,
            'summary': summary
        }
    
    def print_report(self):
        """Print test results"""
        
        print("\n" + "=" * 70)
        print("MIPROV2 TASK MODEL COMPARISON REPORT")
        print("=" * 70)
        
        if not self.results:
            print("No results available. Run tests first.")
            return
        
        # Print summary table
        print("\nPerformance Summary:")
        print("-" * 70)
        print(f"{'Task Model':<15} {'Best Score':<12} {'Mean Score':<12} {'Total Cost':<12}")
        print("-" * 70)
        
        for task_model, result in self.results.items():
            stats = result['statistics']
            print(
                f"{task_model:<15} "
                f"{stats['best_score']:.3f}        "
                f"{stats['mean_score']:.3f}        "
                f"${stats['total_cost']:.4f}"
            )
        
        # Print best configurations
        print("\n" + "=" * 70)
        print("BEST CONFIGURATIONS")
        print("=" * 70)
        
        for task_model, result in self.results.items():
            best = result['best_result']
            print(f"\n{task_model}:")
            print(f"  Score: {best['score']:.3f}")
            print(f"  Temperature: {best['temperature']:.2f}")
            print(f"  Examples: {best['num_examples']}")
            print(f"  Instruction: {best['instruction'][:100]}...")

# ==================== Main Execution ====================

def main():
    """Main execution function"""
    
    # Check if API key is configured
    if OPENAI_API_KEY == "your-api-key-here":
        logger.warning("No OpenAI API key found. Running in simulation mode.")
        use_real_llm = False
    else:
        logger.info("OpenAI API key found. Using real LLM calls.")
        use_real_llm = True
    
    # Run test suite
    test_suite = MIPROv2TestSuite(use_real_llm=use_real_llm)
    
    # Run comparison across task models
    results = test_suite.run_task_comparison(n_trials=20)  # Reduced for cost management
    
    # Print report
    test_suite.print_report()
    
    # Save results
    with open('miprov2_llm_results.json', 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("\nResults saved to miprov2_llm_results.json")
    
    # Print cost summary if using real LLM
    if use_real_llm:
        total_cost = sum(r['statistics']['total_cost'] for r in test_suite.results.values())
        print(f"\n" + "=" * 70)
        print(f"TOTAL API COST: ${total_cost:.4f}")
        print("=" * 70)

if __name__ == "__main__":
    main()