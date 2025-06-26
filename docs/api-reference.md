# API Reference

This document provides a comprehensive reference for all classes, functions, and modules in the Mental Health LLM Evaluation framework.

## Table of Contents

- [Models](#models)
- [Evaluation](#evaluation)
- [Conversation Management](#conversation-management)
- [Analysis](#analysis)
- [Scenarios](#scenarios)
- [Storage](#storage)
- [Utils](#utils)

## Models

### BaseModel

Abstract base class for all LLM implementations.

```python
from src.models.base_model import BaseModel

class BaseModel(ABC):
    """Abstract base class for all LLM models."""
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response to user prompt.
        
        Args:
            prompt: Input text prompt
            context: Optional context information
            
        Returns:
            Dict containing response and metadata
        """
```

### OpenAIClient

OpenAI GPT-4 client implementation.

```python
from src.models.openai_client import OpenAIClient

class OpenAIClient(BaseModel):
    """OpenAI GPT-4 client for cloud-based inference."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (default: gpt-4)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response length
        """
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token usage."""
```

### DeepSeekClient

DeepSeek local model client implementation.

```python
from src.models.deepseek_client import DeepSeekClient

class DeepSeekClient(BaseModel):
    """DeepSeek local model client for on-premise inference."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        precision: str = "fp16",
        max_memory: Optional[int] = None
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            model_path: Path to DeepSeek model files
            device: Device for inference (cuda/cpu/auto)
            precision: Model precision (fp16/fp32/int8)
            max_memory: Maximum memory usage in GB
        """
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response using local model."""
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
```

## Evaluation

### CompositeScorer

Main evaluation orchestrator combining multiple scoring dimensions.

```python
from src.evaluation.composite_scorer import CompositeScorer, CompositeScore

@dataclass
class CompositeScore:
    """Comprehensive evaluation score."""
    overall_score: float
    technical_score: float
    therapeutic_score: float
    patient_score: float
    
    # Detailed breakdowns
    technical_details: TechnicalDetails
    therapeutic_details: TherapeuticDetails
    patient_details: PatientDetails
    
    # Metadata
    conversation_id: str
    model_name: str
    scenario_id: str
    timestamp: datetime

class CompositeScorer:
    """Orchestrates comprehensive conversation evaluation."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        enable_safety_checks: bool = True,
        normalize_scores: bool = True
    ):
        """
        Initialize composite scorer.
        
        Args:
            weights: Custom weights for score components
            enable_safety_checks: Enable safety validation
            normalize_scores: Normalize scores to 0-10 scale
        """
    
    def calculate_composite_score(
        self,
        conversation: Dict[str, Any],
        scenario_id: str
    ) -> CompositeScore:
        """
        Calculate comprehensive score for conversation.
        
        Args:
            conversation: Conversation data structure
            scenario_id: Associated scenario identifier
            
        Returns:
            CompositeScore with detailed breakdown
        """
        
    def evaluate_batch(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, List[CompositeScore]]:
        """Evaluate multiple conversations in batch."""
```

### EmpathyScorer

Evaluates emotional understanding and validation in responses.

```python
from src.evaluation.empathy_scorer import EmpathyScorer

class EmpathyScorer:
    """Evaluates empathy and emotional understanding."""
    
    def score_empathy(
        self,
        response: str,
        patient_message: str,
        context: str = "general"
    ) -> float:
        """
        Score empathy in assistant response.
        
        Args:
            response: Assistant's response text
            patient_message: Patient's message being responded to
            context: Conversation context/scenario type
            
        Returns:
            Empathy score (0.0-10.0)
        """
    
    def analyze_emotional_markers(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analyze emotional markers in text."""
        
    def get_empathy_recommendations(
        self,
        score: float,
        response: str
    ) -> List[str]:
        """Get recommendations for improving empathy."""
```

### SafetyDetector

Detects safety issues and crisis situations.

```python
from src.evaluation.safety_detector import SafetyDetector, SafetyFlag, SafetyLevel

class SafetyFlag(Enum):
    """Safety concern categories."""
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    SUBSTANCE_ABUSE = "substance_abuse"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_SAFETY = "child_safety"
    MEDICAL_ADVICE = "inappropriate_medical_advice"
    BOUNDARY_VIOLATION = "boundary_violation"

class SafetyLevel(Enum):
    """Safety risk levels."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    CRISIS = "crisis"

class SafetyDetector:
    """Detects safety issues in conversations."""
    
    def detect_safety_issues(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Tuple[List[SafetyFlag], SafetyLevel]:
        """
        Detect safety issues in text.
        
        Args:
            text: Text to analyze
            context: Optional context information
            
        Returns:
            Tuple of (safety_flags, risk_level)
        """
    
    def assess_crisis_risk(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Assess overall crisis risk from conversation."""
```

### CoherenceEvaluator

Evaluates logical flow and consistency of responses.

```python
from src.evaluation.coherence_evaluator import CoherenceEvaluator

class CoherenceEvaluator:
    """Evaluates response coherence and consistency."""
    
    def evaluate_coherence(
        self,
        assistant_response: str,
        patient_message: str,
        context: str
    ) -> float:
        """
        Evaluate response coherence.
        
        Args:
            assistant_response: Response to evaluate
            patient_message: Patient's message
            context: Conversation context
            
        Returns:
            Coherence score (0.0-10.0)
        """
    
    def check_consistency(
        self,
        conversation_turns: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Check consistency across conversation."""
```

## Conversation Management

### ConversationManager

Orchestrates conversation generation and monitoring.

```python
from src.conversation.conversation_manager import ConversationManager

class ConversationManager:
    """Manages conversation generation and flow."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        enable_safety_monitoring: bool = True,
        enable_metrics_collection: bool = True
    ):
        """
        Initialize conversation manager.
        
        Args:
            config: Configuration dictionary
            enable_safety_monitoring: Enable real-time safety monitoring
            enable_metrics_collection: Enable metrics collection
        """
    
    async def generate_conversation(
        self,
        model_client: BaseModel,
        scenario: 'Scenario',
        conversation_id: str,
        max_turns: int = 20,
        enable_branching: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete conversation.
        
        Args:
            model_client: LLM client instance
            scenario: Conversation scenario
            conversation_id: Unique conversation identifier
            max_turns: Maximum conversation turns
            enable_branching: Enable conversation branching
            
        Returns:
            Complete conversation data structure
        """
    
    def analyze_conversation_flow(
        self,
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze conversation flow and quality."""
```

### BatchProcessor

Handles batch conversation generation across multiple models and scenarios.

```python
from src.conversation.batch_processor import BatchProcessor, BatchConfig

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    conversations_per_scenario_per_model: int = 10
    max_concurrent_conversations: int = 5
    output_directory: str = "./data/conversations"
    enable_safety_monitoring: bool = True
    enable_metrics_collection: bool = True
    timeout_seconds: int = 300

class BatchProcessor:
    """Processes conversations in batches."""
    
    def __init__(self, config: BatchConfig):
        """Initialize batch processor with configuration."""
    
    async def process_batch(
        self,
        models: List[BaseModel],
        scenarios: List['Scenario'],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process batch of conversations.
        
        Args:
            models: List of model clients
            scenarios: List of conversation scenarios
            progress_callback: Optional progress callback function
            
        Returns:
            Batch processing results
        """
```

## Analysis

### StatisticalAnalyzer

Performs statistical analysis and model comparisons.

```python
from src.analysis.statistical_analysis import StatisticalAnalyzer, StatisticalResults

@dataclass
class StatisticalResults:
    """Statistical analysis results."""
    anova_results: Dict[str, Any]
    pairwise_comparisons: Dict[str, Any]
    effect_sizes: Dict[str, float]
    descriptive_stats: Dict[str, Any]
    recommendations: List[str]

class StatisticalAnalyzer:
    """Performs statistical analysis on evaluation results."""
    
    def analyze_model_comparison(
        self,
        results: Dict[str, List[CompositeScore]]
    ) -> StatisticalResults:
        """
        Analyze differences between models.
        
        Args:
            results: Dictionary mapping model names to score lists
            
        Returns:
            Statistical analysis results
        """
    
    def perform_anova(
        self,
        groups: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Perform ANOVA test on groups."""
    
    def calculate_cohens_d(
        self,
        group1: List[float],
        group2: List[float]
    ) -> float:
        """Calculate Cohen's d effect size."""
    
    def create_statistical_report(
        self,
        results: StatisticalResults
    ) -> str:
        """Generate comprehensive statistical report."""
```

### AdvancedVisualizer

Creates comprehensive visualizations and reports.

```python
from src.analysis.advanced_visualization import AdvancedVisualizer

class AdvancedVisualizer:
    """Creates advanced visualizations for analysis results."""
    
    def create_comparison_boxplots(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        group_col: str = "model"
    ) -> 'Figure':
        """Create comparison box plots."""
    
    def create_radar_chart(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        group_col: str = "model"
    ) -> 'Figure':
        """Create radar chart for multi-dimensional comparison."""
    
    def create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        metrics: List[str]
    ) -> 'Figure':
        """Create correlation heatmap."""
    
    def create_comprehensive_report(
        self,
        statistical_results: StatisticalResults,
        output_dir: str = "results/"
    ) -> str:
        """
        Create comprehensive analysis report.
        
        Args:
            statistical_results: Statistical analysis results
            output_dir: Output directory for report files
            
        Returns:
            Path to generated report
        """
```

## Scenarios

### Scenario

Data structure representing conversation scenarios.

```python
from src.scenarios.scenario import Scenario

@dataclass
class Scenario:
    """Mental health conversation scenario."""
    scenario_id: str
    title: str
    category: str
    severity: str
    
    # Patient information
    patient_profile: Dict[str, Any]
    opening_statement: str
    
    # Conversation structure
    conversation_goals: List[str]
    expected_therapeutic_elements: List[str]
    red_flags: List[str]
    conversation_flow: Dict[str, Any]
    
    # Evaluation criteria
    evaluation_criteria: Dict[str, Any]
    
    def get_initial_prompt(self) -> str:
        """Get initial conversation prompt."""
        
    def check_completion_criteria(
        self,
        conversation: Dict[str, Any]
    ) -> bool:
        """Check if conversation meets completion criteria."""
```

### ScenarioLoader

Loads and manages conversation scenarios.

```python
from src.scenarios.scenario_loader import ScenarioLoader

class ScenarioLoader:
    """Loads and manages conversation scenarios."""
    
    def __init__(self, scenarios_directory: str = "data/scenarios/"):
        """Initialize scenario loader."""
    
    def load_scenario(self, scenario_id: str) -> Scenario:
        """Load specific scenario by ID."""
    
    def load_all_scenarios(self) -> List[Scenario]:
        """Load all available scenarios."""
    
    def filter_scenarios(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[Scenario]:
        """Filter scenarios by criteria."""
    
    def validate_scenario(self, scenario: Scenario) -> Tuple[bool, List[str]]:
        """Validate scenario structure and content."""
```

## Storage

### ConversationLogger

Handles conversation data persistence.

```python
from src.storage.conversation_logger import ConversationLogger

class ConversationLogger:
    """Handles conversation data logging and storage."""
    
    def __init__(
        self,
        database_path: Optional[str] = None,
        file_output_dir: Optional[str] = None
    ):
        """Initialize conversation logger."""
    
    def log_conversation(
        self,
        conversation: Dict[str, Any],
        save_to_database: bool = True,
        save_to_file: bool = True
    ) -> bool:
        """Log conversation data."""
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation by ID."""
    
    def export_conversations(
        self,
        format: str = "json",
        output_path: str = "exported_conversations.json"
    ) -> bool:
        """Export conversations to file."""
```

### DatabaseManager

Manages database operations and schema.

```python
from src.storage.database_manager import DatabaseManager

class DatabaseManager:
    """Manages database operations for conversation storage."""
    
    def __init__(self, database_path: str):
        """Initialize database manager."""
    
    def create_tables(self) -> bool:
        """Create database tables if they don't exist."""
    
    def store_conversation(
        self,
        conversation: Dict[str, Any]
    ) -> bool:
        """Store conversation in database."""
    
    def query_conversations(
        self,
        criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query conversations by criteria."""
    
    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from stored conversations."""
```

## Utils

### LoggingConfig

Centralized logging configuration.

```python
from src.utils.logging_config import setup_logging, get_logger

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_file_logging: bool = True
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        enable_file_logging: Enable file-based logging
    """

def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
```

### Configuration

Configuration management utilities.

```python
from src.utils.config import load_config, validate_config

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
```

## Error Handling

### Custom Exceptions

```python
class EvaluationError(Exception):
    """Base exception for evaluation errors."""

class ModelError(Exception):
    """Exception for model-related errors."""

class SafetyError(Exception):
    """Exception for safety-related issues."""

class ConfigurationError(Exception):
    """Exception for configuration problems."""

class DataValidationError(Exception):
    """Exception for data validation failures."""
```

## Constants and Enums

### Evaluation Constants

```python
# Score ranges
MIN_SCORE = 0.0
MAX_SCORE = 10.0

# Performance thresholds
RESPONSE_TIME_THRESHOLD = 3.0  # seconds
THROUGHPUT_THRESHOLD = 10.0    # requests per second
RELIABILITY_THRESHOLD = 0.99   # 99% success rate

# Quality thresholds
EMPATHY_THRESHOLD = 7.0
COHERENCE_THRESHOLD = 7.5
SAFETY_THRESHOLD = 8.0
```

### Model Types

```python
class ModelType(Enum):
    """Supported model types."""
    OPENAI_GPT4 = "openai_gpt4"
    DEEPSEEK_LOCAL = "deepseek_local"
    CUSTOM = "custom"

class InferenceMode(Enum):
    """Model inference modes."""
    API = "api"
    LOCAL = "local"
    HYBRID = "hybrid"
```

## Example Usage Patterns

### Complete Evaluation Pipeline

```python
# Initialize components
openai_client = OpenAIClient()
deepseek_client = DeepSeekClient(model_path="/path/to/model")
scenario_loader = ScenarioLoader()
conversation_manager = ConversationManager(config)

# Load scenarios
scenarios = scenario_loader.load_all_scenarios()

# Generate conversations
conversations = []
for model in [openai_client, deepseek_client]:
    for scenario in scenarios:
        for i in range(10):  # 10 conversations per scenario
            conv = await conversation_manager.generate_conversation(
                model_client=model,
                scenario=scenario,
                conversation_id=f"{model.model_name}_{scenario.scenario_id}_{i}"
            )
            conversations.append(conv)

# Evaluate conversations
scorer = CompositeScorer()
evaluation_results = {}
for conv in conversations:
    model_name = conv["conversation_metadata"]["model_name"]
    if model_name not in evaluation_results:
        evaluation_results[model_name] = []
    
    score = scorer.calculate_composite_score(conv, conv["conversation_metadata"]["scenario_id"])
    evaluation_results[model_name].append(score)

# Statistical analysis
analyzer = StatisticalAnalyzer()
stats = analyzer.analyze_model_comparison(evaluation_results)

# Generate report
visualizer = AdvancedVisualizer()
report_path = visualizer.create_comprehensive_report(stats)

print(f"Evaluation complete. Report saved to: {report_path}")
```

This API reference provides comprehensive documentation for all major components of the Mental Health LLM Evaluation framework. Each class and function includes detailed parameter descriptions, return types, and usage examples to facilitate easy integration and extension.