"""
Configuration schema and validation for mental health LLM evaluation.

This module defines the complete configuration schema with validation rules
for all components of the evaluation system.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Valid logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Valid logging formats."""
    STANDARD = "standard"
    DETAILED = "detailed"
    JSON = "json"


class StorageType(str, Enum):
    """Valid storage backend types."""
    FILE = "file"
    SQLITE = "sqlite"
    MEMORY = "memory"


class ModelType(str, Enum):
    """Valid model types."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class OpenAIConfig:
    """OpenAI model configuration."""
    api_key: Optional[str] = None  # Should come from environment
    organization_id: Optional[str] = None
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate OpenAI configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("OpenAI temperature must be between 0 and 2")
        if not 1 <= self.max_tokens <= 8192:
            raise ValueError("OpenAI max_tokens must be between 1 and 8192")
        if not 0 <= self.top_p <= 1:
            raise ValueError("OpenAI top_p must be between 0 and 1")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("OpenAI frequency_penalty must be between -2.0 and 2.0")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("OpenAI presence_penalty must be between -2.0 and 2.0")
        if self.timeout <= 0:
            raise ValueError("OpenAI timeout must be positive")


@dataclass
class DeepSeekConfig:
    """DeepSeek model configuration."""
    api_key: Optional[str] = None  # For API mode
    use_api: bool = False
    api_url: str = "https://api.deepseek.com"
    model_path: str = "./models/deepseek-llm-7b-chat"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    temperature: float = 0.7
    max_new_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    timeout: float = 60.0
    
    def __post_init__(self):
        """Validate DeepSeek configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("DeepSeek temperature must be between 0 and 2")
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("DeepSeek max_new_tokens must be between 1 and 4096")
        if not 0 <= self.top_p <= 1:
            raise ValueError("DeepSeek top_p must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("DeepSeek top_k must be positive")
        if self.repetition_penalty < 1.0:
            raise ValueError("DeepSeek repetition_penalty must be >= 1.0")
        if self.device not in ["cuda", "cpu", "auto"]:
            raise ValueError("DeepSeek device must be 'cuda', 'cpu', or 'auto'")


@dataclass
class ModelsConfig:
    """Model configurations."""
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    enabled_models: List[ModelType] = field(default_factory=lambda: [ModelType.OPENAI, ModelType.DEEPSEEK])
    
    def __post_init__(self):
        """Validate models configuration."""
        if not self.enabled_models:
            raise ValueError("At least one model must be enabled")
        
        valid_models = set(ModelType)
        for model in self.enabled_models:
            if model not in valid_models:
                raise ValueError(f"Invalid model type: {model}")


@dataclass
class TechnicalMetricsConfig:
    """Technical evaluation metrics configuration."""
    max_response_time_ms: float = 5000.0
    target_throughput_rps: float = 10.0
    concurrent_requests: int = 5
    test_duration_seconds: int = 60
    warmup_requests: int = 5
    timeout_threshold: float = 30.0
    
    # Scoring weights (must sum to 1.0)
    response_time_weight: float = 0.3
    throughput_weight: float = 0.25
    reliability_weight: float = 0.3
    efficiency_weight: float = 0.15
    
    def __post_init__(self):
        """Validate technical metrics configuration."""
        if self.max_response_time_ms <= 0:
            raise ValueError("max_response_time_ms must be positive")
        if self.target_throughput_rps <= 0:
            raise ValueError("target_throughput_rps must be positive")
        if self.concurrent_requests < 1:
            raise ValueError("concurrent_requests must be at least 1")
        if self.test_duration_seconds < 1:
            raise ValueError("test_duration_seconds must be at least 1")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        
        # Validate weights sum to 1.0
        total_weight = (
            self.response_time_weight + self.throughput_weight +
            self.reliability_weight + self.efficiency_weight
        )
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Technical metrics weights must sum to 1.0, got {total_weight}")


@dataclass
class TherapeuticMetricsConfig:
    """Therapeutic evaluation metrics configuration."""
    # Scoring weights (must sum to 1.0)
    empathy_weight: float = 0.3
    coherence_weight: float = 0.25
    safety_weight: float = 0.35
    boundaries_weight: float = 0.1
    
    # Safety thresholds
    min_safety_score: float = 80.0
    crisis_detection_enabled: bool = True
    content_filtering_enabled: bool = True
    
    def __post_init__(self):
        """Validate therapeutic metrics configuration."""
        # Validate weights sum to 1.0
        total_weight = (
            self.empathy_weight + self.coherence_weight +
            self.safety_weight + self.boundaries_weight
        )
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Therapeutic metrics weights must sum to 1.0, got {total_weight}")
        
        if not 0 <= self.min_safety_score <= 100:
            raise ValueError("min_safety_score must be between 0 and 100")


@dataclass
class PatientExperienceConfig:
    """Patient experience metrics configuration."""
    # Scoring weights (must sum to 1.0)
    satisfaction_weight: float = 0.4
    engagement_weight: float = 0.25
    trust_weight: float = 0.25
    accessibility_weight: float = 0.1
    
    # Experience thresholds
    min_satisfaction_score: float = 70.0
    min_trust_score: float = 75.0
    
    def __post_init__(self):
        """Validate patient experience configuration."""
        # Validate weights sum to 1.0
        total_weight = (
            self.satisfaction_weight + self.engagement_weight +
            self.trust_weight + self.accessibility_weight
        )
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Patient experience weights must sum to 1.0, got {total_weight}")
        
        if not 0 <= self.min_satisfaction_score <= 100:
            raise ValueError("min_satisfaction_score must be between 0 and 100")
        if not 0 <= self.min_trust_score <= 100:
            raise ValueError("min_trust_score must be between 0 and 100")


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    technical: TechnicalMetricsConfig = field(default_factory=TechnicalMetricsConfig)
    therapeutic: TherapeuticMetricsConfig = field(default_factory=TherapeuticMetricsConfig)
    patient: PatientExperienceConfig = field(default_factory=PatientExperienceConfig)
    
    # Composite scoring weights (must sum to 1.0)
    technical_weight: float = 0.3
    therapeutic_weight: float = 0.5
    patient_weight: float = 0.2
    
    # Score thresholds
    production_ready_threshold: float = 80.0
    clinical_ready_threshold: float = 90.0
    research_acceptable_threshold: float = 70.0
    minimum_viable_threshold: float = 60.0
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        # Validate composite weights sum to 1.0
        total_weight = self.technical_weight + self.therapeutic_weight + self.patient_weight
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Evaluation composite weights must sum to 1.0, got {total_weight}")
        
        # Validate thresholds are in order
        thresholds = [
            self.minimum_viable_threshold,
            self.research_acceptable_threshold,
            self.production_ready_threshold,
            self.clinical_ready_threshold
        ]
        
        for i in range(len(thresholds)):
            if not 0 <= thresholds[i] <= 100:
                raise ValueError(f"Threshold {i} must be between 0 and 100")
            if i > 0 and thresholds[i] < thresholds[i-1]:
                raise ValueError("Thresholds must be in ascending order")


@dataclass
class ConversationConfig:
    """Conversation generation configuration."""
    max_turns: int = 10
    min_turns: int = 3
    turn_timeout: float = 30.0
    conversation_timeout: float = 300.0
    user_response_probability: float = 0.8
    user_elaboration_probability: float = 0.6
    conversation_end_probability: float = 0.1
    
    def __post_init__(self):
        """Validate conversation configuration."""
        if self.max_turns < self.min_turns:
            raise ValueError("max_turns must be >= min_turns")
        if self.min_turns < 1:
            raise ValueError("min_turns must be at least 1")
        if self.turn_timeout <= 0:
            raise ValueError("turn_timeout must be positive")
        if self.conversation_timeout <= 0:
            raise ValueError("conversation_timeout must be positive")
        if not 0 <= self.user_response_probability <= 1:
            raise ValueError("user_response_probability must be between 0 and 1")
        if not 0 <= self.user_elaboration_probability <= 1:
            raise ValueError("user_elaboration_probability must be between 0 and 1")
        if not 0 <= self.conversation_end_probability <= 1:
            raise ValueError("conversation_end_probability must be between 0 and 1")


@dataclass
class ScenarioConfig:
    """Scenario configuration."""
    default_suite: str = "comprehensive"
    scenarios_dir: str = "./data/scenarios"
    available_suites: List[str] = field(default_factory=lambda: [
        "basic", "comprehensive", "safety", "empathy"
    ])
    categories: List[str] = field(default_factory=lambda: [
        "anxiety", "depression", "trauma", "general_support"
    ])
    severity_levels: List[str] = field(default_factory=lambda: [
        "mild", "moderate", "severe"
    ])
    
    def __post_init__(self):
        """Validate scenario configuration."""
        if self.default_suite not in self.available_suites:
            raise ValueError(f"default_suite '{self.default_suite}' not in available_suites")


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    conversation_count: int = 5
    scenario_suite: str = "comprehensive"
    conversations_per_scenario: int = 2
    parallel_evaluations: bool = True
    max_parallel_workers: int = 4
    random_seed: Optional[int] = None
    enable_warmup: bool = True
    
    # Data collection settings
    save_conversations: bool = True
    save_intermediate_results: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if self.conversation_count < 1:
            raise ValueError("conversation_count must be at least 1")
        if self.conversations_per_scenario < 1:
            raise ValueError("conversations_per_scenario must be at least 1")
        if self.max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be at least 1")
        
        valid_formats = ["json", "csv", "parquet", "xlsx"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}")


@dataclass
class StatisticalConfig:
    """Statistical analysis configuration."""
    alpha: float = 0.05
    confidence_level: float = 0.95
    min_sample_size: int = 5
    bonferroni_correction: bool = True
    
    # Effect size thresholds
    small_effect_size: float = 0.2
    medium_effect_size: float = 0.5
    large_effect_size: float = 0.8
    
    def __post_init__(self):
        """Validate statistical configuration."""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.min_sample_size < 1:
            raise ValueError("min_sample_size must be at least 1")
        
        # Validate effect size thresholds are in order
        if not (0 < self.small_effect_size < self.medium_effect_size < self.large_effect_size):
            raise ValueError("Effect size thresholds must be in ascending order")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.STANDARD
    file_path: Optional[str] = "./logs/evaluation.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_structured: bool = False
    
    # Logger-specific levels
    external_loggers: Dict[str, LogLevel] = field(default_factory=lambda: {
        "urllib3": LogLevel.WARNING,
        "requests": LogLevel.WARNING,
        "matplotlib": LogLevel.WARNING,
        "transformers": LogLevel.WARNING,
        "torch": LogLevel.WARNING,
    })
    
    def __post_init__(self):
        """Validate logging configuration."""
        if self.max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be at least 1")
        if self.backup_count < 0:
            raise ValueError("backup_count must be non-negative")


@dataclass
class StorageConfig:
    """Storage configuration."""
    type: StorageType = StorageType.FILE
    base_dir: str = "./data"
    database_path: str = "./data/evaluation.db"
    backup_enabled: bool = True
    cleanup_days: int = 30
    compression_enabled: bool = False
    
    def __post_init__(self):
        """Validate storage configuration."""
        if self.cleanup_days < 0:
            raise ValueError("cleanup_days must be non-negative")


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "./results"
    evaluations_dir: str = "./results/evaluations"
    reports_dir: str = "./results/reports"
    statistics_dir: str = "./results/statistics"
    visualizations_dir: str = "./results/visualizations"
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    include_timestamp: bool = True
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    generate_visualizations: bool = True
    create_summary_report: bool = True
    
    def __post_init__(self):
        """Validate output configuration."""
        valid_formats = ["json", "csv", "parquet", "xlsx", "html"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}")


@dataclass
class ConfigSchema:
    """Main configuration schema."""
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Global settings
    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate main configuration."""
        valid_environments = ["development", "production", "testing"]
        if self.environment not in valid_environments:
            raise ValueError(f"environment must be one of {valid_environments}")


def validate_config(config_dict: Dict[str, Any]) -> ConfigSchema:
    """
    Validate configuration dictionary and return ConfigSchema instance.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Validated ConfigSchema instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Create nested configurations
        models_config = ModelsConfig(
            openai=OpenAIConfig(**config_dict.get("models", {}).get("openai", {})),
            deepseek=DeepSeekConfig(**config_dict.get("models", {}).get("deepseek", {})),
            enabled_models=config_dict.get("models", {}).get("enabled_models", [ModelType.OPENAI, ModelType.DEEPSEEK])
        )
        
        eval_dict = config_dict.get("evaluation", {})
        evaluation_config = EvaluationConfig(
            technical=TechnicalMetricsConfig(**eval_dict.get("technical", {})),
            therapeutic=TherapeuticMetricsConfig(**eval_dict.get("therapeutic", {})),
            patient=PatientExperienceConfig(**eval_dict.get("patient", {})),
            **{k: v for k, v in eval_dict.items() if k not in ["technical", "therapeutic", "patient"]}
        )
        
        # Create main config
        config = ConfigSchema(
            models=models_config,
            evaluation=evaluation_config,
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            conversation=ConversationConfig(**config_dict.get("conversation", {})),
            scenario=ScenarioConfig(**config_dict.get("scenario", {})),
            statistical=StatisticalConfig(**config_dict.get("statistical", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            storage=StorageConfig(**config_dict.get("storage", {})),
            output=OutputConfig(**config_dict.get("results", {})),
            **{k: v for k, v in config_dict.items() if k not in [
                "models", "evaluation", "experiment", "conversation", "scenario",
                "statistical", "logging", "storage", "results"
            ]}
        )
        
        logger.info("Configuration validation successful")
        return config
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}")


def get_default_config() -> ConfigSchema:
    """Get default configuration."""
    return ConfigSchema()