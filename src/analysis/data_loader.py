"""
Data Loading and Processing Pipeline for LLM Performance Analysis

This module handles loading conversation data from various sources,
preprocessing, validation, and preparation for statistical analysis.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality and completeness."""
    
    total_conversations: int = 0
    valid_conversations: int = 0
    missing_data_counts: Dict[str, int] = field(default_factory=dict)
    outlier_counts: Dict[str, int] = field(default_factory=dict)
    data_quality_score: float = 0.0
    
    # Validation results
    schema_violations: List[str] = field(default_factory=list)
    value_range_violations: List[str] = field(default_factory=list)
    logical_inconsistencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_conversations": self.total_conversations,
            "valid_conversations": self.valid_conversations,
            "data_completeness": self.valid_conversations / max(1, self.total_conversations),
            "missing_data_counts": self.missing_data_counts,
            "outlier_counts": self.outlier_counts,
            "data_quality_score": self.data_quality_score,
            "validation_issues": {
                "schema_violations": self.schema_violations,
                "value_range_violations": self.value_range_violations,
                "logical_inconsistencies": self.logical_inconsistencies
            }
        }


class ConversationDataLoader:
    """
    Comprehensive data loader for conversation analysis with validation,
    quality checks, and preprocessing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration for data loading and processing
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.validate_schema = self.config.get("validate_schema", True)
        self.handle_outliers = self.config.get("handle_outliers", True)
        self.impute_missing = self.config.get("impute_missing", True)
        self.quality_threshold = self.config.get("quality_threshold", 0.8)
        
        # Data validation rules
        self.validation_rules = {
            "conversation_id": {"required": True, "type": str},
            "model_name": {"required": True, "type": str},
            "scenario_id": {"required": True, "type": str},
            "total_turns": {"required": True, "type": int, "min": 1, "max": 50},
            "avg_response_time_ms": {"required": False, "type": float, "min": 0, "max": 60000},
            "total_tokens": {"required": False, "type": int, "min": 0, "max": 50000},
            "safety_flags_count": {"required": False, "type": int, "min": 0, "max": 100},
            "avg_quality_score": {"required": False, "type": float, "min": 0, "max": 10},
            "termination_reason": {"required": False, "type": str},
            "conversation_flow_rating": {"required": False, "type": float, "min": 0, "max": 10}
        }
        
        # Expected metrics columns
        self.expected_metrics = [
            "conversation_id", "model_name", "scenario_id", "scenario_type", "severity_level",
            "total_turns", "assistant_turns", "user_turns", "conversation_duration_ms",
            "avg_response_time_ms", "min_response_time_ms", "max_response_time_ms",
            "total_tokens", "prompt_tokens", "completion_tokens", "avg_tokens_per_response",
            "safety_flags_count", "crisis_interventions", "avg_quality_score",
            "empathy_score", "coherence_score", "therapeutic_effectiveness",
            "conversation_flow_rating", "natural_ending", "termination_reason",
            "error_count", "timeout_count", "model_type", "api_cost"
        ]
    
    def load_from_json_directory(
        self,
        directory_path: Path,
        file_pattern: str = "*.json"
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Load conversation data from directory of JSON files.
        
        Args:
            directory_path: Path to directory containing JSON files
            file_pattern: File pattern to match (default: *.json)
            
        Returns:
            Tuple of (DataFrame with conversation data, Data quality report)
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        json_files = list(directory_path.glob(file_pattern))
        if not json_files:
            raise ValueError(f"No JSON files found in {directory_path} matching {file_pattern}")
        
        self.logger.info(f"Loading data from {len(json_files)} JSON files")
        
        conversations = []
        quality_report = DataQualityReport()
        quality_report.total_conversations = len(json_files)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract conversation data
                conversation_data = self._extract_conversation_metrics(data)
                
                # Validate data
                if self.validate_schema:
                    validation_issues = self._validate_conversation_data(conversation_data)
                    if validation_issues:
                        quality_report.schema_violations.extend(validation_issues)
                        continue
                
                conversations.append(conversation_data)
                quality_report.valid_conversations += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")
                quality_report.schema_violations.append(f"Failed to load {json_file.name}: {str(e)}")
        
        # Convert to DataFrame
        if not conversations:
            raise ValueError("No valid conversations could be loaded")
        
        df = pd.DataFrame(conversations)
        
        # Data quality assessment and preprocessing
        df, quality_report = self._preprocess_dataframe(df, quality_report)
        
        self.logger.info(
            f"Loaded {len(df)} valid conversations from {len(json_files)} files "
            f"(quality score: {quality_report.data_quality_score:.2f})"
        )
        
        return df, quality_report
    
    def load_from_database(
        self,
        db_path: Path,
        query: Optional[str] = None
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Load conversation data from SQLite database.
        
        Args:
            db_path: Path to SQLite database
            query: Optional custom SQL query
            
        Returns:
            Tuple of (DataFrame with conversation data, Data quality report)
        """
        db_path = Path(db_path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Default query to get comprehensive conversation data
        if query is None:
            query = """
            SELECT 
                c.conversation_id,
                c.scenario_id,
                c.model_name,
                c.start_time,
                c.end_time,
                c.total_turns,
                c.total_duration_ms,
                c.termination_reason,
                c.safety_flags_count,
                c.avg_response_time_ms,
                c.avg_quality_score,
                a.total_words,
                a.unique_words,
                a.avg_empathy_score,
                a.avg_coherence_score,
                a.conversation_flow_rating,
                a.natural_ending,
                json_extract(c.conversation_data, '$.metrics.total_tokens') as total_tokens,
                json_extract(c.conversation_data, '$.metrics.prompt_tokens') as prompt_tokens,
                json_extract(c.conversation_data, '$.metrics.completion_tokens') as completion_tokens,
                json_extract(c.conversation_data, '$.metrics.api_errors') as api_errors,
                json_extract(c.conversation_data, '$.metrics.timeout_errors') as timeout_errors
            FROM conversations c
            LEFT JOIN conversation_analytics a ON c.conversation_id = a.conversation_id
            """
        
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn)
            
            quality_report = DataQualityReport()
            quality_report.total_conversations = len(df)
            quality_report.valid_conversations = len(df)
            
            # Process and validate the data
            df, quality_report = self._preprocess_dataframe(df, quality_report)
            
            self.logger.info(f"Loaded {len(df)} conversations from database")
            
            return df, quality_report
            
        except Exception as e:
            self.logger.error(f"Failed to load from database: {e}")
            raise
    
    def load_from_csv(
        self,
        csv_path: Path,
        delimiter: str = ","
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Load conversation data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            delimiter: CSV delimiter (default: comma)
            
        Returns:
            Tuple of (DataFrame with conversation data, Data quality report)
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, delimiter=delimiter)
            
            quality_report = DataQualityReport()
            quality_report.total_conversations = len(df)
            quality_report.valid_conversations = len(df)
            
            # Process and validate the data
            df, quality_report = self._preprocess_dataframe(df, quality_report)
            
            self.logger.info(f"Loaded {len(df)} conversations from CSV")
            
            return df, quality_report
            
        except Exception as e:
            self.logger.error(f"Failed to load from CSV: {e}")
            raise
    
    def _extract_conversation_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized metrics from conversation JSON data."""
        
        # Get conversation metadata
        metadata = data.get("conversation_metadata", {})
        scenario_data = data.get("scenario_data", {})
        analytics_data = data.get("analytics_data", {})
        
        # Extract core metrics
        conversation_metrics = {
            "conversation_id": metadata.get("conversation_id"),
            "model_name": metadata.get("model_name"),
            "scenario_id": metadata.get("scenario_id"),
            "scenario_type": scenario_data.get("scenario_type"),
            "severity_level": scenario_data.get("severity_level"),
            
            # Conversation structure
            "total_turns": metadata.get("total_turns", 0),
            "assistant_turns": metadata.get("metrics", {}).get("assistant_turns", 0),
            "user_turns": metadata.get("metrics", {}).get("user_turns", 0),
            
            # Timing metrics
            "conversation_duration_ms": metadata.get("metrics", {}).get("total_conversation_time_ms", 0),
            "avg_response_time_ms": metadata.get("metrics", {}).get("avg_response_time_ms", 0),
            "min_response_time_ms": metadata.get("metrics", {}).get("min_response_time_ms"),
            "max_response_time_ms": metadata.get("metrics", {}).get("max_response_time_ms"),
            
            # Token metrics
            "total_tokens": metadata.get("metrics", {}).get("total_tokens", 0),
            "prompt_tokens": metadata.get("metrics", {}).get("prompt_tokens", 0),
            "completion_tokens": metadata.get("metrics", {}).get("completion_tokens", 0),
            "avg_tokens_per_response": metadata.get("metrics", {}).get("avg_tokens_per_response", 0),
            
            # Safety metrics
            "safety_flags_count": len(metadata.get("safety_flags_total", [])),
            "crisis_interventions": analytics_data.get("crisis_intervention_triggered", False),
            
            # Quality metrics
            "avg_quality_score": metadata.get("metrics", {}).get("avg_quality_score"),
            "empathy_score": np.mean(analytics_data.get("empathy_scores", [])) if analytics_data.get("empathy_scores") else None,
            "coherence_score": np.mean(analytics_data.get("coherence_scores", [])) if analytics_data.get("coherence_scores") else None,
            "conversation_flow_rating": analytics_data.get("conversation_flow_rating"),
            "natural_ending": analytics_data.get("natural_ending_achieved", False),
            
            # Outcome metrics
            "termination_reason": metadata.get("termination_reason"),
            "error_count": len(metadata.get("errors", [])),
            
            # Content metrics
            "total_words": analytics_data.get("total_words"),
            "unique_words": analytics_data.get("unique_words"),
            
            # Model type
            "model_type": self._infer_model_type(metadata.get("model_name", "")),
            
            # Timestamps
            "start_time": metadata.get("start_time"),
            "end_time": metadata.get("end_time")
        }
        
        # Calculate derived metrics
        if conversation_metrics["total_tokens"] and conversation_metrics["assistant_turns"]:
            conversation_metrics["avg_tokens_per_response"] = (
                conversation_metrics["completion_tokens"] / conversation_metrics["assistant_turns"]
            )
        
        # Calculate therapeutic effectiveness score
        therapeutic_elements = analytics_data.get("therapeutic_element_scores", {})
        if therapeutic_elements:
            therapeutic_effectiveness = np.mean([
                therapeutic_elements.get("validation", 0),
                therapeutic_elements.get("active_listening", 0),
                therapeutic_elements.get("psychoeducation", 0),
                therapeutic_elements.get("coping_strategies", 0)
            ])
            conversation_metrics["therapeutic_effectiveness"] = therapeutic_effectiveness
        
        return conversation_metrics
    
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name."""
        model_name_lower = model_name.lower()
        
        if "gpt" in model_name_lower or "openai" in model_name_lower:
            return "openai"
        elif "deepseek" in model_name_lower:
            return "deepseek"
        elif "claude" in model_name_lower:
            return "anthropic"
        elif "llama" in model_name_lower:
            return "meta"
        else:
            return "unknown"
    
    def _validate_conversation_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate conversation data against schema rules."""
        
        issues = []
        
        for field, rules in self.validation_rules.items():
            value = data.get(field)
            
            # Check required fields
            if rules.get("required", False) and value is None:
                issues.append(f"Missing required field: {field}")
                continue
            
            if value is not None:
                # Check type
                expected_type = rules.get("type")
                if expected_type and not isinstance(value, expected_type):
                    issues.append(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(value).__name__}")
                
                # Check numeric ranges
                if isinstance(value, (int, float)):
                    min_val = rules.get("min")
                    max_val = rules.get("max")
                    
                    if min_val is not None and value < min_val:
                        issues.append(f"Value too low for {field}: {value} < {min_val}")
                    
                    if max_val is not None and value > max_val:
                        issues.append(f"Value too high for {field}: {value} > {max_val}")
        
        # Logical consistency checks
        total_turns = data.get("total_turns", 0)
        assistant_turns = data.get("assistant_turns", 0)
        user_turns = data.get("user_turns", 0)
        
        if assistant_turns + user_turns > 0 and abs(total_turns - (assistant_turns + user_turns)) > 1:
            issues.append("Inconsistent turn counts: total_turns != assistant_turns + user_turns")
        
        return issues
    
    def _preprocess_dataframe(
        self,
        df: pd.DataFrame,
        quality_report: DataQualityReport
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Preprocess DataFrame with cleaning, validation, and imputation."""
        
        original_count = len(df)
        
        # Convert timestamps
        for col in ["start_time", "end_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle missing data
        missing_counts = df.isnull().sum()
        quality_report.missing_data_counts = missing_counts[missing_counts > 0].to_dict()
        
        # Remove rows with too much missing data
        critical_columns = ["conversation_id", "model_name", "scenario_id", "total_turns"]
        df = df.dropna(subset=critical_columns)
        
        # Impute missing values if enabled
        if self.impute_missing:
            df = self._impute_missing_values(df)
        
        # Handle outliers if enabled
        if self.handle_outliers:
            df, outlier_counts = self._handle_outliers(df)
            quality_report.outlier_counts = outlier_counts
        
        # Add derived metrics
        df = self._add_derived_metrics(df)
        
        # Calculate data quality score
        completeness = len(df) / original_count
        missing_ratio = sum(quality_report.missing_data_counts.values()) / (len(df) * len(df.columns))
        quality_report.data_quality_score = completeness * (1 - missing_ratio)
        
        # Ensure all expected columns are present
        for col in self.expected_metrics:
            if col not in df.columns:
                df[col] = np.nan
        
        # Sort by conversation_id for consistency
        df = df.sort_values("conversation_id").reset_index(drop=True)
        
        self.logger.info(f"Preprocessing complete: {len(df)}/{original_count} conversations retained")
        
        return df, quality_report
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using appropriate strategies."""
        
        # Numeric columns - use median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        
        # Categorical columns - use mode or default
        categorical_defaults = {
            "termination_reason": "unknown",
            "model_type": "unknown",
            "scenario_type": "unknown",
            "severity_level": "unknown"
        }
        
        for col, default_value in categorical_defaults.items():
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else default_value
                df[col].fillna(fill_value, inplace=True)
        
        # Boolean columns - use False as default
        boolean_columns = ["natural_ending", "crisis_interventions"]
        for col in boolean_columns:
            if col in df.columns:
                df[col].fillna(False, inplace=True)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Detect and handle outliers using IQR method."""
        
        outlier_counts = {}
        
        # Define columns to check for outliers
        outlier_columns = [
            "total_turns", "conversation_duration_ms", "avg_response_time_ms",
            "total_tokens", "avg_quality_score", "empathy_score", "coherence_score"
        ]
        
        for col in outlier_columns:
            if col in df.columns and df[col].notna().any():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_counts[col] = outlier_count
                    
                    # Cap outliers instead of removing them
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df, outlier_counts
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics calculated from base metrics."""
        
        # Conversation efficiency (turns per minute)
        if "total_turns" in df.columns and "conversation_duration_ms" in df.columns:
            df["turns_per_minute"] = (
                df["total_turns"] / (df["conversation_duration_ms"] / 60000)
            ).replace([np.inf, -np.inf], np.nan)
        
        # Token efficiency (tokens per turn)
        if "total_tokens" in df.columns and "total_turns" in df.columns:
            df["tokens_per_turn"] = (
                df["total_tokens"] / df["total_turns"]
            ).replace([np.inf, -np.inf], np.nan)
        
        # Safety risk score
        if "safety_flags_count" in df.columns and "total_turns" in df.columns:
            df["safety_risk_score"] = (
                df["safety_flags_count"] / df["total_turns"]
            ).replace([np.inf, -np.inf], np.nan)
        
        # Overall quality composite score
        quality_components = []
        weights = {}
        
        if "avg_quality_score" in df.columns:
            quality_components.append("avg_quality_score")
            weights["avg_quality_score"] = 0.3
        
        if "empathy_score" in df.columns:
            quality_components.append("empathy_score")
            weights["empathy_score"] = 0.25
        
        if "coherence_score" in df.columns:
            quality_components.append("coherence_score")
            weights["coherence_score"] = 0.25
        
        if "conversation_flow_rating" in df.columns:
            quality_components.append("conversation_flow_rating")
            weights["conversation_flow_rating"] = 0.2
        
        if quality_components:
            # Normalize scores to 0-1 scale
            normalized_scores = pd.DataFrame()
            for component in quality_components:
                if component in df.columns:
                    col_data = df[component].copy()
                    col_min = col_data.min()
                    col_max = col_data.max()
                    if col_max > col_min:
                        normalized_scores[component] = (col_data - col_min) / (col_max - col_min)
                    else:
                        normalized_scores[component] = 0.5  # Neutral score if no variation
            
            # Calculate weighted composite score
            df["composite_quality_score"] = 0
            for component in quality_components:
                if component in normalized_scores.columns:
                    df["composite_quality_score"] += (
                        normalized_scores[component] * weights.get(component, 1.0 / len(quality_components))
                    )
        
        return df
    
    def export_processed_data(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format: str = "csv"
    ):
        """
        Export processed data to file.
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            format: Export format (csv, json, parquet)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format.lower() == "parquet":
            df.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(df)} conversations to {output_path}")


# Convenience functions for common loading scenarios
def load_conversation_data(
    source_path: Path,
    source_type: str = "auto",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Load conversation data from various sources with automatic type detection.
    
    Args:
        source_path: Path to data source
        source_type: Source type (auto, json_dir, database, csv)
        config: Optional configuration
        
    Returns:
        Tuple of (DataFrame, DataQualityReport)
    """
    loader = ConversationDataLoader(config)
    source_path = Path(source_path)
    
    # Auto-detect source type
    if source_type == "auto":
        if source_path.is_dir():
            source_type = "json_dir"
        elif source_path.suffix.lower() == ".db":
            source_type = "database"
        elif source_path.suffix.lower() == ".csv":
            source_type = "csv"
        else:
            raise ValueError(f"Cannot auto-detect source type for: {source_path}")
    
    # Load based on type
    if source_type == "json_dir":
        return loader.load_from_json_directory(source_path)
    elif source_type == "database":
        return loader.load_from_database(source_path)
    elif source_type == "csv":
        return loader.load_from_csv(source_path)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")