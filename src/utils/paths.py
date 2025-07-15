"""
Centralized path management for Mental Health LLM Evaluation project.

This module provides a single source of truth for all file paths in the project,
using pathlib for cross-platform compatibility and easy maintenance.
"""

import os
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class ProjectPaths:
    """Centralized path management for the project."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize project paths.
        
        Args:
            project_root: Root directory of the project. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root by looking for src/ directory
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "src").exists() and (current / "config").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                # Fallback to current working directory
                project_root = Path.cwd()
        
        self.project_root = Path(project_root).resolve()
        
        # Ensure project root exists
        if not self.project_root.exists():
            raise FileNotFoundError(f"Project root not found: {self.project_root}")
        
        # Define base directories
        self._src_dir = self.project_root / "src"
        self._data_dir = self.project_root / "data"
        self._output_dir = self.project_root / "output"
        self._config_dir = self.project_root / "config"
        self._scripts_dir = self.project_root / "scripts"
        self._docs_dir = self.project_root / "docs"
        
        # Create output directories if they don't exist
        self._ensure_output_dirs()
    
    def _ensure_output_dirs(self):
        """Ensure output directories exist."""
        output_dirs = [
            self.get_output_dir(),
            self.get_conversations_dir(),
            self.get_evaluations_dir(),
            self.get_analysis_dir(),
            self.get_visualizations_dir(),
            self.get_logs_dir(),
        ]
        
        for dir_path in output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Base directories
    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self.project_root
    
    def get_src_dir(self) -> Path:
        """Get the source code directory."""
        return self._src_dir
    
    def get_data_dir(self) -> Path:
        """Get the static data directory."""
        return self._data_dir
    
    def get_output_dir(self) -> Path:
        """Get the output directory for generated files."""
        return self._output_dir
    
    def get_config_dir(self) -> Path:
        """Get the configuration directory."""
        return self._config_dir
    
    def get_scripts_dir(self) -> Path:
        """Get the scripts directory."""
        return self._scripts_dir
    
    def get_docs_dir(self) -> Path:
        """Get the documentation directory."""
        return self._docs_dir
    
    # Data directories
    def get_scenarios_dir(self) -> Path:
        """Get the scenarios data directory."""
        return self._data_dir / "scenarios"
    
    def get_scenario_file(self, scenario_name: str) -> Path:
        """
        Get path to a specific scenario file.
        
        Args:
            scenario_name: Name of the scenario (with or without .json extension)
            
        Returns:
            Path to the scenario file
        """
        if not scenario_name.endswith('.json'):
            scenario_name += '.json'
        return self.get_scenarios_dir() / scenario_name
    
    # Output directories
    def get_conversations_dir(self) -> Path:
        """Get the conversations output directory."""
        return self._output_dir / "conversations"
    
    def get_evaluations_dir(self) -> Path:
        """Get the evaluations output directory."""
        return self._output_dir / "evaluations"
    
    def get_analysis_dir(self) -> Path:
        """Get the analysis output directory."""
        return self._output_dir / "analysis"
    
    def get_visualizations_dir(self) -> Path:
        """Get the visualizations output directory."""
        return self._output_dir / "visualizations"
    
    def get_logs_dir(self) -> Path:
        """Get the logs output directory."""
        return self._output_dir / "logs"
    
    def get_temp_dir(self) -> Path:
        """Get the temporary files directory."""
        return self._output_dir / "temp"
    
    # Configuration directories
    def get_models_config_dir(self) -> Path:
        """Get the models configuration directory."""
        return self._config_dir / "models"
    
    def get_scenarios_config_dir(self) -> Path:
        """Get the scenarios configuration directory."""
        return self._config_dir / "scenarios"
    
    # Specific file paths
    def get_main_config_file(self) -> Path:
        """Get the main configuration file."""
        return self._config_dir / "main.yaml"
    
    def get_env_file(self) -> Path:
        """Get the .env file."""
        return self.project_root / ".env"
    
    def get_env_example_file(self) -> Path:
        """Get the .env.example file."""
        return self.project_root / ".env.example"
    
    def get_readme_file(self) -> Path:
        """Get the README.md file."""
        return self.project_root / "README.md"
    
    def get_gitignore_file(self) -> Path:
        """Get the .gitignore file."""
        return self.project_root / ".gitignore"
    
    # Output file generators
    def get_evaluation_results_file(self, timestamp: Optional[str] = None) -> Path:
        """
        Get path for evaluation results file.
        
        Args:
            timestamp: Optional timestamp for the filename
            
        Returns:
            Path to evaluation results file
        """
        if timestamp:
            filename = f"evaluation_results_{timestamp}.json"
        else:
            filename = "evaluation_results.json"
        return self.get_evaluations_dir() / filename
    
    def get_analysis_report_file(self, timestamp: Optional[str] = None) -> Path:
        """
        Get path for analysis report file.
        
        Args:
            timestamp: Optional timestamp for the filename
            
        Returns:
            Path to analysis report file
        """
        if timestamp:
            filename = f"analysis_report_{timestamp}.txt"
        else:
            filename = "analysis_report.txt"
        return self.get_analysis_dir() / filename
    
    def get_statistical_analysis_file(self, timestamp: Optional[str] = None) -> Path:
        """
        Get path for statistical analysis file.
        
        Args:
            timestamp: Optional timestamp for the filename
            
        Returns:
            Path to statistical analysis file
        """
        if timestamp:
            filename = f"statistical_analysis_{timestamp}.json"
        else:
            filename = "statistical_analysis.json"
        return self.get_analysis_dir() / filename
    
    def get_model_strengths_file(self, timestamp: Optional[str] = None) -> Path:
        """
        Get path for model strengths file.
        
        Args:
            timestamp: Optional timestamp for the filename
            
        Returns:
            Path to model strengths file
        """
        if timestamp:
            filename = f"model_strengths_{timestamp}.json"
        else:
            filename = "model_strengths.json"
        return self.get_evaluations_dir() / filename
    
    def get_conversation_file(self, conversation_id: str) -> Path:
        """
        Get path for a specific conversation file.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Path to conversation file
        """
        filename = f"conversation_{conversation_id}.json"
        return self.get_conversations_dir() / filename
    
    def get_visualization_file(self, chart_name: str, extension: str = "png") -> Path:
        """
        Get path for a visualization file.
        
        Args:
            chart_name: Name of the chart
            extension: File extension (default: png)
            
        Returns:
            Path to visualization file
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        filename = f"{chart_name}{extension}"
        return self.get_visualizations_dir() / filename
    
    def get_log_file(self, log_name: str = "evaluation") -> Path:
        """
        Get path for a log file.
        
        Args:
            log_name: Name of the log file
            
        Returns:
            Path to log file
        """
        filename = f"{log_name}.log"
        return self.get_logs_dir() / filename
    
    # Utility methods
    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Path to the directory
            
        Returns:
            The path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_relative_path(self, path: Path) -> Path:
        """
        Get a path relative to the project root.
        
        Args:
            path: Absolute path
            
        Returns:
            Path relative to project root
        """
        try:
            return path.relative_to(self.project_root)
        except ValueError:
            return path
    
    def is_output_file(self, path: Path) -> bool:
        """
        Check if a path is in the output directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is in output directory
        """
        try:
            path.relative_to(self.get_output_dir())
            return True
        except ValueError:
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        temp_dir = self.get_temp_dir()
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"Cleaned up temporary files in {temp_dir}")
    
    def get_file_size(self, path: Path) -> int:
        """
        Get file size in bytes.
        
        Args:
            path: Path to the file
            
        Returns:
            File size in bytes
        """
        return path.stat().st_size if path.exists() else 0
    
    def list_output_files(self) -> list[Path]:
        """
        List all files in the output directory.
        
        Returns:
            List of file paths in output directory
        """
        output_dir = self.get_output_dir()
        if not output_dir.exists():
            return []
        
        files = []
        for item in output_dir.rglob("*"):
            if item.is_file():
                files.append(item)
        
        return sorted(files)
    
    def __str__(self) -> str:
        """String representation of the paths."""
        return f"ProjectPaths(root={self.project_root})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ProjectPaths(project_root={self.project_root!r})"


# Global instance for easy access
_paths_instance = None


def get_paths() -> ProjectPaths:
    """
    Get the global ProjectPaths instance.
    
    Returns:
        ProjectPaths instance
    """
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = ProjectPaths()
    return _paths_instance


def reset_paths(project_root: Optional[Union[str, Path]] = None):
    """
    Reset the global ProjectPaths instance.
    
    Args:
        project_root: New project root directory
    """
    global _paths_instance
    _paths_instance = ProjectPaths(project_root)


# Convenience functions for common paths
def get_project_root() -> Path:
    """Get the project root directory."""
    return get_paths().get_project_root()


def get_src_dir() -> Path:
    """Get the source code directory."""
    return get_paths().get_src_dir()


def get_data_dir() -> Path:
    """Get the static data directory."""
    return get_paths().get_data_dir()


def get_output_dir() -> Path:
    """Get the output directory."""
    return get_paths().get_output_dir()


def get_config_dir() -> Path:
    """Get the configuration directory."""
    return get_paths().get_config_dir()


def get_scenarios_dir() -> Path:
    """Get the scenarios directory."""
    return get_paths().get_scenarios_dir()


def get_evaluations_dir() -> Path:
    """Get the evaluations output directory."""
    return get_paths().get_evaluations_dir()


def get_analysis_dir() -> Path:
    """Get the analysis output directory."""
    return get_paths().get_analysis_dir()


def get_visualizations_dir() -> Path:
    """Get the visualizations output directory."""
    return get_paths().get_visualizations_dir()


def get_conversations_dir() -> Path:
    """Get the conversations output directory."""
    return get_paths().get_conversations_dir()


def get_logs_dir() -> Path:
    """Get the logs output directory."""
    return get_paths().get_logs_dir()


# Example usage
if __name__ == "__main__":
    # Demo the path management system
    paths = ProjectPaths()
    
    print("Project Paths Demo:")
    print(f"Project Root: {paths.get_project_root()}")
    print(f"Source Code: {paths.get_src_dir()}")
    print(f"Static Data: {paths.get_data_dir()}")
    print(f"Output: {paths.get_output_dir()}")
    print(f"Config: {paths.get_config_dir()}")
    print()
    
    print("Output Subdirectories:")
    print(f"Conversations: {paths.get_conversations_dir()}")
    print(f"Evaluations: {paths.get_evaluations_dir()}")
    print(f"Analysis: {paths.get_analysis_dir()}")
    print(f"Visualizations: {paths.get_visualizations_dir()}")
    print(f"Logs: {paths.get_logs_dir()}")
    print()
    
    print("Example File Paths:")
    print(f"Main Config: {paths.get_main_config_file()}")
    print(f"Scenario File: {paths.get_scenario_file('anxiety_001')}")
    print(f"Evaluation Results: {paths.get_evaluation_results_file('20240101_120000')}")
    print(f"Analysis Report: {paths.get_analysis_report_file('20240101_120000')}")
    print(f"Visualization: {paths.get_visualization_file('overall_comparison')}")