"""
Data storage and persistence utilities for mental health LLM evaluation.

This module provides utilities for storing, retrieving, and managing
evaluation data including results, conversations, and analysis outputs.
"""

import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """Generic data storage utility with multiple backend support."""
    
    def __init__(self, storage_type: str = "file", config: Optional[Dict[str, Any]] = None):
        """
        Initialize data storage.
        
        Args:
            storage_type: Type of storage ("file", "sqlite", "memory")
            config: Storage configuration parameters
        """
        self.storage_type = storage_type
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage backend
        if storage_type == "file":
            self.base_dir = Path(self.config.get("base_dir", "./data"))
            self.base_dir.mkdir(parents=True, exist_ok=True)
        elif storage_type == "sqlite":
            self.db_path = self.config.get("db_path", "./data/evaluation.db")
            self._init_sqlite()
        elif storage_type == "memory":
            self._memory_store = {}
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    model_name TEXT,
                    timestamp TEXT,
                    data TEXT,
                    metadata TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    model_name TEXT,
                    scenario_id TEXT,
                    timestamp TEXT,
                    data TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    analysis_type TEXT,
                    timestamp TEXT,
                    data TEXT
                )
            ''')
    
    def save_data(
        self,
        data: Any,
        key: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save data with specified key and category.
        
        Args:
            data: Data to save
            key: Unique identifier for the data
            category: Data category for organization
            metadata: Optional metadata
            
        Returns:
            True if saved successfully
        """
        try:
            if self.storage_type == "file":
                return self._save_file(data, key, category, metadata)
            elif self.storage_type == "sqlite":
                return self._save_sqlite(data, key, category, metadata)
            elif self.storage_type == "memory":
                return self._save_memory(data, key, category, metadata)
        except Exception as e:
            self.logger.error(f"Error saving data {key}: {e}")
            return False
    
    def load_data(self, key: str, category: str = "general") -> Optional[Any]:
        """
        Load data by key and category.
        
        Args:
            key: Data identifier
            category: Data category
            
        Returns:
            Loaded data or None if not found
        """
        try:
            if self.storage_type == "file":
                return self._load_file(key, category)
            elif self.storage_type == "sqlite":
                return self._load_sqlite(key, category)
            elif self.storage_type == "memory":
                return self._load_memory(key, category)
        except Exception as e:
            self.logger.error(f"Error loading data {key}: {e}")
            return None
    
    def list_keys(self, category: str = "general") -> List[str]:
        """
        List all keys in a category.
        
        Args:
            category: Data category
            
        Returns:
            List of keys
        """
        try:
            if self.storage_type == "file":
                return self._list_file_keys(category)
            elif self.storage_type == "sqlite":
                return self._list_sqlite_keys(category)
            elif self.storage_type == "memory":
                return self._list_memory_keys(category)
        except Exception as e:
            self.logger.error(f"Error listing keys for {category}: {e}")
            return []
    
    def delete_data(self, key: str, category: str = "general") -> bool:
        """
        Delete data by key and category.
        
        Args:
            key: Data identifier
            category: Data category
            
        Returns:
            True if deleted successfully
        """
        try:
            if self.storage_type == "file":
                return self._delete_file(key, category)
            elif self.storage_type == "sqlite":
                return self._delete_sqlite(key, category)
            elif self.storage_type == "memory":
                return self._delete_memory(key, category)
        except Exception as e:
            self.logger.error(f"Error deleting data {key}: {e}")
            return False
    
    # File storage implementation
    def _save_file(self, data: Any, key: str, category: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Save data to file."""
        category_dir = self.base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file format based on data type
        if isinstance(data, (dict, list)):
            file_path = category_dir / f"{key}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, pd.DataFrame):
            file_path = category_dir / f"{key}.parquet"
            data.to_parquet(file_path)
        else:
            file_path = category_dir / f"{key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        
        # Save metadata if provided
        if metadata:
            metadata_path = category_dir / f"{key}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return True
    
    def _load_file(self, key: str, category: str) -> Optional[Any]:
        """Load data from file."""
        category_dir = self.base_dir / category
        
        # Try different file formats
        for ext, loader in [
            ('.json', self._load_json),
            ('.parquet', self._load_parquet),
            ('.pkl', self._load_pickle)
        ]:
            file_path = category_dir / f"{key}{ext}"
            if file_path.exists():
                return loader(file_path)
        
        return None
    
    def _load_json(self, file_path: Path) -> Any:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(file_path)
    
    def _load_pickle(self, file_path: Path) -> Any:
        """Load pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _list_file_keys(self, category: str) -> List[str]:
        """List file keys in category."""
        category_dir = self.base_dir / category
        if not category_dir.exists():
            return []
        
        keys = set()
        for file_path in category_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                key = file_path.stem
                keys.add(key)
        
        return list(keys)
    
    def _delete_file(self, key: str, category: str) -> bool:
        """Delete file data."""
        category_dir = self.base_dir / category
        deleted = False
        
        for ext in ['.json', '.parquet', '.pkl']:
            file_path = category_dir / f"{key}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True
        
        # Delete metadata
        metadata_path = category_dir / f"{key}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        return deleted
    
    # SQLite storage implementation
    def _save_sqlite(self, data: Any, key: str, category: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Save data to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            table_name = self._get_sqlite_table(category)
            data_json = json.dumps(data, default=str)
            metadata_json = json.dumps(metadata or {}, default=str)
            timestamp = datetime.now().isoformat()
            
            if category == "evaluations":
                conn.execute(
                    f"INSERT OR REPLACE INTO {table_name} (id, model_name, timestamp, data, metadata) VALUES (?, ?, ?, ?, ?)",
                    (key, metadata.get("model_name", ""), timestamp, data_json, metadata_json)
                )
            elif category == "conversations":
                conn.execute(
                    f"INSERT OR REPLACE INTO {table_name} (id, model_name, scenario_id, timestamp, data) VALUES (?, ?, ?, ?, ?)",
                    (key, metadata.get("model_name", ""), metadata.get("scenario_id", ""), timestamp, data_json)
                )
            else:
                conn.execute(
                    f"INSERT OR REPLACE INTO results (id, analysis_type, timestamp, data) VALUES (?, ?, ?, ?)",
                    (key, category, timestamp, data_json)
                )
        
        return True
    
    def _load_sqlite(self, key: str, category: str) -> Optional[Any]:
        """Load data from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            table_name = self._get_sqlite_table(category)
            
            if table_name in ["evaluations", "conversations"]:
                cursor = conn.execute(f"SELECT data FROM {table_name} WHERE id = ?", (key,))
            else:
                cursor = conn.execute("SELECT data FROM results WHERE id = ? AND analysis_type = ?", (key, category))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        
        return None
    
    def _list_sqlite_keys(self, category: str) -> List[str]:
        """List SQLite keys in category."""
        with sqlite3.connect(self.db_path) as conn:
            table_name = self._get_sqlite_table(category)
            
            if table_name in ["evaluations", "conversations"]:
                cursor = conn.execute(f"SELECT id FROM {table_name}")
            else:
                cursor = conn.execute("SELECT id FROM results WHERE analysis_type = ?", (category,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def _delete_sqlite(self, key: str, category: str) -> bool:
        """Delete SQLite data."""
        with sqlite3.connect(self.db_path) as conn:
            table_name = self._get_sqlite_table(category)
            
            if table_name in ["evaluations", "conversations"]:
                cursor = conn.execute(f"DELETE FROM {table_name} WHERE id = ?", (key,))
            else:
                cursor = conn.execute("DELETE FROM results WHERE id = ? AND analysis_type = ?", (key, category))
            
            return cursor.rowcount > 0
    
    def _get_sqlite_table(self, category: str) -> str:
        """Get SQLite table name for category."""
        if category in ["evaluation", "evaluations"]:
            return "evaluations"
        elif category in ["conversation", "conversations"]:
            return "conversations"
        else:
            return "results"
    
    # Memory storage implementation
    def _save_memory(self, data: Any, key: str, category: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Save data to memory."""
        if category not in self._memory_store:
            self._memory_store[category] = {}
        
        self._memory_store[category][key] = {
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        return True
    
    def _load_memory(self, key: str, category: str) -> Optional[Any]:
        """Load data from memory."""
        if category in self._memory_store and key in self._memory_store[category]:
            return self._memory_store[category][key]["data"]
        return None
    
    def _list_memory_keys(self, category: str) -> List[str]:
        """List memory keys in category."""
        if category in self._memory_store:
            return list(self._memory_store[category].keys())
        return []
    
    def _delete_memory(self, key: str, category: str) -> bool:
        """Delete memory data."""
        if category in self._memory_store and key in self._memory_store[category]:
            del self._memory_store[category][key]
            return True
        return False


class EvaluationDataManager:
    """Specialized data manager for evaluation results."""
    
    def __init__(self, storage: DataStorage):
        """
        Initialize evaluation data manager.
        
        Args:
            storage: Data storage backend
        """
        self.storage = storage
        self.logger = logging.getLogger(__name__)
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        session_id: str,
        model_name: str
    ) -> bool:
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results data
            session_id: Unique session identifier
            model_name: Name of evaluated model
            
        Returns:
            True if saved successfully
        """
        key = f"{session_id}_{model_name}"
        metadata = {
            "model_name": model_name,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.storage.save_data(results, key, "evaluations", metadata)
    
    def save_conversation_data(
        self,
        conversation: Dict[str, Any],
        conversation_id: str
    ) -> bool:
        """
        Save conversation data.
        
        Args:
            conversation: Conversation data
            conversation_id: Unique conversation identifier
            
        Returns:
            True if saved successfully
        """
        metadata = {
            "model_name": conversation.get("model_name"),
            "scenario_id": conversation.get("scenario_id"),
            "timestamp": conversation.get("timestamp")
        }
        
        return self.storage.save_data(conversation, conversation_id, "conversations", metadata)
    
    def save_analysis_results(
        self,
        analysis: Dict[str, Any],
        analysis_id: str,
        analysis_type: str
    ) -> bool:
        """
        Save analysis results.
        
        Args:
            analysis: Analysis results data
            analysis_id: Unique analysis identifier
            analysis_type: Type of analysis
            
        Returns:
            True if saved successfully
        """
        metadata = {
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.storage.save_data(analysis, analysis_id, analysis_type, metadata)
    
    def load_evaluation_session(self, session_id: str) -> Dict[str, Any]:
        """
        Load all evaluation results for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of evaluation results by model
        """
        results = {}
        
        # Get all evaluation keys
        eval_keys = self.storage.list_keys("evaluations")
        
        for key in eval_keys:
            if key.startswith(session_id):
                data = self.storage.load_data(key, "evaluations")
                if data:
                    model_name = key.replace(f"{session_id}_", "")
                    results[model_name] = data
        
        return results
    
    def get_model_history(self, model_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of results to return
            
        Returns:
            List of evaluation results
        """
        history = []
        
        eval_keys = self.storage.list_keys("evaluations")
        
        # Filter keys for this model and sort by timestamp
        model_keys = [key for key in eval_keys if key.endswith(f"_{model_name}")]
        
        for key in model_keys[:limit]:
            data = self.storage.load_data(key, "evaluations")
            if data:
                history.append(data)
        
        return history
    
    def export_to_csv(self, output_path: str, category: str = "evaluations") -> bool:
        """
        Export data to CSV format.
        
        Args:
            output_path: Output file path
            category: Data category to export
            
        Returns:
            True if exported successfully
        """
        try:
            keys = self.storage.list_keys(category)
            all_data = []
            
            for key in keys:
                data = self.storage.load_data(key, category)
                if data:
                    # Flatten nested dictionaries
                    flattened = self._flatten_dict(data, parent_key=key)
                    all_data.append(flattened)
            
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(output_path, index=False)
                self.logger.info(f"Exported {len(all_data)} records to {output_path}")
                return True
            else:
                self.logger.warning(f"No data found for category {category}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """
        Clean up data older than specified days.
        
        Args:
            days_old: Number of days threshold
            
        Returns:
            Number of items deleted
        """
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        for category in ["evaluations", "conversations", "results"]:
            keys = self.storage.list_keys(category)
            
            for key in keys:
                data = self.storage.load_data(key, category)
                if data and isinstance(data, dict):
                    timestamp_str = data.get("timestamp")
                    if timestamp_str:
                        try:
                            data_timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                            if data_timestamp < cutoff_date:
                                if self.storage.delete_data(key, category):
                                    deleted_count += 1
                        except ValueError:
                            continue
        
        self.logger.info(f"Cleaned up {deleted_count} old data items")
        return deleted_count