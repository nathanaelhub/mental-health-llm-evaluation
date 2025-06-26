"""
Unit Tests for Data Storage and Retrieval Functions

Tests for database operations, file storage, conversation logging,
data serialization, and retrieval functionality.
"""

import pytest
import sqlite3
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from conversation.conversation_logger import ConversationLogger, DatabaseLogger, FileLogger
from storage.database_manager import DatabaseManager
from storage.file_manager import FileManager
from analysis.data_loader import ConversationDataLoader


@pytest.mark.unit
class TestConversationLogger:
    """Test conversation logging functionality."""
    
    @pytest.fixture
    def conversation_logger(self, temp_test_dir):
        """Create conversation logger instance."""
        return ConversationLogger(
            database_path=temp_test_dir / "test_conversations.db",
            file_output_dir=temp_test_dir / "conversations"
        )
    
    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation data for testing."""
        return {
            "conversation_metadata": {
                "conversation_id": "test_conv_001",
                "scenario_id": "MH-TEST-001",
                "model_name": "test_model",
                "start_time": datetime.now().isoformat(),
                "end_time": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "total_turns": 8,
                "termination_reason": "natural_ending"
            },
            "conversation_turns": [
                {
                    "turn_number": 1,
                    "speaker": "patient",
                    "message": "I'm feeling anxious about work.",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "turn_number": 2,
                    "speaker": "assistant",
                    "message": "That sounds challenging. Can you tell me more?",
                    "timestamp": datetime.now().isoformat(),
                    "response_time_ms": 2400
                }
            ],
            "analytics_data": {
                "empathy_scores": [8.0, 7.5],
                "safety_flags": [],
                "conversation_flow_rating": 8.2
            },
            "evaluation_scores": {
                "overall_score": 7.8,
                "empathy_score": 8.0,
                "safety_score": 9.0,
                "coherence_score": 7.5
            }
        }
    
    def test_conversation_logger_initialization(self, conversation_logger):
        """Test conversation logger initialization."""
        assert conversation_logger is not None
        assert hasattr(conversation_logger, 'log_conversation')
        assert hasattr(conversation_logger, 'get_conversation')
        assert hasattr(conversation_logger, 'export_conversations')
    
    def test_log_conversation_to_database(self, conversation_logger, sample_conversation):
        """Test logging conversation to database."""
        success = conversation_logger.log_conversation(sample_conversation)
        
        assert success == True
        
        # Verify conversation was stored
        retrieved = conversation_logger.get_conversation("test_conv_001")
        assert retrieved is not None
        assert retrieved["conversation_metadata"]["conversation_id"] == "test_conv_001"
    
    def test_log_conversation_to_file(self, conversation_logger, sample_conversation, temp_test_dir):
        """Test logging conversation to file."""
        success = conversation_logger.log_conversation(sample_conversation, save_to_file=True)
        
        assert success == True
        
        # Check that file was created
        conversation_files = list((temp_test_dir / "conversations").glob("*.json"))
        assert len(conversation_files) > 0
        
        # Verify file content
        with open(conversation_files[0], 'r') as f:
            file_data = json.load(f)
        
        assert file_data["conversation_metadata"]["conversation_id"] == "test_conv_001"
    
    def test_get_conversation_by_id(self, conversation_logger, sample_conversation):
        """Test retrieving conversation by ID."""
        # Store conversation first
        conversation_logger.log_conversation(sample_conversation)
        
        # Retrieve conversation
        retrieved = conversation_logger.get_conversation("test_conv_001")
        
        assert retrieved is not None
        assert retrieved["conversation_metadata"]["conversation_id"] == "test_conv_001"
        assert len(retrieved["conversation_turns"]) == 2
    
    def test_get_nonexistent_conversation(self, conversation_logger):
        """Test retrieving nonexistent conversation."""
        retrieved = conversation_logger.get_conversation("nonexistent_id")
        
        assert retrieved is None
    
    def test_get_conversations_by_model(self, conversation_logger, sample_conversation):
        """Test retrieving conversations by model."""
        # Store multiple conversations
        conversation_logger.log_conversation(sample_conversation)
        
        # Create second conversation with different model
        conversation2 = sample_conversation.copy()
        conversation2["conversation_metadata"]["conversation_id"] = "test_conv_002"
        conversation2["conversation_metadata"]["model_name"] = "different_model"
        conversation_logger.log_conversation(conversation2)
        
        # Retrieve by model
        test_model_conversations = conversation_logger.get_conversations_by_model("test_model")
        different_model_conversations = conversation_logger.get_conversations_by_model("different_model")
        
        assert len(test_model_conversations) == 1
        assert len(different_model_conversations) == 1
        assert test_model_conversations[0]["conversation_metadata"]["model_name"] == "test_model"
    
    def test_export_conversations_json(self, conversation_logger, sample_conversation, temp_test_dir):
        """Test exporting conversations to JSON."""
        # Store conversation
        conversation_logger.log_conversation(sample_conversation)
        
        # Export to JSON
        export_file = temp_test_dir / "exported_conversations.json"
        success = conversation_logger.export_conversations("json", str(export_file))
        
        assert success == True
        assert export_file.exists()
        
        # Verify exported content
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) >= 1
        assert exported_data[0]["conversation_metadata"]["conversation_id"] == "test_conv_001"
    
    def test_export_conversations_csv(self, conversation_logger, sample_conversation, temp_test_dir):
        """Test exporting conversations to CSV."""
        # Store conversation
        conversation_logger.log_conversation(sample_conversation)
        
        # Export to CSV
        export_file = temp_test_dir / "exported_conversations.csv"
        success = conversation_logger.export_conversations("csv", str(export_file))
        
        assert success == True
        assert export_file.exists()
        
        # Verify CSV content
        df = pd.read_csv(export_file)
        assert len(df) >= 1
        assert "conversation_id" in df.columns
        assert df.iloc[0]["conversation_id"] == "test_conv_001"


@pytest.mark.unit
class TestDatabaseManager:
    """Test database management functionality."""
    
    @pytest.fixture
    def database_manager(self, temp_test_dir):
        """Create database manager instance."""
        db_path = temp_test_dir / "test_database.db"
        return DatabaseManager(str(db_path))
    
    def test_database_manager_initialization(self, database_manager):
        """Test database manager initialization."""
        assert database_manager is not None
        assert hasattr(database_manager, 'create_tables')
        assert hasattr(database_manager, 'store_conversation')
        assert hasattr(database_manager, 'get_conversation')
    
    def test_create_tables(self, database_manager):
        """Test database table creation."""
        success = database_manager.create_tables()
        
        assert success == True
        
        # Verify tables exist
        with sqlite3.connect(database_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ["conversations", "conversation_turns", "evaluation_scores"]
        for table in expected_tables:
            assert table in tables
    
    def test_store_conversation(self, database_manager, sample_conversation):
        """Test storing conversation in database."""
        database_manager.create_tables()
        
        success = database_manager.store_conversation(sample_conversation)
        
        assert success == True
        
        # Verify conversation was stored
        with sqlite3.connect(database_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE conversation_id = ?", 
                          ("test_conv_001",))
            result = cursor.fetchone()
        
        assert result is not None
        assert result[1] == "test_conv_001"  # conversation_id column
    
    def test_get_conversation_from_db(self, database_manager, sample_conversation):
        """Test retrieving conversation from database."""
        database_manager.create_tables()
        database_manager.store_conversation(sample_conversation)
        
        retrieved = database_manager.get_conversation("test_conv_001")
        
        assert retrieved is not None
        assert retrieved["conversation_id"] == "test_conv_001"
        assert retrieved["model_name"] == "test_model"
    
    def test_database_transaction_rollback(self, database_manager):
        """Test database transaction rollback on error."""
        database_manager.create_tables()
        
        # Create invalid conversation data that should cause rollback
        invalid_conversation = {
            "conversation_metadata": {
                "conversation_id": None,  # This should cause an error
                "model_name": "test_model"
            }
        }
        
        success = database_manager.store_conversation(invalid_conversation)
        
        # Should fail gracefully
        assert success == False
    
    def test_database_connection_handling(self, temp_test_dir):
        """Test database connection handling."""
        # Test with invalid database path
        invalid_db_manager = DatabaseManager("/invalid/path/database.db")
        
        # Should handle connection errors gracefully
        success = invalid_db_manager.create_tables()
        assert success == False
    
    def test_concurrent_database_access(self, database_manager, sample_conversation):
        """Test concurrent database access."""
        database_manager.create_tables()
        
        import threading
        import time
        
        results = []
        
        def store_conversation(conv_id):
            conversation = sample_conversation.copy()
            conversation["conversation_metadata"]["conversation_id"] = conv_id
            result = database_manager.store_conversation(conversation)
            results.append(result)
        
        # Create multiple threads accessing database
        threads = []
        for i in range(5):
            thread = threading.Thread(target=store_conversation, args=(f"conv_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        assert all(results)
        assert len(results) == 5


@pytest.mark.unit
class TestFileManager:
    """Test file management functionality."""
    
    @pytest.fixture
    def file_manager(self, temp_test_dir):
        """Create file manager instance."""
        return FileManager(base_directory=str(temp_test_dir))
    
    def test_file_manager_initialization(self, file_manager):
        """Test file manager initialization."""
        assert file_manager is not None
        assert hasattr(file_manager, 'save_conversation')
        assert hasattr(file_manager, 'load_conversation')
        assert hasattr(file_manager, 'list_conversations')
    
    def test_save_conversation_json(self, file_manager, sample_conversation):
        """Test saving conversation as JSON."""
        file_path = file_manager.save_conversation(
            sample_conversation, 
            file_format="json",
            subdirectory="conversations"
        )
        
        assert file_path is not None
        assert Path(file_path).exists()
        assert Path(file_path).suffix == ".json"
    
    def test_load_conversation_json(self, file_manager, sample_conversation):
        """Test loading conversation from JSON."""
        # Save conversation first
        file_path = file_manager.save_conversation(sample_conversation, "json")
        
        # Load conversation
        loaded_conversation = file_manager.load_conversation(file_path)
        
        assert loaded_conversation is not None
        assert loaded_conversation["conversation_metadata"]["conversation_id"] == "test_conv_001"
    
    def test_list_conversations(self, file_manager, sample_conversation):
        """Test listing conversations in directory."""
        # Save multiple conversations
        for i in range(3):
            conversation = sample_conversation.copy()
            conversation["conversation_metadata"]["conversation_id"] = f"test_conv_{i:03d}"
            file_manager.save_conversation(conversation, "json", "conversations")
        
        # List conversations
        conversation_files = file_manager.list_conversations("conversations", "*.json")
        
        assert len(conversation_files) >= 3
        assert all(Path(f).suffix == ".json" for f in conversation_files)
    
    def test_save_conversation_invalid_format(self, file_manager, sample_conversation):
        """Test saving conversation with invalid format."""
        with pytest.raises(ValueError):
            file_manager.save_conversation(sample_conversation, "invalid_format")
    
    def test_load_nonexistent_file(self, file_manager):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            file_manager.load_conversation("nonexistent_file.json")
    
    def test_file_naming_strategy(self, file_manager, sample_conversation):
        """Test file naming strategy."""
        file_path = file_manager.save_conversation(sample_conversation, "json")
        
        file_name = Path(file_path).name
        
        # Should include conversation ID and timestamp
        assert "test_conv_001" in file_name
        assert file_name.endswith(".json")
    
    def test_directory_creation(self, file_manager, sample_conversation, temp_test_dir):
        """Test automatic directory creation."""
        subdirectory = "new_subdir/nested_dir"
        
        file_path = file_manager.save_conversation(
            sample_conversation, 
            "json", 
            subdirectory
        )
        
        # Directory should be created automatically
        expected_dir = temp_test_dir / subdirectory
        assert expected_dir.exists()
        assert Path(file_path).parent == expected_dir


@pytest.mark.unit
class TestDataIntegrity:
    """Test data integrity and validation."""
    
    @pytest.fixture
    def data_validator(self):
        """Create data validator instance."""
        from storage.data_validator import DataValidator
        return DataValidator()
    
    def test_conversation_data_validation(self, data_validator, sample_conversation):
        """Test conversation data validation."""
        is_valid, errors = data_validator.validate_conversation(sample_conversation)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_invalid_conversation_data(self, data_validator):
        """Test validation of invalid conversation data."""
        invalid_conversation = {
            "conversation_metadata": {
                # Missing required fields
                "conversation_id": "test_001"
            },
            "conversation_turns": [
                {
                    "turn_number": "invalid",  # Should be int
                    "speaker": "unknown_speaker",  # Invalid speaker
                    "message": ""  # Empty message
                }
            ]
        }
        
        is_valid, errors = data_validator.validate_conversation(invalid_conversation)
        
        assert is_valid == False
        assert len(errors) > 0
    
    def test_data_sanitization(self, data_validator):
        """Test data sanitization for security."""
        malicious_conversation = {
            "conversation_metadata": {
                "conversation_id": "test_001",
                "model_name": "<script>alert('xss')</script>",
                "scenario_id": "'; DROP TABLE conversations; --"
            },
            "conversation_turns": [
                {
                    "turn_number": 1,
                    "speaker": "patient",
                    "message": "Normal message with <script>bad code</script>"
                }
            ]
        }
        
        sanitized = data_validator.sanitize_conversation(malicious_conversation)
        
        # Should remove or escape malicious content
        assert "<script>" not in str(sanitized)
        assert "DROP TABLE" not in str(sanitized)
    
    def test_data_consistency_check(self, data_validator, sample_conversation):
        """Test data consistency checking."""
        # Modify conversation to create inconsistency
        inconsistent_conversation = sample_conversation.copy()
        inconsistent_conversation["conversation_metadata"]["total_turns"] = 5
        # But conversation_turns has only 2 turns
        
        is_consistent, issues = data_validator.check_consistency(inconsistent_conversation)
        
        assert is_consistent == False
        assert len(issues) > 0
        assert any("turn count" in issue.lower() for issue in issues)


@pytest.mark.unit
class TestDataRetrieval:
    """Test data retrieval and querying functionality."""
    
    @pytest.fixture
    def data_retriever(self, temp_test_dir):
        """Create data retriever instance."""
        from storage.data_retriever import DataRetriever
        return DataRetriever(database_path=temp_test_dir / "test.db")
    
    def test_query_conversations_by_criteria(self, data_retriever):
        """Test querying conversations by various criteria."""
        # This would test the querying functionality
        criteria = {
            "model_name": "test_model",
            "scenario_category": "Anxiety Disorders",
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31"
            }
        }
        
        results = data_retriever.query_conversations(criteria)
        
        # Should return results matching criteria
        assert isinstance(results, list)
    
    def test_aggregate_statistics(self, data_retriever):
        """Test aggregate statistics calculation."""
        stats = data_retriever.get_aggregate_statistics()
        
        expected_stats = [
            "total_conversations",
            "conversations_by_model",
            "avg_scores_by_model",
            "conversation_count_by_date"
        ]
        
        for stat in expected_stats:
            assert stat in stats
    
    def test_export_filtered_data(self, data_retriever, temp_test_dir):
        """Test exporting filtered data."""
        filters = {
            "model_name": "test_model",
            "min_score": 7.0
        }
        
        export_path = temp_test_dir / "filtered_export.json"
        
        success = data_retriever.export_filtered_data(filters, str(export_path))
        
        assert success == True
        # Additional verification would depend on implementation