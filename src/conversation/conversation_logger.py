"""
Conversation Logger for Mental Health LLM Evaluation

This module provides structured logging and data storage for conversations,
including JSON serialization, database integration, and export capabilities.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import asdict
import csv
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager

from .model_interface import ConversationContext, ConversationTurn
from .metrics_collector import ConversationAnalytics, MetricsCollector
from ..scenarios.scenario import Scenario

logger = logging.getLogger(__name__)


class ConversationLogger:
    """
    Comprehensive conversation logging system for mental health LLM evaluation.
    
    Provides structured storage, JSON serialization, database integration,
    and various export formats for conversation data and analytics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize conversation logger.
        
        Args:
            config: Configuration dictionary for logging system
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.output_dir = Path(self.config.get("output_dir", "./data/conversations"))
        self.enable_json_logging = self.config.get("enable_json_logging", True)
        self.enable_database_logging = self.config.get("enable_database_logging", True)
        self.enable_csv_export = self.config.get("enable_csv_export", True)
        self.compress_large_files = self.config.get("compress_large_files", True)
        self.max_file_size_mb = self.config.get("max_file_size_mb", 50)
        
        # Storage setup
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.output_dir / "json"
        self.csv_dir = self.output_dir / "csv"
        self.db_path = self.output_dir / "conversations.db"
        
        if self.enable_json_logging:
            self.json_dir.mkdir(exist_ok=True)
        
        if self.enable_csv_export:
            self.csv_dir.mkdir(exist_ok=True)
        
        # Database initialization
        if self.enable_database_logging:
            asyncio.create_task(self._initialize_database())
        
        # Logging queue for async operations
        self._logging_queue = asyncio.Queue()
        self._logging_task = None
        self._start_logging_task()
        
        self.logger.info(f"ConversationLogger initialized with output dir: {self.output_dir}")
    
    async def _initialize_database(self):
        """Initialize SQLite database for conversation storage."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        scenario_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        total_turns INTEGER,
                        total_duration_ms REAL,
                        termination_reason TEXT,
                        safety_flags_count INTEGER,
                        avg_response_time_ms REAL,
                        avg_quality_score REAL,
                        conversation_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        turn_number INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        response_time_ms REAL,
                        token_count INTEGER,
                        quality_score REAL,
                        safety_flags TEXT,
                        metadata TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_analytics (
                        conversation_id TEXT PRIMARY KEY,
                        total_words INTEGER,
                        unique_words INTEGER,
                        avg_empathy_score REAL,
                        avg_coherence_score REAL,
                        therapeutic_elements TEXT,
                        safety_events TEXT,
                        user_engagement TEXT,
                        conversation_flow_rating REAL,
                        natural_ending BOOLEAN,
                        analytics_data TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_model_scenario 
                    ON conversations (model_name, scenario_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_turns_conversation 
                    ON conversation_turns (conversation_id, turn_number)
                """)
                
                await db.commit()
            
            self.logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _start_logging_task(self):
        """Start background task for processing logging queue."""
        async def process_logging_queue():
            while True:
                try:
                    log_item = await self._logging_queue.get()
                    await self._process_log_item(log_item)
                    self._logging_queue.task_done()
                except Exception as e:
                    self.logger.error(f"Error processing log item: {e}")
        
        self._logging_task = asyncio.create_task(process_logging_queue())
    
    async def _process_log_item(self, log_item: Dict[str, Any]):
        """Process individual log item from queue."""
        log_type = log_item.get("type")
        
        if log_type == "conversation":
            await self._log_conversation_to_storage(
                log_item["context"],
                log_item.get("scenario"),
                log_item.get("analytics")
            )
        elif log_type == "turn":
            await self._log_turn_to_storage(
                log_item["conversation_id"],
                log_item["turn"]
            )
        elif log_type == "batch_summary":
            await self._log_batch_summary(
                log_item["summary"],
                log_item["conversations"]
            )
    
    async def log_conversation(
        self,
        context: ConversationContext,
        scenario: Optional[Scenario] = None,
        analytics: Optional[ConversationAnalytics] = None
    ):
        """
        Log a completed conversation asynchronously.
        
        Args:
            context: Conversation context with full history
            scenario: Original scenario used (optional)
            analytics: Conversation analytics data (optional)
        """
        log_item = {
            "type": "conversation",
            "context": context,
            "scenario": scenario,
            "analytics": analytics
        }
        
        await self._logging_queue.put(log_item)
    
    async def log_turn(self, conversation_id: str, turn: ConversationTurn):
        """
        Log a single conversation turn asynchronously.
        
        Args:
            conversation_id: ID of the conversation
            turn: Conversation turn to log
        """
        log_item = {
            "type": "turn",
            "conversation_id": conversation_id,
            "turn": turn
        }
        
        await self._logging_queue.put(log_item)
    
    async def _log_conversation_to_storage(
        self,
        context: ConversationContext,
        scenario: Optional[Scenario] = None,
        analytics: Optional[ConversationAnalytics] = None
    ):
        """Log conversation to all enabled storage backends."""
        
        # JSON logging
        if self.enable_json_logging:
            await self._log_conversation_json(context, scenario, analytics)
        
        # Database logging
        if self.enable_database_logging:
            await self._log_conversation_database(context, analytics)
        
        # CSV export
        if self.enable_csv_export:
            await self._append_conversation_csv(context, analytics)
        
        self.logger.debug(f"Logged conversation {context.conversation_id}")
    
    async def _log_conversation_json(
        self,
        context: ConversationContext,
        scenario: Optional[Scenario] = None,
        analytics: Optional[ConversationAnalytics] = None
    ):
        """Log conversation to JSON file."""
        try:
            # Prepare conversation data
            conversation_data = {
                "conversation_metadata": context.to_dict(),
                "scenario_data": scenario.to_dict() if scenario else None,
                "analytics_data": analytics.to_dict() if analytics else None,
                "export_timestamp": datetime.now().isoformat(),
                "logger_version": "1.0"
            }
            
            # Determine output file
            filename = f"{context.conversation_id}.json"
            filepath = self.json_dir / filename
            
            # Check file size and compress if necessary
            json_str = json.dumps(conversation_data, indent=2, ensure_ascii=False)
            
            if len(json_str.encode('utf-8')) > self.max_file_size_mb * 1024 * 1024:
                if self.compress_large_files:
                    import gzip
                    filepath = filepath.with_suffix('.json.gz')
                    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                        f.write(json_str)
                else:
                    self.logger.warning(f"Large conversation file not compressed: {filepath}")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(json_str)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            self.logger.debug(f"Conversation JSON saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to log conversation JSON: {e}")
    
    async def _log_conversation_database(
        self,
        context: ConversationContext,
        analytics: Optional[ConversationAnalytics] = None
    ):
        """Log conversation to SQLite database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Insert conversation record
                await db.execute("""
                    INSERT OR REPLACE INTO conversations (
                        conversation_id, scenario_id, model_name, start_time, end_time,
                        total_turns, total_duration_ms, termination_reason,
                        safety_flags_count, avg_response_time_ms, avg_quality_score,
                        conversation_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.conversation_id,
                    context.scenario_id,
                    context.model_name,
                    context.start_time.isoformat(),
                    context.end_time.isoformat() if context.end_time else None,
                    context.metrics.total_turns,
                    context.metrics.total_conversation_time_ms,
                    context.termination_reason,
                    len(context.safety_flags_total),
                    context.metrics.avg_response_time_ms,
                    context.metrics.avg_quality_score,
                    json.dumps(context.to_dict())
                ))
                
                # Insert conversation turns
                for turn in context.turns:
                    await db.execute("""
                        INSERT OR REPLACE INTO conversation_turns (
                            conversation_id, turn_number, role, content, timestamp,
                            response_time_ms, token_count, quality_score,
                            safety_flags, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        context.conversation_id,
                        turn.turn_number,
                        turn.role,
                        turn.content,
                        turn.timestamp.isoformat(),
                        turn.response_time_ms,
                        turn.token_count,
                        turn.quality_score,
                        json.dumps(turn.safety_flags) if turn.safety_flags else None,
                        json.dumps(turn.metadata) if turn.metadata else None
                    ))
                
                # Insert analytics if available
                if analytics:
                    avg_empathy = (
                        sum(analytics.empathy_scores) / len(analytics.empathy_scores)
                        if analytics.empathy_scores else None
                    )
                    avg_coherence = (
                        sum(analytics.coherence_scores) / len(analytics.coherence_scores)
                        if analytics.coherence_scores else None
                    )
                    
                    await db.execute("""
                        INSERT OR REPLACE INTO conversation_analytics (
                            conversation_id, total_words, unique_words,
                            avg_empathy_score, avg_coherence_score,
                            therapeutic_elements, safety_events, user_engagement,
                            conversation_flow_rating, natural_ending, analytics_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        context.conversation_id,
                        analytics.total_words,
                        analytics.unique_words,
                        avg_empathy,
                        avg_coherence,
                        json.dumps(analytics.therapeutic_element_scores),
                        json.dumps(analytics.safety_flag_timeline),
                        json.dumps(analytics.user_engagement_indicators),
                        analytics.conversation_flow_rating,
                        analytics.natural_ending_achieved,
                        json.dumps(analytics.to_dict())
                    ))
                
                await db.commit()
            
            self.logger.debug(f"Conversation database record saved: {context.conversation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log conversation to database: {e}")
    
    async def _append_conversation_csv(
        self,
        context: ConversationContext,
        analytics: Optional[ConversationAnalytics] = None
    ):
        """Append conversation summary to CSV file."""
        try:
            csv_file = self.csv_dir / "conversation_summary.csv"
            
            # Prepare row data
            row_data = {
                "conversation_id": context.conversation_id,
                "scenario_id": context.scenario_id,
                "model_name": context.model_name,
                "start_time": context.start_time.isoformat(),
                "end_time": context.end_time.isoformat() if context.end_time else "",
                "total_turns": context.metrics.total_turns,
                "assistant_turns": context.metrics.assistant_turns,
                "total_duration_ms": context.metrics.total_conversation_time_ms,
                "avg_response_time_ms": context.metrics.avg_response_time_ms,
                "total_tokens": context.metrics.total_tokens,
                "avg_quality_score": context.metrics.avg_quality_score or "",
                "safety_flags_count": len(context.safety_flags_total),
                "termination_reason": context.termination_reason,
                "api_errors": context.metrics.api_errors,
                "timeout_errors": context.metrics.timeout_errors
            }
            
            # Add analytics data if available
            if analytics:
                row_data.update({
                    "total_words": analytics.total_words,
                    "unique_words": analytics.unique_words,
                    "conversation_flow_rating": analytics.conversation_flow_rating,
                    "natural_ending": analytics.natural_ending_achieved,
                    "crisis_intervention": analytics.crisis_intervention_triggered,
                    "risk_escalation": analytics.risk_escalation_detected,
                    "engagement_indicators_count": len(analytics.user_engagement_indicators)
                })
            
            # Write to CSV
            file_exists = csv_file.exists()
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(row_data)
            
            self.logger.debug(f"Conversation CSV entry appended: {context.conversation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to append conversation to CSV: {e}")
    
    async def log_batch_summary(
        self,
        summary: Dict[str, Any],
        conversations: List[ConversationContext]
    ):
        """
        Log batch processing summary.
        
        Args:
            summary: Batch summary data
            conversations: List of completed conversations
        """
        log_item = {
            "type": "batch_summary",
            "summary": summary,
            "conversations": conversations
        }
        
        await self._logging_queue.put(log_item)
    
    async def _log_batch_summary(
        self,
        summary: Dict[str, Any],
        conversations: List[ConversationContext]
    ):
        """Log batch summary to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_summary_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Prepare comprehensive batch data
            batch_data = {
                "batch_summary": summary,
                "export_timestamp": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "conversation_ids": [conv.conversation_id for conv in conversations],
                "model_performance": self._calculate_model_performance(conversations),
                "scenario_coverage": self._calculate_scenario_coverage(conversations),
                "quality_metrics": self._calculate_batch_quality_metrics(conversations),
                "safety_analysis": self._calculate_batch_safety_metrics(conversations)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Batch summary saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to log batch summary: {e}")
    
    def _calculate_model_performance(self, conversations: List[ConversationContext]) -> Dict[str, Any]:
        """Calculate model performance metrics across batch."""
        model_stats = {}
        
        for conv in conversations:
            model = conv.model_name
            if model not in model_stats:
                model_stats[model] = {
                    "conversation_count": 0,
                    "total_turns": 0,
                    "total_response_time": 0.0,
                    "total_tokens": 0,
                    "safety_flags": 0,
                    "quality_scores": [],
                    "successful_completions": 0
                }
            
            stats = model_stats[model]
            stats["conversation_count"] += 1
            stats["total_turns"] += conv.metrics.total_turns
            stats["total_response_time"] += conv.metrics.avg_response_time_ms or 0
            stats["total_tokens"] += conv.metrics.total_tokens
            stats["safety_flags"] += len(conv.safety_flags_total)
            
            if conv.metrics.avg_quality_score:
                stats["quality_scores"].append(conv.metrics.avg_quality_score)
            
            if conv.termination_reason in ["natural_ending", "max_turns_reached"]:
                stats["successful_completions"] += 1
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["conversation_count"] > 0:
                stats["avg_turns_per_conversation"] = stats["total_turns"] / stats["conversation_count"]
                stats["avg_response_time"] = stats["total_response_time"] / stats["conversation_count"]
                stats["avg_tokens_per_conversation"] = stats["total_tokens"] / stats["conversation_count"]
                stats["success_rate"] = stats["successful_completions"] / stats["conversation_count"]
                stats["avg_quality_score"] = (
                    sum(stats["quality_scores"]) / len(stats["quality_scores"])
                    if stats["quality_scores"] else None
                )
        
        return model_stats
    
    def _calculate_scenario_coverage(self, conversations: List[ConversationContext]) -> Dict[str, int]:
        """Calculate scenario coverage across batch."""
        scenario_counts = {}
        for conv in conversations:
            scenario_counts[conv.scenario_id] = scenario_counts.get(conv.scenario_id, 0) + 1
        return scenario_counts
    
    def _calculate_batch_quality_metrics(self, conversations: List[ConversationContext]) -> Dict[str, Any]:
        """Calculate quality metrics across batch."""
        all_quality_scores = []
        turn_counts = []
        response_times = []
        
        for conv in conversations:
            if conv.metrics.avg_quality_score:
                all_quality_scores.append(conv.metrics.avg_quality_score)
            turn_counts.append(conv.metrics.total_turns)
            if conv.metrics.avg_response_time_ms:
                response_times.append(conv.metrics.avg_response_time_ms)
        
        return {
            "avg_quality_score": sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else None,
            "avg_conversation_length": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "quality_score_distribution": {
                "min": min(all_quality_scores) if all_quality_scores else None,
                "max": max(all_quality_scores) if all_quality_scores else None,
                "median": sorted(all_quality_scores)[len(all_quality_scores)//2] if all_quality_scores else None
            }
        }
    
    def _calculate_batch_safety_metrics(self, conversations: List[ConversationContext]) -> Dict[str, Any]:
        """Calculate safety metrics across batch."""
        total_safety_flags = 0
        conversations_with_flags = 0
        flag_types = {}
        
        for conv in conversations:
            conv_flags = len(conv.safety_flags_total)
            total_safety_flags += conv_flags
            
            if conv_flags > 0:
                conversations_with_flags += 1
            
            for flag in conv.safety_flags_total:
                flag_types[flag] = flag_types.get(flag, 0) + 1
        
        return {
            "total_safety_flags": total_safety_flags,
            "conversations_with_safety_flags": conversations_with_flags,
            "safety_flag_rate": conversations_with_flags / len(conversations) if conversations else 0,
            "avg_flags_per_conversation": total_safety_flags / len(conversations) if conversations else 0,
            "flag_type_distribution": flag_types
        }
    
    async def export_conversations(
        self,
        output_format: str = "json",
        filter_criteria: Optional[Dict[str, Any]] = None,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Export conversations in specified format with optional filtering.
        
        Args:
            output_format: Export format ("json", "csv", "jsonl")
            filter_criteria: Optional filtering criteria
            output_file: Optional output file path
            
        Returns:
            Path to exported file
        """
        if not self.enable_database_logging:
            raise ValueError("Database logging must be enabled for export functionality")
        
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"conversations_export_{timestamp}.{output_format}"
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query based on filter criteria
                query = "SELECT * FROM conversations"
                params = []
                
                if filter_criteria:
                    conditions = []
                    if "model_name" in filter_criteria:
                        conditions.append("model_name = ?")
                        params.append(filter_criteria["model_name"])
                    
                    if "scenario_id" in filter_criteria:
                        conditions.append("scenario_id = ?")
                        params.append(filter_criteria["scenario_id"])
                    
                    if "start_date" in filter_criteria:
                        conditions.append("start_time >= ?")
                        params.append(filter_criteria["start_date"])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                # Execute query
                async with db.execute(query, params) as cursor:
                    conversations = await cursor.fetchall()
                    column_names = [description[0] for description in cursor.description]
                
                # Export based on format
                if output_format == "json":
                    await self._export_json(conversations, column_names, output_file)
                elif output_format == "csv":
                    await self._export_csv(conversations, column_names, output_file)
                elif output_format == "jsonl":
                    await self._export_jsonl(conversations, column_names, output_file)
                else:
                    raise ValueError(f"Unsupported export format: {output_format}")
            
            self.logger.info(f"Conversations exported to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export conversations: {e}")
            raise
    
    async def _export_json(self, conversations: List, column_names: List[str], output_file: Path):
        """Export conversations to JSON format."""
        data = []
        for row in conversations:
            data.append(dict(zip(column_names, row)))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _export_csv(self, conversations: List, column_names: List[str], output_file: Path):
        """Export conversations to CSV format."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            writer.writerows(conversations)
    
    async def _export_jsonl(self, conversations: List, column_names: List[str], output_file: Path):
        """Export conversations to JSON Lines format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in conversations:
                record = dict(zip(column_names, row))
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    async def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about logged conversations."""
        if not self.enable_database_logging:
            return {"error": "Database logging not enabled"}
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}
                
                # Basic counts
                async with db.execute("SELECT COUNT(*) FROM conversations") as cursor:
                    result = await cursor.fetchone()
                    stats["total_conversations"] = result[0]
                
                # Model distribution
                async with db.execute("""
                    SELECT model_name, COUNT(*) as count 
                    FROM conversations 
                    GROUP BY model_name
                """) as cursor:
                    model_dist = await cursor.fetchall()
                    stats["model_distribution"] = dict(model_dist)
                
                # Scenario distribution
                async with db.execute("""
                    SELECT scenario_id, COUNT(*) as count 
                    FROM conversations 
                    GROUP BY scenario_id
                """) as cursor:
                    scenario_dist = await cursor.fetchall()
                    stats["scenario_distribution"] = dict(scenario_dist)
                
                # Average metrics
                async with db.execute("""
                    SELECT 
                        AVG(total_turns) as avg_turns,
                        AVG(total_duration_ms) as avg_duration,
                        AVG(avg_response_time_ms) as avg_response_time,
                        AVG(safety_flags_count) as avg_safety_flags
                    FROM conversations
                """) as cursor:
                    result = await cursor.fetchone()
                    if result:
                        stats["averages"] = {
                            "turns": result[0],
                            "duration_ms": result[1],
                            "response_time_ms": result[2],
                            "safety_flags": result[3]
                        }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get conversation statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources and ensure all pending logs are processed."""
        if self._logging_task:
            # Wait for queue to be empty
            await self._logging_queue.join()
            
            # Cancel the logging task
            self._logging_task.cancel()
            try:
                await self._logging_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("ConversationLogger cleanup completed")