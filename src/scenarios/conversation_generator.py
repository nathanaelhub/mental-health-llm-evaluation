"""
Conversation generator for therapeutic evaluation scenarios.

This module generates complete conversations between users and models
for comprehensive evaluation of therapeutic interactions.
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import random
import logging

from ..models.base_model import BaseModel, ModelResponse
from .scenario_loader import TherapeuticScenario
from ..utils.paths import get_conversations_dir

logger = logging.getLogger(__name__)


@dataclass
class ConversationData:
    """Represents a complete therapeutic conversation for evaluation."""
    
    conversation_id: str
    scenario_id: str
    model_name: str
    timestamp: datetime
    
    messages: List[Dict[str, str]]     # Complete conversation history
    scenario_context: TherapeuticScenario
    model_responses: List[ModelResponse]
    
    conversation_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation data to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "scenario_id": self.scenario_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "messages": self.messages,
            "scenario_context": self.scenario_context.to_dict(),
            "model_responses": [response.to_dict() for response in self.model_responses],
            "conversation_metrics": self.conversation_metrics,
            "metadata": self.metadata
        }
    
    def get_conversation_length(self) -> int:
        """Get total number of message exchanges."""
        return len([msg for msg in self.messages if msg.get("role") == "user"])
    
    def get_response_times(self) -> List[float]:
        """Get response times for all model responses."""
        return [response.response_time_ms for response in self.model_responses]
    
    def get_average_response_time(self) -> float:
        """Get average response time across conversation."""
        response_times = self.get_response_times()
        return sum(response_times) / len(response_times) if response_times else 0.0


class ConversationGenerator:
    """Generates therapeutic conversations for model evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize conversation generator.
        
        Args:
            config: Configuration for conversation generation
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.max_turns = self.config.get("max_turns", 10)
        self.min_turns = self.config.get("min_turns", 3)
        self.turn_timeout = self.config.get("turn_timeout", 30)
        self.conversation_timeout = self.config.get("conversation_timeout", 300)
        
        # User simulation parameters
        self.user_response_probability = self.config.get("user_response_probability", 0.8)
        self.user_elaboration_probability = self.config.get("user_elaboration_probability", 0.6)
        self.conversation_end_probability = self.config.get("conversation_end_probability", 0.1)
        
    async def generate_conversation(
        self,
        model: BaseModel,
        scenario: TherapeuticScenario,
        **kwargs
    ) -> ConversationData:
        """
        Generate a complete therapeutic conversation.
        
        Args:
            model: Model to evaluate
            scenario: Therapeutic scenario for conversation
            **kwargs: Additional generation parameters
            
        Returns:
            Complete conversation data
        """
        conversation_id = f"{model.model_name}_{scenario.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Generating conversation {conversation_id}")
        
        try:
            # Initialize conversation
            messages = scenario.conversation_history.copy()
            model_responses = []
            
            # Add initial user message
            messages.append({
                "role": "user",
                "content": scenario.user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate conversation turns
            turn_count = 0
            start_time = datetime.now()
            
            while turn_count < self.max_turns:
                # Check timeout
                if (datetime.now() - start_time).seconds > self.conversation_timeout:
                    self.logger.warning(f"Conversation {conversation_id} timed out")
                    break
                
                # Generate model response
                conversation_history = messages.copy()
                current_prompt = messages[-1]["content"]
                
                try:
                    response = await model.generate_response(
                        current_prompt,
                        conversation_history[:-1],  # Exclude current message
                        **kwargs
                    )
                    
                    model_responses.append(response)
                    
                    if response.is_successful:
                        messages.append({
                            "role": "assistant",
                            "content": response.content,
                            "timestamp": datetime.now().isoformat(),
                            "response_time_ms": response.response_time_ms
                        })
                    else:
                        self.logger.warning(f"Model response failed: {response.error}")
                        break
                
                except Exception as e:
                    self.logger.error(f"Error generating model response: {e}")
                    break
                
                turn_count += 1
                
                # Check if conversation should continue
                if turn_count >= self.min_turns:
                    # Decide if conversation should end
                    if random.random() < self.conversation_end_probability:
                        break
                    
                    # Check for natural ending points
                    if self._is_natural_ending(response.content):
                        break
                
                # Generate next user message (simulate user response)
                next_user_message = await self._generate_user_response(
                    messages, scenario, turn_count
                )
                
                if next_user_message:
                    messages.append({
                        "role": "user", 
                        "content": next_user_message,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    # User chooses to end conversation
                    break
            
            # Calculate conversation metrics
            conversation_metrics = self._calculate_conversation_metrics(
                messages, model_responses, scenario
            )
            
            conversation_data = ConversationData(
                conversation_id=conversation_id,
                scenario_id=scenario.id,
                model_name=model.model_name,
                timestamp=start_time,
                messages=messages,
                scenario_context=scenario,
                model_responses=model_responses,
                conversation_metrics=conversation_metrics,
                metadata={
                    "generation_config": self.config,
                    "total_turns": turn_count,
                    "generation_duration_seconds": (datetime.now() - start_time).seconds
                }
            )
            
            self.logger.info(f"Generated conversation {conversation_id} with {turn_count} turns")
            return conversation_data
        
        except Exception as e:
            self.logger.error(f"Error generating conversation {conversation_id}: {e}")
            raise
    
    async def generate_batch_conversations(
        self,
        models: List[BaseModel],
        scenarios: List[TherapeuticScenario],
        conversations_per_scenario: int = 1,
        **kwargs
    ) -> List[ConversationData]:
        """
        Generate multiple conversations across models and scenarios.
        
        Args:
            models: List of models to evaluate
            scenarios: List of scenarios to use
            conversations_per_scenario: Number of conversations per scenario
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated conversation data
        """
        self.logger.info(
            f"Generating batch conversations: {len(models)} models, "
            f"{len(scenarios)} scenarios, {conversations_per_scenario} conversations each"
        )
        
        tasks = []
        
        for model in models:
            for scenario in scenarios:
                for i in range(conversations_per_scenario):
                    task = self.generate_conversation(model, scenario, **kwargs)
                    tasks.append(task)
        
        # Execute all conversations concurrently
        conversations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_conversations = []
        failed_count = 0
        
        for conv in conversations:
            if isinstance(conv, Exception):
                self.logger.error(f"Conversation generation failed: {conv}")
                failed_count += 1
            else:
                successful_conversations.append(conv)
        
        self.logger.info(
            f"Batch generation complete: {len(successful_conversations)} successful, "
            f"{failed_count} failed"
        )
        
        return successful_conversations
    
    async def _generate_user_response(
        self,
        messages: List[Dict[str, str]],
        scenario: TherapeuticScenario,
        turn_count: int
    ) -> Optional[str]:
        """
        Generate a simulated user response to continue the conversation.
        
        Args:
            messages: Current conversation history
            scenario: Therapeutic scenario context
            turn_count: Current turn number
            
        Returns:
            Generated user message or None to end conversation
        """
        # Probability of user continuing conversation decreases over time
        continue_probability = self.user_response_probability * (0.9 ** turn_count)
        
        if random.random() > continue_probability:
            return None
        
        # Get last assistant response
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            return None
        
        last_response = assistant_messages[-1]["content"]
        
        # Generate contextually appropriate user response
        user_responses = self._get_contextual_user_responses(scenario, last_response, turn_count)
        
        if user_responses:
            return random.choice(user_responses)
        
        return None
    
    def _get_contextual_user_responses(
        self,
        scenario: TherapeuticScenario,
        last_response: str,
        turn_count: int
    ) -> List[str]:
        """Generate contextually appropriate user responses."""
        
        responses = []
        
        # Category-specific response patterns
        if scenario.category == "anxiety":
            anxiety_responses = [
                "I understand what you're saying, but I still feel really worried about it.",
                "That makes sense. I think my anxiety is just getting the better of me.",
                "Thank you for explaining that. It's hard to think clearly when I'm this anxious.",
                "I'll try that, but what if it doesn't work? What if I panic anyway?",
                "I appreciate your support. Sometimes I just need someone to understand."
            ]
            responses.extend(anxiety_responses)
        
        elif scenario.category == "depression":
            depression_responses = [
                "I know you're trying to help, but it's hard to believe things will get better.",
                "I've tried things like that before, but I just don't have the energy anymore.",
                "Thank you for listening. It means a lot to have someone who understands.",
                "I want to feel better, but everything just seems so overwhelming right now.",
                "Some days are harder than others. Today has been particularly difficult."
            ]
            responses.extend(depression_responses)
        
        elif scenario.category == "trauma":
            trauma_responses = [
                "It's hard to talk about this, but I know I need to process what happened.",
                "Sometimes I feel like I'm going crazy. The memories feel so real.",
                "Thank you for being patient with me. This is really difficult to discuss.",
                "I want to move forward, but I don't know how to stop reliving the past.",
                "I feel safer talking about it here, but it's still scary."
            ]
            responses.extend(trauma_responses)
        
        # Generic therapeutic responses
        generic_responses = [
            "Can you help me understand that better?",
            "That's helpful. Can you tell me more about that approach?",
            "I'm not sure I understand. Can you explain it differently?",
            "That gives me something to think about. Thank you.",
            "I feel a bit better talking about this with you."
        ]
        responses.extend(generic_responses)
        
        # Later turn responses (more reflective)
        if turn_count > 5:
            late_responses = [
                "This conversation has been really helpful. I feel like I'm starting to understand.",
                "I think I'm beginning to see things differently.",
                "Thank you for taking the time to talk with me about this.",
                "I feel like I have some things to work on now."
            ]
            responses.extend(late_responses)
        
        return responses
    
    def _is_natural_ending(self, response_content: str) -> bool:
        """Check if the response suggests a natural conversation ending."""
        
        ending_indicators = [
            "take care",
            "best of luck",
            "feel free to reach out",
            "hope this helps",
            "wishing you well",
            "thank you for sharing",
            "good luck with",
            "hope you feel better"
        ]
        
        response_lower = response_content.lower()
        return any(indicator in response_lower for indicator in ending_indicators)
    
    def _calculate_conversation_metrics(
        self,
        messages: List[Dict[str, str]],
        model_responses: List[ModelResponse],
        scenario: TherapeuticScenario
    ) -> Dict[str, Any]:
        """Calculate metrics for the completed conversation."""
        
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        # Basic conversation metrics
        total_turns = len(assistant_messages)
        conversation_length = sum(len(msg["content"]) for msg in messages)
        
        # Response time metrics
        response_times = [response.response_time_ms for response in model_responses]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Content analysis
        total_user_words = sum(len(msg["content"].split()) for msg in user_messages)
        total_assistant_words = sum(len(msg["content"].split()) for msg in assistant_messages)
        
        avg_user_message_length = total_user_words / len(user_messages) if user_messages else 0
        avg_assistant_message_length = total_assistant_words / len(assistant_messages) if assistant_messages else 0
        
        # Engagement metrics
        user_engagement_score = min(100, total_turns * 10)  # Higher turns = higher engagement
        
        return {
            "total_turns": total_turns,
            "conversation_length_chars": conversation_length,
            "average_response_time_ms": avg_response_time,
            "total_user_words": total_user_words,
            "total_assistant_words": total_assistant_words,
            "avg_user_message_length": avg_user_message_length,
            "avg_assistant_message_length": avg_assistant_message_length,
            "user_engagement_score": user_engagement_score,
            "successful_responses": len([r for r in model_responses if r.is_successful]),
            "failed_responses": len([r for r in model_responses if not r.is_successful])
        }
    
    async def save_conversation(
        self,
        conversation: ConversationData,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Save conversation data to file.
        
        Args:
            conversation: Conversation data to save
            output_dir: Directory to save conversation files (defaults to configured conversations directory)
        """
        import os
        from pathlib import Path
        
        # Use configured conversations directory if no output_dir specified
        if output_dir is None:
            output_dir = str(get_conversations_dir())
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{conversation.conversation_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved conversation to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error saving conversation {conversation.conversation_id}: {e}")
            raise
    
    async def load_conversation(self, filepath: str) -> ConversationData:
        """
        Load conversation data from file.
        
        Args:
            filepath: Path to conversation file
            
        Returns:
            Loaded conversation data
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct objects
            scenario_context = TherapeuticScenario.from_dict(data["scenario_context"])
            
            model_responses = []
            for response_data in data["model_responses"]:
                # Convert timestamp string back to datetime
                if isinstance(response_data["timestamp"], str):
                    response_data["timestamp"] = datetime.fromisoformat(response_data["timestamp"])
                
                model_responses.append(ModelResponse(**response_data))
            
            conversation = ConversationData(
                conversation_id=data["conversation_id"],
                scenario_id=data["scenario_id"],
                model_name=data["model_name"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                messages=data["messages"],
                scenario_context=scenario_context,
                model_responses=model_responses,
                conversation_metrics=data["conversation_metrics"],
                metadata=data["metadata"]
            )
            
            return conversation
        
        except Exception as e:
            self.logger.error(f"Error loading conversation from {filepath}: {e}")
            raise