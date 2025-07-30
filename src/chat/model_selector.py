"""
Dynamic Model Selection System

Intelligently selects the best LLM for each conversation based on 
real-time parallel evaluation of response quality.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..evaluation.multi_model_evaluator import MultiModelEvaluator
from ..evaluation.evaluation_metrics import TherapeuticEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionResult:
    """Result of model selection process"""
    selected_model: str
    selection_score: float
    selection_time_ms: float
    all_scores: Dict[str, float]
    response_preview: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'selected_model': self.selected_model,
            'selection_score': self.selection_score,
            'selection_time_ms': self.selection_time_ms,
            'all_scores': self.all_scores,
            'response_preview': self.response_preview[:100] + "..." if len(self.response_preview) > 100 else self.response_preview,
            'timestamp': self.timestamp.isoformat()
        }


class ModelSelector:
    """
    Dynamically selects the best model for each conversation based on
    parallel evaluation of initial responses.
    """
    
    def __init__(self, 
                 available_models: List[str] = None,
                 fallback_model: str = 'openai',
                 timeout_seconds: float = 30.0):
        """
        Initialize model selector
        
        Args:
            available_models: List of models to consider
            fallback_model: Model to use if selection fails
            timeout_seconds: Maximum time for selection process
        """
        self.available_models = available_models or ['openai', 'deepseek', 'claude', 'gemma']
        self.fallback_model = fallback_model
        self.timeout_seconds = timeout_seconds
        
        # Initialize evaluator and model clients
        self.evaluator = TherapeuticEvaluator()
        self.model_clients = {}
        self._initialize_clients()
        
        logger.info(f"ModelSelector initialized with models: {self.available_models}")
    
    def _initialize_clients(self):
        """Initialize all available model clients"""
        for model_name in self.available_models:
            try:
                client = self._create_client(model_name)
                if client:
                    self.model_clients[model_name] = client
                    logger.info(f"Initialized {model_name} client")
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
    
    def _create_client(self, model_name: str):
        """Create a client for the specified model"""
        try:
            if model_name == 'openai':
                from ..models.openai_client import OpenAIClient
                return OpenAIClient()
            elif model_name == 'claude':
                from ..models.claude_client import ClaudeClient
                return ClaudeClient()
            elif model_name == 'deepseek':
                from ..models.deepseek_client import DeepSeekClient
                return DeepSeekClient()
            elif model_name == 'gemma':
                from ..models.gemma_client import GemmaClient
                return GemmaClient()
            else:
                logger.error(f"Unknown model: {model_name}")
                return None
        except ImportError as e:
            logger.warning(f"Could not import {model_name} client: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create {model_name} client: {e}")
            return None
    
    async def select_best_model(self, 
                              user_prompt: str,
                              system_prompt: Optional[str] = None,
                              conversation_history: Optional[List[Dict[str, str]]] = None) -> ModelSelectionResult:
        """
        Select the best model by running parallel evaluations
        
        Args:
            user_prompt: The user's input message
            system_prompt: Optional system prompt
            conversation_history: Previous conversation context
            
        Returns:
            ModelSelectionResult with the selected model and scores
        """
        start_time = time.time()
        
        logger.info(f"Starting model selection for prompt: {user_prompt[:50]}...")
        
        # Run parallel evaluations
        try:
            model_results = await asyncio.wait_for(
                self._evaluate_models_parallel(user_prompt, system_prompt, conversation_history),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Model selection timed out after {self.timeout_seconds}s")
            return self._fallback_selection(user_prompt, start_time)
        
        # Select best model based on scores
        if not model_results:
            logger.warning("No model results available, using fallback")
            return self._fallback_selection(user_prompt, start_time)
        
        best_model = max(model_results.items(), key=lambda x: x[1]['score'])
        selected_model_name = best_model[0]
        selected_result = best_model[1]
        
        selection_time_ms = (time.time() - start_time) * 1000
        
        # Extract all scores for comparison
        all_scores = {model: result['score'] for model, result in model_results.items()}
        
        result = ModelSelectionResult(
            selected_model=selected_model_name,
            selection_score=selected_result['score'],
            selection_time_ms=selection_time_ms,
            all_scores=all_scores,
            response_preview=selected_result['response'],
            timestamp=datetime.now()
        )
        
        logger.info(f"Selected {selected_model_name} with score {selected_result['score']:.2f} in {selection_time_ms:.0f}ms")
        
        return result
    
    async def _evaluate_models_parallel(self, 
                                      user_prompt: str,
                                      system_prompt: Optional[str] = None,
                                      conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Dict[str, Any]]:
        """Run parallel evaluations across all available models"""
        
        # Create tasks for all available models
        tasks = {}
        for model_name, client in self.model_clients.items():
            task = asyncio.create_task(
                self._evaluate_single_model(
                    model_name, client, user_prompt, system_prompt, conversation_history
                )
            )
            tasks[model_name] = task
        
        # Wait for all tasks to complete
        results = {}
        for model_name, task in tasks.items():
            try:
                result = await task
                if result:
                    results[model_name] = result
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return results
    
    async def _evaluate_single_model(self,
                                   model_name: str,
                                   client: Any,
                                   user_prompt: str,
                                   system_prompt: Optional[str] = None,
                                   conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
        """Evaluate a single model's response"""
        try:
            # Generate response
            response_start = time.time()
            
            if hasattr(client, 'generate_response'):
                # Use the standardized async interface
                response_obj = await client.generate_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history
                )
                response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                response_time_ms = response_obj.response_time_ms if hasattr(response_obj, 'response_time_ms') else (time.time() - response_start) * 1000
            else:
                # Fallback for older clients
                response_obj = client.chat(user_prompt)
                response_text = response_obj.get('content', str(response_obj)) if isinstance(response_obj, dict) else str(response_obj)
                response_time_ms = (time.time() - response_start) * 1000
            
            # Evaluate response quality
            evaluation = self.evaluator.evaluate_response(
                prompt=user_prompt,
                response=response_text,
                response_time_ms=response_time_ms,
                input_tokens=len(user_prompt.split()) * 1.3,  # Rough estimate
                output_tokens=len(response_text.split()) * 1.3
            )
            
            # Extract composite score
            if hasattr(evaluation, 'composite_score'):
                score = evaluation.composite_score
            elif isinstance(evaluation, dict):
                score = evaluation.get('composite_score', evaluation.get('composite', 0.0))
            else:
                score = 0.0
            
            return {
                'response': response_text,
                'score': score,
                'response_time_ms': response_time_ms,
                'evaluation': evaluation
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return None
    
    def _fallback_selection(self, user_prompt: str, start_time: float) -> ModelSelectionResult:
        """Create fallback selection result"""
        selection_time_ms = (time.time() - start_time) * 1000
        
        return ModelSelectionResult(
            selected_model=self.fallback_model,
            selection_score=0.0,
            selection_time_ms=selection_time_ms,
            all_scores={self.fallback_model: 0.0},
            response_preview="Fallback selection used",
            timestamp=datetime.now()
        )
    
    async def get_model_health_status(self) -> Dict[str, bool]:
        """Check health status of all models"""
        health_status = {}
        
        for model_name, client in self.model_clients.items():
            try:
                if hasattr(client, 'health_check'):
                    is_healthy = await client.health_check()
                else:
                    # Simple test generation
                    test_response = await self._evaluate_single_model(
                        model_name, client, "Hello", None, None
                    )
                    is_healthy = test_response is not None
                
                health_status[model_name] = is_healthy
                
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {e}")
                health_status[model_name] = False
        
        return health_status
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        return list(self.model_clients.keys())
    
    def set_fallback_model(self, model_name: str):
        """Set the fallback model"""
        if model_name in self.available_models:
            self.fallback_model = model_name
            logger.info(f"Fallback model set to {model_name}")
        else:
            logger.warning(f"Model {model_name} not available, fallback unchanged")