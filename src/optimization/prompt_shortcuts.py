"""
Prompt Classification Shortcuts for Common Patterns

Fast pattern recognition system that bypasses full evaluation for well-known
prompt patterns, providing instant responses for common scenarios.
"""

import re
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Pattern, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..chat.dynamic_model_selector import PromptType

logger = logging.getLogger(__name__)


class ShortcutConfidence(Enum):
    """Confidence levels for shortcut classifications"""
    VERY_HIGH = 0.95    # Emergency/crisis patterns
    HIGH = 0.90         # Clear therapeutic patterns
    MEDIUM = 0.80       # Common patterns
    LOW = 0.70          # Weak indicators


@dataclass
class PatternRule:
    """Pattern matching rule for prompt classification"""
    name: str
    pattern: Pattern
    prompt_type: PromptType
    confidence: ShortcutConfidence
    suggested_model: str
    priority: int = 1  # 1=highest priority, 10=lowest
    min_match_length: int = 3  # Minimum characters that must match
    context_required: bool = False  # Whether surrounding context matters
    
    def matches(self, text: str) -> Optional[Tuple[float, int]]:
        """
        Check if pattern matches text
        
        Returns:
            Tuple of (confidence_score, match_position) or None
        """
        match = self.pattern.search(text.lower())
        if match:
            matched_text = match.group()
            if len(matched_text) >= self.min_match_length:
                # Calculate confidence based on match quality
                confidence_score = self.confidence.value
                
                # Adjust confidence based on match context
                if self.context_required:
                    confidence_score *= 0.9  # Slightly lower confidence
                
                return confidence_score, match.start()
        return None


@dataclass
class ShortcutResult:
    """Result from prompt shortcut classification"""
    prompt_type: PromptType
    confidence: float
    suggested_model: str
    reasoning: str
    matched_patterns: List[str]
    classification_time_ms: float
    bypass_full_evaluation: bool = True


@dataclass
class ShortcutMetrics:
    """Metrics for shortcut classification performance"""
    total_classifications: int = 0
    shortcut_hits: int = 0
    avg_classification_time_ms: float = 0.0
    
    # Accuracy tracking
    false_positives: int = 0  # Shortcuts that were wrong
    false_negatives: int = 0  # Missed shortcuts that should have been caught
    
    # Pattern performance
    pattern_hit_counts: Dict[str, int] = field(default_factory=dict)
    pattern_accuracy: Dict[str, float] = field(default_factory=dict)
    
    def hit_rate(self) -> float:
        """Calculate shortcut hit rate"""
        if self.total_classifications == 0:
            return 0.0
        return (self.shortcut_hits / self.total_classifications) * 100
    
    def accuracy_rate(self) -> float:
        """Calculate overall accuracy rate"""
        total_evaluations = self.false_positives + self.false_negatives + self.shortcut_hits
        if total_evaluations == 0:
            return 0.0
        return (self.shortcut_hits / total_evaluations) * 100


class PromptShortcuts:
    """
    Fast pattern recognition system for common prompt types
    
    Features:
    - Regex-based pattern matching for instant classification
    - Hierarchical rule system with priority ordering
    - Learning from classification accuracy
    - Context-aware pattern recognition
    - Emergency pattern detection with highest priority
    """
    
    def __init__(self):
        self.metrics = ShortcutMetrics()
        self.pattern_rules: List[PatternRule] = []
        self.recent_classifications = deque(maxlen=1000)
        
        # Initialize built-in patterns
        self._initialize_patterns()
        
        # Learning system
        self.pattern_learning_enabled = True
        self._learning_data = defaultdict(list)
        
        logger.info("PromptShortcuts initialized with pattern recognition")
    
    def _initialize_patterns(self):
        """Initialize built-in pattern recognition rules"""
        
        # CRISIS PATTERNS (Highest Priority)
        crisis_patterns = [
            # Immediate danger
            (r"\b(?:kill\s+myself|suicide|end\s+it\s+all|want\s+to\s+die)\b", 
             "crisis_immediate_danger", "claude-3-opus"),
            
            # Self-harm
            (r"\b(?:hurt\s+myself|cut\s+myself|harm\s+myself|self[\s-]harm)\b", 
             "crisis_self_harm", "claude-3-opus"),
            
            # Emergency situations
            (r"\b(?:emergency|crisis|urgent|help\s+me\s+now|immediate\s+help)\b", 
             "crisis_emergency", "claude-3-opus"),
            
            # Hopelessness indicators
            (r"\b(?:hopeless|no\s+point|nothing\s+matters|give\s+up)\b", 
             "crisis_hopelessness", "claude-3-opus"),
        ]
        
        for pattern, name, model in crisis_patterns:
            self.pattern_rules.append(PatternRule(
                name=name,
                pattern=re.compile(pattern, re.IGNORECASE),
                prompt_type=PromptType.CRISIS,
                confidence=ShortcutConfidence.VERY_HIGH,
                suggested_model=model,
                priority=1,
                min_match_length=3
            ))
        
        # ANXIETY PATTERNS
        anxiety_patterns = [
            # Panic-related
            (r"\b(?:panic\s+attack|panic|overwhelming|can't\s+breathe)\b", 
             "anxiety_panic", "claude-3-sonnet"),
            
            # General anxiety
            (r"\b(?:anxious|anxiety|worried|stress|nervous|tense)\b", 
             "anxiety_general", "claude-3-sonnet"),
            
            # Social anxiety
            (r"\b(?:social\s+anxiety|afraid\s+of\s+people|fear\s+judgment)\b", 
             "anxiety_social", "claude-3-sonnet"),
            
            # Physical symptoms
            (r"\b(?:racing\s+heart|sweating|trembling|shaking)\b", 
             "anxiety_physical", "claude-3-sonnet"),
        ]
        
        for pattern, name, model in anxiety_patterns:
            self.pattern_rules.append(PatternRule(
                name=name,
                pattern=re.compile(pattern, re.IGNORECASE),
                prompt_type=PromptType.ANXIETY,
                confidence=ShortcutConfidence.HIGH,
                suggested_model=model,
                priority=2
            ))
        
        # DEPRESSION PATTERNS
        depression_patterns = [
            # Core symptoms
            (r"\b(?:depressed|depression|sad|sadness|empty|numb)\b", 
             "depression_core", "claude-3-sonnet"),
            
            # Sleep and energy
            (r"\b(?:can't\s+sleep|insomnia|tired|exhausted|no\s+energy)\b", 
             "depression_fatigue", "claude-3-sonnet"),
            
            # Self-worth
            (r"\b(?:worthless|failure|not\s+good\s+enough|hate\s+myself)\b", 
             "depression_self_worth", "claude-3-sonnet"),
            
            # Isolation
            (r"\b(?:alone|lonely|isolated|no\s+friends|nobody\s+cares)\b", 
             "depression_isolation", "claude-3-sonnet"),
        ]
        
        for pattern, name, model in depression_patterns:
            self.pattern_rules.append(PatternRule(
                name=name,
                pattern=re.compile(pattern, re.IGNORECASE),
                prompt_type=PromptType.DEPRESSION,
                confidence=ShortcutConfidence.HIGH,
                suggested_model=model,
                priority=2
            ))
        
        # INFORMATION SEEKING PATTERNS
        info_patterns = [
            # Question words
            (r"^(?:what|how|why|when|where|who)\s+(?:is|are|do|does|can|should)", 
             "info_question", "gpt-4-turbo"),
            
            # Explanation requests
            (r"\b(?:explain|tell\s+me\s+about|help\s+me\s+understand|learn\s+about)\b", 
             "info_explanation", "gpt-4-turbo"),
            
            # Research queries
            (r"\b(?:research|studies|evidence|facts|information)\b", 
             "info_research", "gpt-4-turbo"),
            
            # Definitions
            (r"\b(?:define|definition|meaning|what\s+does.*mean)\b", 
             "info_definition", "gpt-3.5-turbo"),
        ]
        
        for pattern, name, model in info_patterns:
            self.pattern_rules.append(PatternRule(
                name=name,
                pattern=re.compile(pattern, re.IGNORECASE),
                prompt_type=PromptType.INFORMATION_SEEKING,
                confidence=ShortcutConfidence.MEDIUM,
                suggested_model=model,
                priority=4
            ))
        
        # GENERAL WELLNESS PATTERNS
        wellness_patterns = [
            # Greetings
            (r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))", 
             "wellness_greeting", "gpt-3.5-turbo"),
            
            # General check-ins
            (r"\b(?:how\s+are\s+you|feeling\s+okay|doing\s+well|good\s+day)\b", 
             "wellness_checkin", "gpt-3.5-turbo"),
            
            # Wellness practices
            (r"\b(?:meditation|mindfulness|exercise|self[-\s]care|wellness)\b", 
             "wellness_practices", "claude-3-haiku"),
            
            # Positive affirmations
            (r"\b(?:grateful|thankful|blessed|positive|optimistic)\b", 
             "wellness_positive", "gpt-3.5-turbo"),
        ]
        
        for pattern, name, model in wellness_patterns:
            self.pattern_rules.append(PatternRule(
                name=name,
                pattern=re.compile(pattern, re.IGNORECASE),
                prompt_type=PromptType.GENERAL_WELLNESS,
                confidence=ShortcutConfidence.MEDIUM,
                suggested_model=model,
                priority=5
            ))
        
        # EXCLUSION PATTERNS (Things that should NOT trigger shortcuts)
        exclusion_patterns = [
            # Academic/theoretical discussions
            r"\b(?:research\s+paper|academic|theoretical|hypothesis|study\s+shows)\b",
            
            # Fictional/hypothetical scenarios
            r"\b(?:imagine\s+if|hypothetically|in\s+a\s+story|fictional)\b",
            
            # Medical/clinical terminology (should go to full evaluation)
            r"\b(?:diagnosis|symptoms|medical|clinical|therapy|treatment\s+plan)\b",
        ]
        
        self.exclusion_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in exclusion_patterns]
        
        # Sort rules by priority
        self.pattern_rules.sort(key=lambda rule: rule.priority)
        
        logger.info(f"Initialized with {len(self.pattern_rules)} pattern rules")
    
    def classify_prompt(self, prompt: str) -> Optional[ShortcutResult]:
        """
        Attempt to classify prompt using pattern shortcuts
        
        Args:
            prompt: User prompt to classify
            
        Returns:
            ShortcutResult if pattern match found, None otherwise
        """
        start_time = time.time()
        self.metrics.total_classifications += 1
        
        try:
            # Check exclusion patterns first
            for exclusion_pattern in self.exclusion_patterns:
                if exclusion_pattern.search(prompt.lower()):
                    logger.debug("Prompt matched exclusion pattern - skipping shortcuts")
                    return None
            
            # Find matching patterns
            matches = []
            for rule in self.pattern_rules:
                match_result = rule.matches(prompt)
                if match_result:
                    confidence, position = match_result
                    matches.append((rule, confidence, position))
            
            if not matches:
                return None
            
            # Sort by priority and confidence
            matches.sort(key=lambda x: (x[0].priority, -x[1]))
            best_match = matches[0]
            rule, confidence, position = best_match
            
            # Record pattern hit
            self.metrics.shortcut_hits += 1
            pattern_name = rule.name
            self.metrics.pattern_hit_counts[pattern_name] = self.metrics.pattern_hit_counts.get(pattern_name, 0) + 1
            
            # Calculate classification time
            classification_time = (time.time() - start_time) * 1000
            self.metrics.avg_classification_time_ms = (
                (self.metrics.avg_classification_time_ms * (self.metrics.shortcut_hits - 1) + classification_time) /
                self.metrics.shortcut_hits
            )
            
            # Create reasoning
            matched_patterns = [match[0].name for match in matches[:3]]  # Top 3 matches
            reasoning = f"Pattern shortcut: {rule.name} (confidence: {confidence:.2f}, patterns: {matched_patterns})"
            
            # Create result
            result = ShortcutResult(
                prompt_type=rule.prompt_type,
                confidence=confidence,
                suggested_model=rule.suggested_model,
                reasoning=reasoning,
                matched_patterns=matched_patterns,
                classification_time_ms=classification_time,
                bypass_full_evaluation=confidence >= ShortcutConfidence.HIGH.value
            )
            
            # Record classification
            self.recent_classifications.append({
                'timestamp': datetime.now().isoformat(),
                'prompt_length': len(prompt),
                'pattern_name': pattern_name,
                'prompt_type': rule.prompt_type.value,
                'confidence': confidence,
                'suggested_model': rule.suggested_model,
                'classification_time_ms': classification_time
            })
            
            logger.debug(f"Shortcut classification: {rule.prompt_type.value} via {pattern_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error in shortcut classification: {e}")
            return None
    
    def learn_from_evaluation(self, prompt: str, shortcut_result: ShortcutResult, full_evaluation_result: 'ModelSelection'):
        """
        Learn from comparison between shortcut and full evaluation
        
        Args:
            prompt: Original prompt
            shortcut_result: Result from shortcut classification
            full_evaluation_result: Result from full model selection
        """
        if not self.pattern_learning_enabled:
            return
        
        # Check if shortcut was correct
        shortcut_correct = (
            shortcut_result.prompt_type == full_evaluation_result.prompt_classification and
            abs(shortcut_result.confidence - full_evaluation_result.confidence_score) < 0.2
        )
        
        # Update pattern accuracy
        for pattern_name in shortcut_result.matched_patterns:
            if pattern_name not in self.metrics.pattern_accuracy:
                self.metrics.pattern_accuracy[pattern_name] = 0.0
            
            # Update running accuracy
            current_count = self.metrics.pattern_hit_counts.get(pattern_name, 1)
            current_accuracy = self.metrics.pattern_accuracy[pattern_name]
            
            if shortcut_correct:
                new_accuracy = ((current_accuracy * (current_count - 1)) + 1.0) / current_count
            else:
                new_accuracy = ((current_accuracy * (current_count - 1)) + 0.0) / current_count
                
                # Record false positive
                self.metrics.false_positives += 1
            
            self.metrics.pattern_accuracy[pattern_name] = new_accuracy
        
        # Store learning data
        learning_entry = {
            'prompt': prompt[:100],  # First 100 chars for privacy
            'shortcut_type': shortcut_result.prompt_type.value,
            'shortcut_confidence': shortcut_result.confidence,
            'full_type': full_evaluation_result.prompt_classification.value,
            'full_confidence': full_evaluation_result.confidence_score,
            'correct': shortcut_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        self._learning_data[shortcut_result.prompt_type.value].append(learning_entry)
        
        # Limit learning data size
        for prompt_type_data in self._learning_data.values():
            if len(prompt_type_data) > 100:
                prompt_type_data.pop(0)  # Remove oldest entry
        
        logger.debug(f"Learned from evaluation: shortcut {'correct' if shortcut_correct else 'incorrect'}")
    
    def get_pattern_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each pattern"""
        stats = {}
        
        for rule in self.pattern_rules:
            pattern_name = rule.name
            hit_count = self.metrics.pattern_hit_counts.get(pattern_name, 0)
            accuracy = self.metrics.pattern_accuracy.get(pattern_name, 0.0)
            
            stats[pattern_name] = {
                'hit_count': hit_count,
                'accuracy': accuracy,
                'prompt_type': rule.prompt_type.value,
                'confidence_level': rule.confidence.value,
                'priority': rule.priority
            }
        
        return stats
    
    def optimize_patterns(self) -> Dict[str, Any]:
        """Optimize patterns based on learning data"""
        optimization_results = {
            'patterns_adjusted': 0,
            'patterns_disabled': 0,
            'new_patterns_suggested': 0,
            'recommendations': []
        }
        
        # Analyze pattern performance
        for rule in self.pattern_rules[:]:  # Copy list to allow modification
            pattern_name = rule.name
            accuracy = self.metrics.pattern_accuracy.get(pattern_name, 1.0)
            hit_count = self.metrics.pattern_hit_counts.get(pattern_name, 0)
            
            # Disable poorly performing patterns
            if hit_count >= 10 and accuracy < 0.6:
                self.pattern_rules.remove(rule)
                optimization_results['patterns_disabled'] += 1
                optimization_results['recommendations'].append(
                    f"Disabled pattern '{pattern_name}' due to low accuracy ({accuracy:.2f})"
                )
                
            # Adjust confidence for moderately performing patterns
            elif hit_count >= 5 and 0.6 <= accuracy < 0.8:
                # Lower confidence level
                if rule.confidence == ShortcutConfidence.HIGH:
                    rule.confidence = ShortcutConfidence.MEDIUM
                    optimization_results['patterns_adjusted'] += 1
                    optimization_results['recommendations'].append(
                        f"Reduced confidence for pattern '{pattern_name}' from HIGH to MEDIUM"
                    )
        
        # Analyze learning data for new pattern opportunities
        for prompt_type, learning_entries in self._learning_data.items():
            if len(learning_entries) >= 20:  # Need sufficient data
                false_negatives = [entry for entry in learning_entries if not entry['correct']]
                
                if len(false_negatives) >= 5:
                    optimization_results['new_patterns_suggested'] += 1
                    optimization_results['recommendations'].append(
                        f"Consider adding new patterns for {prompt_type} - detected {len(false_negatives)} missed cases"
                    )
        
        logger.info(f"Pattern optimization complete: {optimization_results}")
        return optimization_results
    
    def add_custom_pattern(self, 
                          name: str, 
                          pattern: str, 
                          prompt_type: PromptType, 
                          confidence: ShortcutConfidence, 
                          suggested_model: str,
                          priority: int = 5) -> bool:
        """
        Add a custom pattern rule
        
        Args:
            name: Unique name for the pattern
            pattern: Regex pattern string
            prompt_type: Target prompt type
            confidence: Confidence level
            suggested_model: Model to suggest
            priority: Priority level (1=highest)
            
        Returns:
            True if pattern was added successfully
        """
        try:
            # Validate pattern
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            
            # Check for duplicate names
            if any(rule.name == name for rule in self.pattern_rules):
                logger.warning(f"Pattern with name '{name}' already exists")
                return False
            
            # Create and add rule
            new_rule = PatternRule(
                name=name,
                pattern=compiled_pattern,
                prompt_type=prompt_type,
                confidence=confidence,
                suggested_model=suggested_model,
                priority=priority
            )
            
            self.pattern_rules.append(new_rule)
            
            # Re-sort by priority
            self.pattern_rules.sort(key=lambda rule: rule.priority)
            
            logger.info(f"Added custom pattern: {name}")
            return True
            
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding custom pattern: {e}")
            return False
    
    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern rule by name"""
        for rule in self.pattern_rules:
            if rule.name == name:
                self.pattern_rules.remove(rule)
                logger.info(f"Removed pattern: {name}")
                return True
        
        logger.warning(f"Pattern '{name}' not found")
        return False
    
    def get_metrics(self) -> ShortcutMetrics:
        """Get current shortcut metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.metrics = ShortcutMetrics()
        self.recent_classifications.clear()
        self._learning_data.clear()
        logger.info("Reset shortcut metrics")
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export pattern configuration for backup/sharing"""
        return {
            'version': '1.0',
            'export_timestamp': datetime.now().isoformat(),
            'patterns': [
                {
                    'name': rule.name,
                    'pattern': rule.pattern.pattern,
                    'prompt_type': rule.prompt_type.value,
                    'confidence': rule.confidence.value,
                    'suggested_model': rule.suggested_model,
                    'priority': rule.priority,
                    'min_match_length': rule.min_match_length,
                    'context_required': rule.context_required
                }
                for rule in self.pattern_rules
            ],
            'metrics': self.metrics.pattern_accuracy
        }
    
    def import_patterns(self, pattern_data: Dict[str, Any]) -> bool:
        """Import pattern configuration from backup"""
        try:
            patterns = pattern_data.get('patterns', [])
            imported_count = 0
            
            for pattern_info in patterns:
                success = self.add_custom_pattern(
                    name=pattern_info['name'],
                    pattern=pattern_info['pattern'],
                    prompt_type=PromptType(pattern_info['prompt_type']),
                    confidence=ShortcutConfidence(pattern_info['confidence']),
                    suggested_model=pattern_info['suggested_model'],
                    priority=pattern_info.get('priority', 5)
                )
                
                if success:
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} pattern rules")
            return True
            
        except Exception as e:
            logger.error(f"Error importing patterns: {e}")
            return False