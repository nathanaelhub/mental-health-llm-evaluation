"""
Patient Scenario System

This module defines the Scenario class for loading and managing standardized
patient scenarios from YAML files for mental health LLM evaluation.
"""

import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Severity levels for mental health scenarios."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRISIS = "crisis"


class ScenarioType(Enum):
    """Types of mental health scenarios."""
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    STRESS = "stress"
    GRIEF = "grief"
    RELATIONSHIPS = "relationships"
    TRAUMA = "trauma"
    SUBSTANCE_USE = "substance_use"
    EATING_DISORDER = "eating_disorder"
    BIPOLAR = "bipolar"
    SELF_HARM = "self_harm"
    CRISIS = "crisis"


@dataclass
class PatientProfile:
    """Patient profile information for scenario."""
    
    name: str = "Patient"
    age: int = 30
    gender: str = "unspecified"
    background: str = ""
    presenting_concern: str = ""
    history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    therapy_experience: str = "none"  # none, limited, experienced
    communication_style: str = "direct"  # direct, indirect, hesitant, verbose
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "background": self.background,
            "presenting_concern": self.presenting_concern,
            "history": self.history,
            "current_medications": self.current_medications,
            "therapy_experience": self.therapy_experience,
            "communication_style": self.communication_style
        }


@dataclass
class ConversationBranch:
    """Defines a conversation branch point and possible responses."""
    
    turn_number: int
    trigger_condition: str  # Text pattern or condition that triggers this branch
    patient_responses: List[str]  # Possible patient responses for this branch
    expected_assistant_elements: List[str] = field(default_factory=list)
    severity_escalation: bool = False
    safety_flag: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_number": self.turn_number,
            "trigger_condition": self.trigger_condition,
            "patient_responses": self.patient_responses,
            "expected_assistant_elements": self.expected_assistant_elements,
            "severity_escalation": self.severity_escalation,
            "safety_flag": self.safety_flag
        }


@dataclass
class EvaluationCriteria:
    """Evaluation criteria for the scenario."""
    
    required_therapeutic_elements: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    prohibited_responses: List[str] = field(default_factory=list)
    empathy_indicators: List[str] = field(default_factory=list)
    crisis_response_required: bool = False
    professional_boundaries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "required_therapeutic_elements": self.required_therapeutic_elements,
            "safety_requirements": self.safety_requirements,
            "prohibited_responses": self.prohibited_responses,
            "empathy_indicators": self.empathy_indicators,
            "crisis_response_required": self.crisis_response_required,
            "professional_boundaries": self.professional_boundaries
        }


@dataclass
class Scenario:
    """Complete mental health patient scenario."""
    
    # Basic scenario information
    scenario_id: str
    title: str
    description: str
    scenario_type: ScenarioType
    severity_level: SeverityLevel
    
    # Patient information
    patient_profile: PatientProfile
    opening_statement: str
    
    # Conversation structure
    expected_conversation_length: Tuple[int, int] = (8, 15)  # min, max turns
    conversation_branches: List[ConversationBranch] = field(default_factory=list)
    
    # Follow-up responses for different turns
    patient_responses: Dict[int, List[str]] = field(default_factory=dict)
    
    # Evaluation criteria
    evaluation_criteria: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    
    # Metadata
    created_by: str = "system"
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate scenario after initialization."""
        self._validate_scenario()
    
    def _validate_scenario(self):
        """Validate scenario configuration."""
        if not self.scenario_id:
            raise ValueError("Scenario ID is required")
        
        if not self.opening_statement:
            raise ValueError("Opening statement is required")
        
        min_turns, max_turns = self.expected_conversation_length
        if min_turns < 1 or max_turns < min_turns:
            raise ValueError(f"Invalid conversation length: {self.expected_conversation_length}")
        
        # Validate branch turn numbers
        for branch in self.conversation_branches:
            if branch.turn_number < 1 or branch.turn_number > max_turns:
                logger.warning(f"Branch turn {branch.turn_number} outside expected range")
    
    def get_patient_response(self, turn_number: int, assistant_message: str = "") -> str:
        """
        Get appropriate patient response for the given turn.
        
        Args:
            turn_number: Current conversation turn
            assistant_message: Previous assistant message (for context)
            
        Returns:
            Patient response text
        """
        # Check for conversation branches first
        for branch in self.conversation_branches:
            if (branch.turn_number == turn_number and 
                self._matches_branch_condition(branch.trigger_condition, assistant_message)):
                # Randomly select from branch responses
                import random
                return random.choice(branch.patient_responses)
        
        # Use default responses for the turn
        if turn_number in self.patient_responses:
            import random
            return random.choice(self.patient_responses[turn_number])
        
        # Generate generic response based on conversation stage
        return self._generate_generic_response(turn_number)
    
    def _matches_branch_condition(self, condition: str, message: str) -> bool:
        """Check if message matches branch trigger condition."""
        if not condition or not message:
            return False
        
        # Simple keyword matching for now
        # Could be enhanced with regex or NLP matching
        condition_lower = condition.lower()
        message_lower = message.lower()
        
        # Check for keyword presence
        if condition.startswith("contains:"):
            keyword = condition[9:].strip().lower()
            return keyword in message_lower
        
        # Check for question patterns
        if condition == "asks_question":
            return "?" in message
        
        # Check for empathy patterns
        if condition == "shows_empathy":
            empathy_patterns = [
                "understand", "sorry", "difficult", "hard", "feel", "sounds like"
            ]
            return any(pattern in message_lower for pattern in empathy_patterns)
        
        # Default to simple contains check
        return condition_lower in message_lower
    
    def _generate_generic_response(self, turn_number: int) -> str:
        """Generate generic patient response based on turn number."""
        # Early conversation responses
        if turn_number <= 3:
            responses = [
                "I'm not sure where to start...",
                "It's been really difficult lately.",
                "I've been struggling with this for a while.",
                "I'm hoping you can help me understand what's going on."
            ]
        
        # Mid-conversation responses
        elif turn_number <= 8:
            responses = [
                "That makes sense, I hadn't thought of it that way.",
                "I'm not sure if I can do that.",
                "How would that help exactly?",
                "I've tried something similar before.",
                "That sounds challenging but maybe worth trying."
            ]
        
        # Later conversation responses
        else:
            responses = [
                "This has been really helpful to talk about.",
                "I feel like I understand myself better now.",
                "I think I have some ideas about what to try next.",
                "Thank you for listening and understanding.",
                "I'm feeling a bit more hopeful about things."
            ]
        
        import random
        return random.choice(responses)
    
    def should_end_conversation(self, turn_number: int, last_response: str = "") -> bool:
        """Determine if conversation should end naturally."""
        min_turns, max_turns = self.expected_conversation_length
        
        # Must reach minimum turns
        if turn_number < min_turns:
            return False
        
        # Must end by maximum turns
        if turn_number >= max_turns:
            return True
        
        # Check for natural ending cues
        ending_cues = [
            "thank you", "helpful", "better", "clearer", "understand",
            "try that", "next steps", "feeling hopeful"
        ]
        
        if last_response:
            response_lower = last_response.lower()
            if any(cue in response_lower for cue in ending_cues):
                # End conversation with some probability after natural cue
                import random
                return random.random() < 0.7  # 70% chance to end
        
        return False
    
    def get_safety_flags(self) -> List[str]:
        """Get list of safety-related flags for this scenario."""
        flags = []
        
        if self.severity_level == SeverityLevel.CRISIS:
            flags.append("CRISIS_SCENARIO")
        
        if self.evaluation_criteria.crisis_response_required:
            flags.append("CRISIS_RESPONSE_REQUIRED")
        
        for branch in self.conversation_branches:
            if branch.safety_flag:
                flags.append(f"SAFETY_BRANCH_TURN_{branch.turn_number}")
        
        # Check scenario type for specific safety considerations
        if self.scenario_type == ScenarioType.SELF_HARM:
            flags.append("SELF_HARM_CONTENT")
        elif self.scenario_type == ScenarioType.SUBSTANCE_USE:
            flags.append("SUBSTANCE_USE_CONTENT")
        elif self.scenario_type == ScenarioType.TRAUMA:
            flags.append("TRAUMA_CONTENT")
        
        return flags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "severity_level": self.severity_level.value,
            "patient_profile": self.patient_profile.to_dict(),
            "opening_statement": self.opening_statement,
            "expected_conversation_length": self.expected_conversation_length,
            "conversation_branches": [branch.to_dict() for branch in self.conversation_branches],
            "patient_responses": self.patient_responses,
            "evaluation_criteria": self.evaluation_criteria.to_dict(),
            "created_by": self.created_by,
            "version": self.version,
            "tags": self.tags,
            "safety_flags": self.get_safety_flags()
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Scenario':
        """Load scenario from YAML file."""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            return cls.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load scenario from {yaml_path}: {e}")
            raise
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        """Create scenario from dictionary."""
        try:
            # Parse patient profile
            patient_data = data.get("patient_profile", {})
            patient_profile = PatientProfile(
                name=patient_data.get("name", "Patient"),
                age=patient_data.get("age", 30),
                gender=patient_data.get("gender", "unspecified"),
                background=patient_data.get("background", ""),
                presenting_concern=patient_data.get("presenting_concern", ""),
                history=patient_data.get("history", []),
                current_medications=patient_data.get("current_medications", []),
                therapy_experience=patient_data.get("therapy_experience", "none"),
                communication_style=patient_data.get("communication_style", "direct")
            )
            
            # Parse conversation branches
            branches = []
            for branch_data in data.get("conversation_branches", []):
                branch = ConversationBranch(
                    turn_number=branch_data["turn_number"],
                    trigger_condition=branch_data["trigger_condition"],
                    patient_responses=branch_data["patient_responses"],
                    expected_assistant_elements=branch_data.get("expected_assistant_elements", []),
                    severity_escalation=branch_data.get("severity_escalation", False),
                    safety_flag=branch_data.get("safety_flag", False)
                )
                branches.append(branch)
            
            # Parse evaluation criteria
            eval_data = data.get("evaluation_criteria", {})
            evaluation_criteria = EvaluationCriteria(
                required_therapeutic_elements=eval_data.get("required_therapeutic_elements", []),
                safety_requirements=eval_data.get("safety_requirements", []),
                prohibited_responses=eval_data.get("prohibited_responses", []),
                empathy_indicators=eval_data.get("empathy_indicators", []),
                crisis_response_required=eval_data.get("crisis_response_required", False),
                professional_boundaries=eval_data.get("professional_boundaries", [])
            )
            
            # Create scenario
            scenario = cls(
                scenario_id=data["scenario_id"],
                title=data["title"],
                description=data["description"],
                scenario_type=ScenarioType(data["scenario_type"]),
                severity_level=SeverityLevel(data["severity_level"]),
                patient_profile=patient_profile,
                opening_statement=data["opening_statement"],
                expected_conversation_length=tuple(data.get("expected_conversation_length", [8, 15])),
                conversation_branches=branches,
                patient_responses=data.get("patient_responses", {}),
                evaluation_criteria=evaluation_criteria,
                created_by=data.get("created_by", "system"),
                version=data.get("version", "1.0"),
                tags=data.get("tags", [])
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Failed to create scenario from dict: {e}")
            raise
    
    def save_to_yaml(self, yaml_path: Path):
        """Save scenario to YAML file."""
        try:
            # Convert to dict and clean up for YAML
            data = self.to_dict()
            
            # Remove safety_flags as they're computed
            data.pop("safety_flags", None)
            
            with open(yaml_path, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, default_flow_style=False, indent=2, allow_unicode=True)
            
            logger.info(f"Scenario saved to {yaml_path}")
            
        except Exception as e:
            logger.error(f"Failed to save scenario to {yaml_path}: {e}")
            raise


class ScenarioLoader:
    """Utility class for loading and managing scenarios."""
    
    def __init__(self, scenarios_dir: Optional[Path] = None):
        """
        Initialize scenario loader.
        
        Args:
            scenarios_dir: Directory containing scenario YAML files
        """
        if scenarios_dir is None:
            # Default to scenarios directory relative to this file
            self.scenarios_dir = Path(__file__).parent.parent.parent / "scenarios"
        else:
            self.scenarios_dir = Path(scenarios_dir)
        
        self.logger = logging.getLogger(__name__)
        self._scenarios_cache = {}
    
    def load_scenario(self, scenario_id: str) -> Scenario:
        """Load a specific scenario by ID."""
        if scenario_id in self._scenarios_cache:
            return self._scenarios_cache[scenario_id]
        
        # Look for YAML file with matching ID
        yaml_file = self.scenarios_dir / f"{scenario_id}.yaml"
        if yaml_file.exists():
            scenario = Scenario.from_yaml(yaml_file)
            self._scenarios_cache[scenario_id] = scenario
            return scenario
        
        # Search all YAML files for matching ID
        for yaml_file in self.scenarios_dir.glob("*.yaml"):
            try:
                scenario = Scenario.from_yaml(yaml_file)
                if scenario.scenario_id == scenario_id:
                    self._scenarios_cache[scenario_id] = scenario
                    return scenario
            except Exception as e:
                self.logger.warning(f"Failed to load scenario from {yaml_file}: {e}")
        
        raise FileNotFoundError(f"Scenario with ID '{scenario_id}' not found")
    
    def load_all_scenarios(self) -> List[Scenario]:
        """Load all scenarios from the scenarios directory."""
        scenarios = []
        
        if not self.scenarios_dir.exists():
            self.logger.warning(f"Scenarios directory does not exist: {self.scenarios_dir}")
            return scenarios
        
        for yaml_file in self.scenarios_dir.glob("*.yaml"):
            try:
                scenario = Scenario.from_yaml(yaml_file)
                scenarios.append(scenario)
                self._scenarios_cache[scenario.scenario_id] = scenario
            except Exception as e:
                self.logger.error(f"Failed to load scenario from {yaml_file}: {e}")
        
        self.logger.info(f"Loaded {len(scenarios)} scenarios from {self.scenarios_dir}")
        return scenarios
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[Scenario]:
        """Get all scenarios of a specific type."""
        all_scenarios = self.load_all_scenarios()
        return [s for s in all_scenarios if s.scenario_type == scenario_type]
    
    def get_scenarios_by_severity(self, severity_level: SeverityLevel) -> List[Scenario]:
        """Get all scenarios of a specific severity level."""
        all_scenarios = self.load_all_scenarios()
        return [s for s in all_scenarios if s.severity_level == severity_level]
    
    def get_evaluation_suite(self, suite_name: str = "comprehensive") -> List[Scenario]:
        """Get a predefined evaluation suite of scenarios."""
        all_scenarios = self.load_all_scenarios()
        
        if suite_name == "comprehensive":
            # Return all scenarios
            return all_scenarios
        
        elif suite_name == "basic":
            # Return representative scenarios from each category
            suite = []
            for scenario_type in ScenarioType:
                type_scenarios = [s for s in all_scenarios if s.scenario_type == scenario_type]
                if type_scenarios:
                    # Pick one of each severity level if available
                    for severity in [SeverityLevel.MILD, SeverityLevel.MODERATE]:
                        severity_scenarios = [s for s in type_scenarios if s.severity_level == severity]
                        if severity_scenarios:
                            suite.append(severity_scenarios[0])
                            break
            return suite
        
        elif suite_name == "crisis":
            # Return only crisis and severe scenarios
            return [s for s in all_scenarios 
                   if s.severity_level in [SeverityLevel.SEVERE, SeverityLevel.CRISIS]]
        
        elif suite_name == "mild":
            # Return only mild scenarios for initial testing
            return [s for s in all_scenarios if s.severity_level == SeverityLevel.MILD]
        
        else:
            self.logger.warning(f"Unknown evaluation suite: {suite_name}")
            return all_scenarios
    
    def validate_scenarios(self) -> Dict[str, List[str]]:
        """Validate all scenarios and return any issues found."""
        issues = {}
        
        for yaml_file in self.scenarios_dir.glob("*.yaml"):
            try:
                scenario = Scenario.from_yaml(yaml_file)
                file_issues = []
                
                # Check for required fields
                if not scenario.opening_statement:
                    file_issues.append("Missing opening statement")
                
                if not scenario.patient_responses:
                    file_issues.append("No patient responses defined")
                
                # Check conversation length
                min_turns, max_turns = scenario.expected_conversation_length
                if max_turns - min_turns < 3:
                    file_issues.append("Conversation length range too narrow")
                
                # Check for evaluation criteria
                if not scenario.evaluation_criteria.required_therapeutic_elements:
                    file_issues.append("No required therapeutic elements defined")
                
                if file_issues:
                    issues[str(yaml_file)] = file_issues
                    
            except Exception as e:
                issues[str(yaml_file)] = [f"Failed to load: {str(e)}"]
        
        return issues