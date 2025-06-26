# Mental Health LLM Evaluation Scenarios - Usage Guide

## 🎯 Overview

This collection provides 15 comprehensive, standardized mental health scenarios designed for rigorous LLM evaluation across diverse conditions, severity levels, and demographic groups. Each scenario includes detailed patient profiles, conversation goals, therapeutic elements, safety considerations, and evaluation criteria.

## 📋 Scenario Collection Summary

### **15 Standardized Scenarios Across 6 Categories:**

#### **Anxiety Disorders (3 scenarios)**
- **MH-ANX-001**: General Anxiety and Work Performance (Mild - 3/10)
- **MH-ANX-002**: Social Anxiety and Relationship Difficulties (Moderate - 6/10)  
- **MH-ANX-003**: Severe Generalized Anxiety with Physical Symptoms (Severe - 8/10)

#### **Depression (3 scenarios)**
- **MH-DEP-001**: Seasonal Low Mood and Motivation Issues (Mild - 3/10)
- **MH-DEP-002**: Persistent Sadness Following Life Transition (Moderate - 5/10)
- **MH-DEP-003**: Severe Depression with Hopelessness (Severe - 8/10)

#### **Stress Management (2 scenarios)**
- **MH-STR-001**: Overwhelming Work Deadlines and Burnout Risk (Moderate - 5/10)
- **MH-STR-002**: Relationship Conflict and Communication Breakdown (Moderate - 6/10)

#### **Crisis Situations (2 scenarios)**
- **MH-CRI-001**: Active Suicidal Ideation with Plan (Severe - 9/10)
- **MH-CRI-002**: Self-Harm Behavior and Emotional Dysregulation (Severe - 8/10)

#### **General Mental Health (3 scenarios)**
- **MH-GMH-001**: Chronic Insomnia and Sleep Anxiety (Moderate - 4/10)
- **MH-GMH-002**: Low Self-Esteem and Negative Self-Talk (Moderate - 5/10)
- **MH-GMH-003**: Major Life Transition and Identity Uncertainty (Moderate - 4/10)

#### **Substance Use Concerns (2 scenarios)**
- **MH-SUB-001**: Concerning Alcohol Use and Social Anxiety (Moderate - 6/10)
- **MH-SUB-002**: Prescription Opioid Misuse Following Surgery (Severe - 7/10)

## 🏗️ Scenario Structure

Each scenario includes these standardized components:

### **Core Information**
- `scenario_id`: Unique identifier (e.g., "MH-ANX-001")
- `title`: Descriptive scenario name
- `category`: Mental health category
- `severity`: Clinical severity rating (1-10 scale)

### **Patient Profile**
- `age`: Patient age (17-55 years across scenarios)
- `gender`: Gender identity (diverse representation)
- `occupation`: Professional/educational context
- `history`: Relevant mental health and treatment history

### **Conversation Framework**
- `opening_statement`: Realistic patient opening (2-3 sentences)
- `conversation_goals`: 4-5 specific therapeutic objectives
- `expected_therapeutic_elements`: Appropriate intervention techniques
- `red_flags`: Safety concerns and inappropriate responses to monitor

### **Dynamic Elements**
- `conversation_flow`: Minimum/maximum turn counts and branching points
- `evaluation_criteria`: Specific empathy markers, techniques, and safety priorities

## 🎯 Implementation Guidelines

### **Conversation Generation Requirements**
- **Exactly 20 conversations per scenario per model** (300 total per model)
- **8-15 turn conversations** as specified in requirements
- **Randomized scenario presentation** to prevent learning effects
- **Balanced demographic representation** across evaluation runs

### **Model Testing Protocol**
1. **Load scenarios** from YAML files using the scenario loader
2. **Present opening statement** to the LLM model
3. **Simulate patient responses** based on branching logic
4. **Generate 8-15 turn conversations** per scenario specification
5. **Apply evaluation criteria** throughout the conversation
6. **Document safety flags** and therapeutic responses

### **Safety Protocols**

#### **Crisis Scenarios (MH-CRI-001, MH-CRI-002)**
- ⚠️ **Immediate safety assessment required**
- 🚨 **Monitor for appropriate crisis intervention**
- 📞 **Evaluate emergency resource provision**
- 🛡️ **Assess risk mitigation strategies**

#### **All Scenarios**
- 🩺 **Monitor for inappropriate medical advice**
- 🚧 **Assess professional boundary maintenance**
- 🔗 **Evaluate referral recommendations**
- 🏃 **Track safety flag generation**

## 📊 Demographic Representation

### **Age Distribution**
- **Adolescent (1)**: 17 years
- **Young Adult (5)**: 19-29 years  
- **Adult (7)**: 31-45 years
- **Middle Age (2)**: 55 years

### **Gender Identity**
- **Female (8)**: Majority representation
- **Male (5)**: Substantial representation
- **Non-binary (2)**: LGBTQ+ inclusion

### **Occupational Diversity**
- Healthcare, Education, Technology, Business
- Manual Labor, Service, Students
- Unemployed, Retired

## 🔍 Evaluation Focus Areas

### **Universal Assessment (All 15 scenarios)**
- ✅ **Safety Assessment**: Risk evaluation and response
- ❤️ **Empathy Demonstration**: Validation and support
- 🤝 **Therapeutic Rapport**: Relationship building
- 🏥 **Resource Connection**: Appropriate referrals

### **Specialized Assessment**
- 🚨 **Crisis Intervention (2 scenarios)**: Immediate safety response
- 📚 **Psychoeducation (10 scenarios)**: Educational components
- 🛠️ **Coping Skill Building (12 scenarios)**: Skill development

## 📈 Severity Distribution for Comprehensive Testing

- **Mild (2 scenarios)**: Severity 3/10 - Early intervention focus
- **Moderate (8 scenarios)**: Severity 4-6/10 - Primary care level
- **Severe (5 scenarios)**: Severity 7-9/10 - Intensive support needs

## 🧪 Usage in Conversation Generation System

### **Integration with Existing System**
```python
# Load scenarios using the existing scenario loader
from src.scenarios.scenario import ScenarioLoader

loader = ScenarioLoader()
scenarios = loader.load_all_scenarios()  # Loads all 15 scenarios

# Generate exactly 20 conversations per scenario per model
for scenario in scenarios:
    for model in models:
        for i in range(20):  # 20 conversations per scenario per model
            conversation = conversation_manager.generate_conversation(
                model=model,
                scenario=scenario,
                conversation_id=f"{model.name}_{scenario.scenario_id}_{i:03d}"
            )
```

### **Branching Point Implementation**
Each scenario includes branching points that create dynamic conversation flows:
```yaml
branching_points:
  - turn: 3
    condition: "Patient mentions physical symptoms"
    options: ["explore_symptoms", "focus_cognitive", "discuss_coping"]
```

### **Evaluation Criteria Application**
```python
# Apply scenario-specific evaluation criteria
empathy_score = evaluate_empathy_markers(
    response=model_response,
    expected_markers=scenario.evaluation_criteria.empathy_markers
)

safety_score = evaluate_safety_response(
    response=model_response,
    safety_priorities=scenario.evaluation_criteria.safety_priorities
)
```

## 🛡️ Ethical Considerations

### **Crisis Scenario Handling**
- **Never minimize** suicidal ideation or self-harm
- **Always assess** immediate safety and risk
- **Provide appropriate** crisis resources and referrals
- **Document responses** for human review

### **Professional Boundaries**
- **No medical advice** or diagnosis provision
- **Appropriate referrals** to licensed professionals
- **Maintain therapeutic** but not treatment relationship
- **Respect cultural** and demographic differences

## 📚 File Organization

```
data/scenarios/
├── scenario_index.yaml              # Master index of all scenarios
├── anxiety_scenarios.yaml           # 3 anxiety disorder scenarios
├── depression_scenarios.yaml        # 3 depression scenarios
├── stress_scenarios.yaml           # 2 stress management scenarios
├── crisis_scenarios.yaml           # 2 crisis situation scenarios
├── general_mental_health_scenarios.yaml  # 3 general MH scenarios
└── substance_use_scenarios.yaml    # 2 substance use scenarios
```

## 🎯 Expected Outcomes

### **Comprehensive Model Evaluation**
- **300 conversations per model** (15 scenarios × 20 conversations)
- **Diverse condition coverage** across mental health spectrum
- **Severity range testing** from mild to severe presentations
- **Demographic sensitivity** assessment across age, gender, occupation
- **Safety protocol** evaluation in crisis situations

### **Comparative Analysis Capability**
- **Statistical significance** with 20 conversations per condition
- **Condition-specific performance** analysis
- **Severity-adjusted** evaluation metrics
- **Demographic bias** detection and measurement
- **Safety competency** comparison across models

This scenario collection provides the foundation for rigorous, comprehensive mental health LLM evaluation that meets clinical standards while enabling meaningful model comparison and improvement.