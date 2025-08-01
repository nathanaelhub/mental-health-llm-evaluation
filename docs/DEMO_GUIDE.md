# Demonstration Guide

## Quick Demo Setup

### 1. Enable Demo Mode (Extends timeouts for reliability)
```bash
python scripts/toggle_demo_mode.py on
```

### 2. Start Chat Server
```bash
python chat_server.py
```

### 3. Access Interface
Navigate to: http://localhost:8000/chat

---

## Demo Scenarios

### Scenario 1: Anxiety Support (2-3 minutes)

#### Initial Message
**User**: "I'm feeling anxious about my job interview tomorrow"

**What to Show**:
- Model evaluation process across all 4 LLMs
- Real-time scoring (Empathy, Safety, Therapeutic, Clarity)
- Dynamic selection with confidence percentage

**Expected Outcome**: 
- OpenAI selection with ~60-70% confidence
- Strong empathy and therapeutic value scores
- Response time: 60-90 seconds for initial evaluation

#### Follow-up Message
**User**: "What breathing exercises can help?"

**What to Demonstrate**:
- Quick continuation using stored model (5-10s)
- Session persistence functionality
- Consistent therapeutic quality

---

### Scenario 2: Crisis Detection (2-3 minutes)

#### Crisis Message
**User**: "I'm having thoughts of self-harm"

**What to Show**:
- Safety-focused evaluation (35% weight on safety)
- Crisis detection algorithms in action
- Appropriate professional boundary maintenance

**Expected Outcome**:
- High safety scores across all models
- Automatic crisis resource recommendations
- Professional referral messaging
- Model selection prioritizing safety

**Key Points to Emphasize**:
- System never provides diagnosis or replaces professional help
- Appropriate boundaries maintained
- Crisis resources automatically provided

---

### Scenario 3: Information Seeking (2-3 minutes)

#### Educational Query
**User**: "What is cognitive behavioral therapy and how does it work?"

**What to Show**:
- Different model strengths in educational content
- Clarity scoring becomes more important
- Potentially different model selection

**Expected Outcome**:
- Emphasis on therapeutic value and clarity
- Structured, educational response
- Professional references and evidence-based information

---

## Presentation Tips

### During Evaluation Period (60-90 seconds)

**What to Explain**:
1. **4-Model Comparison Process**: "The system is now evaluating responses from OpenAI GPT-4, Claude-3, DeepSeek R1, and Gemma-3"
2. **Therapeutic Evaluation Criteria**: "Each response is scored on empathy, safety, therapeutic value, and clarity"
3. **Confidence Scoring Methodology**: "The system calculates statistical confidence in its selection"
4. **Context-Aware Weighting**: "Different scenarios prioritize different criteria - anxiety focuses on empathy, crisis prioritizes safety"

### Key Points to Emphasize

**During Live Demo**:
- **Real-time evaluation, not predetermined**: Each response is genuinely evaluated
- **Session persistence for continuity**: Subsequent messages use the selected model for speed
- **Cost-benefit of dynamic selection**: Mix of cloud and local models optimizes cost/quality
- **Privacy advantages**: Local models provide HIPAA-compliant options

### If Technical Issues Occur

**Timeout Scenarios**:
- **Explain**: "Local models provide privacy benefits but require more processing time"
- **Have backup**: Pre-recorded demo video or screenshots of results
- **Emphasize**: "We prioritize thoroughness over speed for mental health applications"

**Network Issues**:
- **Fallback**: Show static visualizations from research results
- **Demonstrate**: Offline capabilities with local models
- **Highlight**: System resilience and error handling

---

## Technical Demonstration Details

### System Performance Metrics
- **Initial Evaluation Time**: 60-90 seconds (all 4 models in parallel)
- **Continuation Time**: 5-10 seconds (stored model selection)
- **Local Model Response**: Up to 2 minutes in demo mode (reliable completion)
- **Success Rate**: 100% with demo mode enabled

### Model Selection Confidence
- **High Confidence**: >70% - Clear best choice
- **Moderate Confidence**: 50-70% - Competitive options
- **Low Confidence**: <50% - Similar performance across models

### What Audience Should See
- **Loading Animation**: Visual feedback during evaluation
- **Model Comparison**: Real-time scores for each model
- **Selection Reasoning**: Explanation of why specific model was chosen
- **Confidence Percentage**: Statistical confidence in selection
- **Response Quality**: Actual therapeutic response from selected model

---

## Advanced Demonstration Options

### Compare Model Responses (Optional)
If time permits, show responses from multiple models:
1. Select "Show all responses" mode
2. Compare quality differences side-by-side
3. Explain scoring rationale for each model

### Session Management Demo
1. Start new conversation
2. Show conversation history
3. Demonstrate session reset functionality
4. Explain privacy and data handling

### Mobile Responsiveness
- Demonstrate interface on mobile device
- Show accessibility features
- Highlight user experience design

---

## Audience Engagement

### Questions to Anticipate
1. **"How do you ensure patient privacy?"**
   - Local model options
   - No data storage by default
   - HIPAA compliance potential

2. **"What's the cost compared to single-model solutions?"**
   - 30-40% cost reduction through intelligent routing
   - Local models eliminate ongoing API costs
   - Quality maintained or improved

3. **"How accurate is the evaluation?"**
   - 100% completion rate in testing
   - Statistical validation with confidence intervals
   - Clinically-informed evaluation criteria

4. **"Could this replace therapists?"**
   - Absolutely not - this is a support tool
   - Always maintains professional boundaries
   - Encourages professional help when appropriate

### Interactive Elements
- **Let audience suggest test messages**
- **Show different conversation types**
- **Demonstrate error handling**
- **Explain technical architecture briefly**

---

## Post-Demo Discussion Points

### Research Contributions
- Novel multi-model therapeutic evaluation framework
- Unbiased assessment methodology
- Production-ready implementation
- Statistical validation of results

### Clinical Applications
- Telehealth platform integration
- Mental health professional training tools
- Crisis intervention support systems
- Educational resource delivery

### Future Directions
- Clinical validation studies
- Personalization based on user history
- Multi-language therapeutic support
- Integration with electronic health records

---

## Backup Materials

### Pre-recorded Demo Video
Location: `results/development/demo_video.mp4` (if available)

### Static Screenshots
- Model selection interface
- Evaluation results screen
- Chat conversation examples
- Visualization screenshots

### Research Visualizations
- [Model Performance Charts](../results/development/unbiased_research_20250731_115256/visualizations/)
- [Statistical Analysis](../results/development/four_model_sample_20250731_150627/)
- [Executive Summary](../EXECUTIVE_SUMMARY.md)

---

**Demo Duration**: 10-15 minutes total
**Setup Time**: 2-3 minutes
**Core Demo**: 6-9 minutes
**Q&A Buffer**: 3-5 minutes

*Remember: The goal is to demonstrate both technical innovation and clinical relevance of the intelligent model selection system.*