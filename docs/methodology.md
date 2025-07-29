# Research Methodology

## Overview

This capstone project evaluates the comparative effectiveness of cloud-based versus local Large Language Models (LLMs) for mental health conversation support. The study focuses on therapeutic quality, safety, and practical deployment considerations.

## Research Questions

1. **Primary**: How do local/open-source LLMs (DeepSeek, Gemma) compare to commercial cloud LLMs (OpenAI GPT-4, Claude) in therapeutic conversation quality for mental health support?

2. **Secondary**:
   - What are the cost-benefit trade-offs between local and cloud LLM deployment?
   - How do models perform across different mental health condition categories (anxiety, depression, crisis)?
   - What safety considerations arise when deploying LLMs for mental health applications?
   - Which model provides the best balance of quality, cost, and deployment flexibility?

## Methodology

### Experimental Design

**Design Type**: Comparative cross-sectional study
**Comparison**: Between-subjects comparison of four LLM systems
**Sample Size**: 10 standardized mental health scenarios × 1 response per model = 40 total responses

### Models Under Evaluation

1. **OpenAI GPT-4**: Cloud-based commercial model
   - Rationale: Industry-leading performance benchmark
   - Access: API-based, requires internet connectivity
   - Cost: Per-token pricing model

2. **Claude (Anthropic)**: Cloud-based commercial model
   - Rationale: Strong safety focus and conversational abilities
   - Access: API-based, requires internet connectivity
   - Cost: Per-token pricing model

3. **DeepSeek**: Local/cloud hybrid model
   - Rationale: High-performing open-source model for local deployment
   - Access: Self-hosted or API, offline capability
   - Cost: Free for local deployment

4. **Gemma (Google)**: Open-source model
   - Rationale: Lightweight, efficient model for resource-constrained deployment
   - Access: Self-hosted, offline capability
   - Cost: Free for local deployment

### Evaluation Scenarios

**Scenario Categories** (10 total):
- **Anxiety** (2 scenarios): General workplace anxiety, panic attacks
- **Depression** (2 scenarios): Recurrent depression, seasonal affective symptoms
- **Crisis** (2 scenarios): Suicidal ideation, self-harm urges
- **General Mental Health** (4 scenarios): Relationship issues, grief, academic stress, insomnia

**Scenario Design Criteria**:
- Realistic patient presentations
- Varied severity levels (mild, moderate, severe, crisis)
- Diverse demographic contexts
- Clear evaluation expectations

### Evaluation Metrics

#### Therapeutic Quality Dimensions

1. **Empathy Score** (30% weight)
   - Validation of feelings
   - Emotional reflection
   - Supportive language
   - Range: 0-10 points

2. **Therapeutic Value** (25% weight)
   - Helpful coping strategies
   - Psychoeducational content
   - Professional help referrals
   - Range: 0-10 points

3. **Safety Score** (35% weight) - *Highest Priority*
   - Absence of harmful advice
   - Appropriate crisis handling
   - Professional boundary maintenance
   - Range: 0-10 points

4. **Clarity Score** (10% weight)
   - Response readability
   - Appropriate length
   - Clear communication
   - Range: 0-10 points

#### Performance Metrics

- **Response Time**: Milliseconds to generate response
- **Cost Efficiency**: USD cost per response
- **Token Usage**: Input/output token consumption
- **Error Rate**: Failed response percentage

### Statistical Analysis Plan

**Note on Statistical Power**: This is an exploratory pilot study with n=10 conversations per model. Effect sizes are reported instead of p-values due to limited sample size. Future research should expand to n=20+ for adequate statistical power.

#### Descriptive Statistics
- Mean, median, standard deviation for all metrics
- Distribution analysis (histograms, box plots)
- Category-specific performance breakdowns

#### Inferential Statistics
- **Normality Testing**: Shapiro-Wilk test
- **Primary Analysis**: 
  - If normal: Independent t-test
  - If non-normal: Mann-Whitney U test
- **Effect Size**: Cohen's d calculation
- **Significance Level**: α = 0.05

#### Practical Significance Threshold
- Minimum meaningful difference: 0.5 points (medium effect size)
- Clinical significance: 1.0 points (large effect size)

### Data Collection Procedure

1. **Response Generation**:
   - Standardized system prompts for consistency
   - Temperature = 0.7 for balanced creativity/consistency
   - Max tokens = 2048 to prevent overly lengthy responses
   - Single response per scenario (no multiple trials)

2. **Evaluation Process**:
   - Automated scoring using validated linguistic patterns
   - Pattern-based safety violation detection
   - Professional help reference identification
   - Crisis indicator flagging

3. **Quality Assurance**:
   - Response content validation
   - Error handling and logging
   - Timestamp tracking for audit trail

### Limitations

1. **Sample Size**: Limited to 10 scenarios per model (n=20 total responses)
2. **Single Trial**: No multiple response averaging
3. **Automated Evaluation**: No human expert validation of scores
4. **Model Versions**: Specific to tested model versions (may change)
5. **Scenario Bias**: Researcher-designed scenarios may not represent all use cases
6. **Temporal Validity**: Results reflect current model capabilities only

### Ethical Considerations

1. **Synthetic Data**: All scenarios are researcher-created, not real patient data
2. **Safety Protocols**: Crisis responses include appropriate resource referrals
3. **Disclaimer Requirements**: Clear indication this is research, not therapeutic advice
4. **Harm Prevention**: Evaluation includes safety violation detection
5. **Professional Boundaries**: Models instructed to maintain appropriate limitations

### Expected Outcomes

**Hypothesis**: Cloud models (OpenAI) will demonstrate superior therapeutic quality due to larger training datasets and fine-tuning, but local models (DeepSeek) will provide comparable performance with significant cost advantages.

**Success Criteria**:
- Statistically significant difference detected (if present)
- Effect size calculation for practical significance
- Comprehensive safety assessment completed
- Cost-benefit analysis with deployment recommendations

### Reproducibility

- **Code Availability**: All evaluation code publicly available
- **Scenario Sharing**: Mental health scenarios provided for replication
- **Environment Documentation**: Model versions and parameters recorded
- **Random Seeds**: Set for consistent results when applicable

## Implementation Timeline

1. **Week 1-2**: Scenario development and validation
2. **Week 3**: Response generation and data collection
3. **Week 4**: Statistical analysis and visualization
4. **Week 5**: Results interpretation and reporting
5. **Week 6**: Final documentation and presentation preparation