# Data Directory

This directory contains all data files for the mental health LLM evaluation project.

## Directory Structure

- `scenarios/` - Therapeutic conversation scenarios for evaluation
- `conversations/` - Generated conversation transcripts between users and models
- `evaluations/` - Evaluation results and metrics
- `results/` - Final analysis results and comparisons

## Data Format Guidelines

### Scenarios
- JSON format with standardized schema
- Include metadata about category, severity, and expected qualities
- Store safety considerations and evaluation criteria

### Conversations
- JSON format with complete message history
- Include timing data and model response metadata
- Preserve scenario context and user simulation data

### Evaluations
- JSON format with comprehensive metrics
- Include technical, therapeutic, and patient experience scores
- Store detailed breakdowns and component metrics

### Results
- Multiple formats supported (JSON, CSV, Parquet)
- Include statistical analysis results
- Store visualization outputs and reports

## Privacy and Safety

- All data is synthetic and anonymized
- No real patient information is stored
- Conversations are generated for research purposes only
- Safety considerations are documented for each scenario

## Usage Notes

- Data files are automatically managed by the evaluation system
- Manual editing should preserve the expected schema
- Backup important evaluation results before cleanup operations
- Use the data storage utilities for programmatic access