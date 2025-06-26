# Contributing to Mental Health LLM Evaluation

Thank you for your interest in contributing to the Mental Health LLM Evaluation framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and mental health concepts
- Familiarity with Python development

### Development Environment

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/mental-health-llm-evaluation.git
   cd mental-health-llm-evaluation
   ```

2. **Set up Upstream Remote**
   ```bash
   git remote add upstream https://github.com/original-username/mental-health-llm-evaluation.git
   ```

3. **Create Development Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Local Environment

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .  # Install in development mode
   ```

3. **Set up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize Database**
   ```bash
   python scripts/init_database.py
   ```

### Docker Environment (Optional)

```bash
# Build development image
docker build -f Dockerfile.dev -t mental-health-llm-eval-dev .

# Run development container
docker run -it --gpus all -v $(pwd):/app mental-health-llm-eval-dev
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

#### 🐛 Bug Reports
- Found a bug? Please check existing issues first
- Use the bug report template
- Include system information and reproduction steps

#### ✨ Feature Requests
- Suggest new features or improvements
- Use the feature request template
- Explain the use case and expected behavior

#### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Refactoring
- Documentation improvements

#### 📚 Documentation
- API documentation
- Tutorials and guides
- Example code
- Translation improvements

#### 🧪 Testing
- Unit tests
- Integration tests
- Performance tests
- Test data creation

### Contribution Areas

#### High Priority Areas

1. **Evaluation Metrics**
   - New therapeutic assessment methods
   - Cultural sensitivity evaluation
   - Bias detection algorithms

2. **Safety Features**
   - Crisis detection improvements
   - Safety intervention protocols
   - Risk assessment algorithms

3. **Model Integration**
   - Support for new LLM architectures
   - Local model optimizations
   - API client improvements

4. **Performance**
   - Evaluation speed improvements
   - Memory optimization
   - Parallel processing enhancements

#### Medium Priority Areas

1. **User Interface**
   - Web dashboard improvements
   - CLI enhancements
   - Visualization tools

2. **Data Management**
   - Data export/import tools
   - Database optimizations
   - Data validation

3. **Documentation**
   - Tutorial improvements
   - API documentation
   - Deployment guides

#### Beginner-Friendly Areas

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Test coverage increases
- Code style fixes
- Example code creation

## Pull Request Process

### Before Creating a Pull Request

1. **Sync with Upstream**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run tests
   python -m pytest tests/ -v
   
   # Check code style
   pre-commit run --all-files
   
   # Test installation
   python scripts/verify_installation.py
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues
Closes #(issue_number)
```

### Review Process

1. **Automated Checks**
   - All tests must pass
   - Code style checks must pass
   - No merge conflicts

2. **Manual Review**
   - Code quality assessment
   - Documentation review
   - Feature functionality testing

3. **Approval Requirements**
   - At least one maintainer approval
   - All conversations resolved
   - CI/CD pipeline passes

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_models/
│   ├── test_evaluation/
│   └── test_analysis/
├── integration/             # Integration tests
│   ├── test_end_to_end/
│   └── test_workflows/
├── safety/                  # Safety-specific tests
│   └── test_crisis_detection/
├── quality/                 # Quality assurance tests
│   └── test_reliability/
└── fixtures/                # Test data and fixtures
```

### Writing Tests

#### Unit Tests

```python
import pytest
from src.evaluation.empathy_scorer import EmpathyScorer

class TestEmpathyScorer:
    @pytest.fixture
    def empathy_scorer(self):
        return EmpathyScorer()
    
    def test_score_empathy_high_empathy_response(self, empathy_scorer):
        """Test scoring of highly empathetic response."""
        response = "I can really understand how difficult this must be for you."
        patient_message = "I'm struggling with anxiety."
        context = "anxiety_support"
        
        score = empathy_scorer.score_empathy(response, patient_message, context)
        
        assert 7.0 <= score <= 10.0, f"Expected high empathy score, got {score}"
    
    def test_score_empathy_low_empathy_response(self, empathy_scorer):
        """Test scoring of low empathy response."""
        response = "Just get over it."
        patient_message = "I'm struggling with anxiety."
        context = "anxiety_support"
        
        score = empathy_scorer.score_empathy(response, patient_message, context)
        
        assert 0.0 <= score <= 4.0, f"Expected low empathy score, got {score}"
    
    @pytest.mark.parametrize("response,expected_range", [
        ("I understand your feelings.", (6.0, 9.0)),
        ("That's not a real problem.", (0.0, 3.0)),
        ("Let's work through this together.", (7.0, 10.0)),
    ])
    def test_empathy_scoring_ranges(self, empathy_scorer, response, expected_range):
        """Test empathy scoring with various responses."""
        patient_message = "I'm having a hard time."
        context = "general_support"
        
        score = empathy_scorer.score_empathy(response, patient_message, context)
        
        assert expected_range[0] <= score <= expected_range[1]
```

#### Integration Tests

```python
import pytest
import asyncio
from src.conversation.conversation_manager import ConversationManager
from src.models.openai_client import OpenAIClient
from src.evaluation.composite_scorer import CompositeScorer

@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndWorkflow:
    @pytest.fixture
    async def conversation_manager(self):
        config = {"enable_safety_monitoring": True}
        return ConversationManager(config)
    
    @pytest.fixture
    def mock_openai_client(self):
        # Return mock client for testing
        return MockOpenAIClient()
    
    async def test_complete_conversation_workflow(
        self, 
        conversation_manager, 
        mock_openai_client,
        sample_scenario
    ):
        """Test complete conversation generation and evaluation."""
        # Generate conversation
        conversation = await conversation_manager.generate_conversation(
            model_client=mock_openai_client,
            scenario=sample_scenario,
            conversation_id="test_integration"
        )
        
        # Verify conversation structure
        assert "conversation_metadata" in conversation
        assert "conversation_turns" in conversation
        assert len(conversation["conversation_turns"]) >= 4
        
        # Evaluate conversation
        scorer = CompositeScorer()
        score = scorer.calculate_composite_score(
            conversation, 
            sample_scenario.scenario_id
        )
        
        # Verify evaluation results
        assert 0.0 <= score.overall_score <= 10.0
        assert hasattr(score, 'therapeutic_details')
        assert hasattr(score, 'technical_details')
```

#### Safety Tests

```python
@pytest.mark.safety
class TestCrisisDetection:
    @pytest.fixture
    def safety_detector(self):
        from src.evaluation.safety_detector import SafetyDetector
        return SafetyDetector()
    
    @pytest.mark.parametrize("statement,should_detect", [
        ("I want to kill myself", True),
        ("I'm feeling sad today", False),
        ("Everyone would be better without me", True),
        ("This movie is killing me with laughter", False),
    ])
    def test_crisis_detection_accuracy(self, safety_detector, statement, should_detect):
        """Test crisis detection accuracy."""
        flags, level = safety_detector.detect_safety_issues(statement)
        
        has_crisis_flags = any(
            flag.value in ['suicidal_ideation', 'self_harm'] 
            for flag in flags
        )
        
        assert has_crisis_flags == should_detect, (
            f"Statement: '{statement}' - "
            f"Expected crisis detection: {should_detect}, "
            f"Got: {has_crisis_flags} (flags: {flags})"
        )
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/safety/ -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
python -m pytest tests/unit/test_empathy_scorer.py -v

# Run tests matching pattern
python -m pytest -k "test_empathy" -v

# Run tests with specific markers
python -m pytest -m "safety" -v

# Run tests in parallel
python -m pytest tests/ -n auto
```

## Documentation

### Documentation Types

1. **API Documentation**
   - Docstrings for all public functions/classes
   - Type hints required
   - Examples in docstrings

2. **User Guides**
   - Installation instructions
   - Usage tutorials
   - Configuration guides

3. **Developer Documentation**
   - Architecture overview
   - Contributing guidelines
   - API reference

### Docstring Style

Follow Google-style docstrings:

```python
def score_empathy(
    self,
    response: str,
    patient_message: str,
    context: str = "general"
) -> float:
    """
    Score the empathy level of an assistant's response.
    
    This function evaluates how well the assistant's response demonstrates
    empathy towards the patient's emotional state and concerns.
    
    Args:
        response: The assistant's response to evaluate.
        patient_message: The patient's message being responded to.
        context: The conversation context or scenario type.
            Defaults to "general".
    
    Returns:
        Empathy score between 0.0 and 10.0, where higher scores
        indicate greater empathy.
    
    Raises:
        ValueError: If response or patient_message is empty.
        TypeError: If inputs are not strings.
    
    Example:
        >>> scorer = EmpathyScorer()
        >>> score = scorer.score_empathy(
        ...     "I understand how difficult this must be for you.",
        ...     "I'm struggling with anxiety.",
        ...     "anxiety_support"
        ... )
        >>> print(f"Empathy score: {score:.2f}")
        Empathy score: 8.5
    """
```

### Documentation Building

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8080 -d _build/html
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use double quotes for strings
- Use type hints for all public functions
- Use meaningful variable names

### Automated Formatting

```bash
# Format code with Black
black src/ tests/ scripts/

# Sort imports with isort
isort src/ tests/ scripts/

# Check style with flake8
flake8 src/ tests/ scripts/

# Type checking with mypy
mypy src/
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-yaml
      - id: check-json
      - id: check-toml

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Code Review Checklist

#### For Authors
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No sensitive information exposed
- [ ] Performance considerations addressed

#### For Reviewers
- [ ] Code is readable and maintainable
- [ ] Logic is correct and efficient
- [ ] Tests adequately cover new code
- [ ] Documentation is clear and complete
- [ ] Security implications considered
- [ ] Backwards compatibility maintained

## Issue Reporting

### Bug Reports

Use the bug report template and include:

1. **Environment Information**
   - Operating system
   - Python version
   - Package versions
   - Hardware specifications

2. **Reproduction Steps**
   - Minimal code to reproduce
   - Expected vs. actual behavior
   - Error messages and stack traces

3. **Context**
   - What you were trying to accomplish
   - Any recent changes
   - Workarounds attempted

### Feature Requests

Use the feature request template and include:

1. **Problem Description**
   - What problem does this solve?
   - Who would benefit?
   - How urgent is this need?

2. **Proposed Solution**
   - Detailed description of desired functionality
   - Alternative solutions considered
   - Implementation suggestions

3. **Additional Context**
   - Mockups, examples, or references
   - Related issues or discussions

### Security Issues

For security vulnerabilities:

1. **DO NOT** create public issues
2. Email security@[project-email] with details
3. Include steps to reproduce
4. Allow reasonable time for response

## Community

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discord/Slack**: Real-time community chat
- **Mailing List**: Major announcements

### Community Guidelines

1. **Be Respectful**: Treat all community members with respect
2. **Be Constructive**: Provide helpful feedback and suggestions
3. **Be Patient**: Maintainers and contributors are often volunteers
4. **Be Inclusive**: Welcome newcomers and diverse perspectives

### Recognition

Contributors are recognized through:

- **AUTHORS.md**: List of all contributors
- **Release Notes**: Acknowledgment of major contributions
- **GitHub Contributors**: Automatic recognition on repository
- **Special Thanks**: Notable contributions highlighted

### Maintainer Responsibilities

Project maintainers:

- Respond to issues and PRs in reasonable time
- Provide constructive feedback
- Maintain project vision and standards
- Foster inclusive community environment
- Make final decisions on controversial topics

## Getting Help

### For Contributors

- Check existing documentation first
- Search closed issues for similar problems
- Ask questions in GitHub Discussions
- Join community chat for real-time help

### For Maintainers

- Follow project governance guidelines
- Consult other maintainers for major decisions
- Document decisions and rationale
- Maintain consistent communication style

---

Thank you for contributing to the Mental Health LLM Evaluation project! Your contributions help advance responsible AI development in mental health applications.

For questions about contributing, please:
- Open a discussion on GitHub
- Check the [documentation](docs/)
- Contact the maintainers directly

**Remember**: Every contribution, no matter how small, makes a difference! 🎉