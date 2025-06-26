# Model Addition Checklist

Use this checklist to ensure complete and proper integration of new LLM models into the Mental Health LLM Evaluation framework.

## Pre-Implementation Setup

### 📋 Planning Phase
- [ ] **Choose Model Type**
  - [ ] Cloud-based (API) model
  - [ ] Local (self-hosted) model
  - [ ] Hybrid (API + local fallback)

- [ ] **Gather Requirements**
  - [ ] API documentation (for cloud models)
  - [ ] Model files/weights (for local models)  
  - [ ] Required Python packages/dependencies
  - [ ] Hardware requirements (GPU memory, etc.)
  - [ ] API keys/credentials needed

- [ ] **Select Template**
  - [ ] Use `new_model_template.py` for custom implementation
  - [ ] Use `claude_integration_template.py` for Anthropic models
  - [ ] Use `llama_integration_template.py` for Meta Llama models

## Implementation Phase

### 🔧 Core Implementation
- [ ] **Copy Template File**
  - [ ] Copy appropriate template to `src/models/{model_name}_client.py`
  - [ ] Rename class from template name to your model name
  - [ ] Update file header and docstring

- [ ] **Update Model Registration**
  - [ ] Set correct `name` in `@register_model_decorator`
  - [ ] Set appropriate `provider` (add to enum if needed)
  - [ ] Set correct `model_type` (CLOUD or LOCAL)
  - [ ] Write descriptive `description`
  - [ ] List all required packages in `requirements`
  - [ ] Define comprehensive `default_config`

- [ ] **Implement Required Methods**
  - [ ] `__init__()` - Model initialization
  - [ ] `generate_response()` - Main inference method
  - [ ] `validate_configuration()` - Config validation
  - [ ] `health_check()` - Service health verification
  - [ ] `get_model_info()` - Model metadata

- [ ] **Implement Helper Methods**
  - [ ] `_calculate_cost()` - Cost estimation
  - [ ] `_count_tokens()` - Token counting (if needed)
  - [ ] Model-specific utilities

### 🔌 Provider Integration
- [ ] **Add Provider to Enum** (if new)
  - [ ] Add to `ModelProvider` enum in `src/models/base_model.py`
  - [ ] Use descriptive, lowercase string value

- [ ] **Install Dependencies**
  - [ ] Install required packages: `pip install package-name`
  - [ ] Update `requirements.txt` if needed
  - [ ] Test import statements work

- [ ] **Setup Credentials** (for cloud models)
  - [ ] Set environment variables for API keys
  - [ ] Document required env vars in code comments
  - [ ] Test authentication works

- [ ] **Setup Model Files** (for local models)
  - [ ] Download model weights to specified path
  - [ ] Verify file structure and permissions
  - [ ] Test model loading works

## Configuration Phase

### ⚙️ Configuration Updates
- [ ] **Update Experiment Configuration**
  - [ ] Add model to `config/experiment_template.yaml`
  - [ ] Set `enabled: false` initially
  - [ ] Include all necessary parameters
  - [ ] Document any special requirements

- [ ] **Test Configuration Validation**
  - [ ] Run: `python scripts/model_management.py validate-config config/experiment_template.yaml`
  - [ ] Fix any validation errors
  - [ ] Verify all parameters are recognized

- [ ] **Update Package Imports**
  - [ ] Add import to `src/models/__init__.py` if needed
  - [ ] Verify auto-registration works
  - [ ] Test model appears in registry

## Testing Phase

### 🧪 Unit Testing
- [ ] **Create Test File**
  - [ ] Copy `templates/test_template.py` to `tests/models/test_{model_name}_client.py`
  - [ ] Update class names and imports
  - [ ] Customize test cases for your model

- [ ] **Implement Core Tests**
  - [ ] Test model initialization
  - [ ] Test response generation
  - [ ] Test configuration validation
  - [ ] Test health check functionality
  - [ ] Test error handling

- [ ] **Run Unit Tests**
  - [ ] Execute: `pytest tests/models/test_{model_name}_client.py -v`
  - [ ] Achieve >90% test coverage
  - [ ] Fix any failing tests

### 🔍 Integration Testing
- [ ] **Test Model Registration**
  - [ ] Verify model appears in: `python scripts/model_management.py list`
  - [ ] Check provider and type are correct
  - [ ] Verify availability status

- [ ] **Test Model Creation**
  - [ ] Run: `python scripts/model_management.py info {model_name}`
  - [ ] Verify all metadata is correct
  - [ ] Test factory creation works

- [ ] **Test Health Check**
  - [ ] Run: `python scripts/model_management.py test {model_name}`
  - [ ] Should pass without errors
  - [ ] Verify response generation works

### 🚀 End-to-End Testing
- [ ] **Test Pipeline Integration**
  - [ ] Enable model in config: `models.{type}[x].enabled: true`
  - [ ] Run: `python scripts/setup_experiment.py --config config/experiment_template.yaml --dry-run`
  - [ ] Should detect and validate your model

- [ ] **Test Conversation Generation**
  - [ ] Run: `python scripts/run_conversations.py --experiment test_exp --models {model_name} --dry-run`
  - [ ] Verify no errors in dry run
  - [ ] Test actual conversation generation

- [ ] **Test Full Pipeline**
  - [ ] Run complete pipeline with your model enabled
  - [ ] Verify evaluation and analysis work
  - [ ] Check reports include your model

## Documentation Phase

### 📚 Documentation Updates
- [ ] **Code Documentation**
  - [ ] Add comprehensive docstrings
  - [ ] Document all parameters and return values
  - [ ] Include usage examples in docstrings

- [ ] **Configuration Documentation**
  - [ ] Document all configuration parameters
  - [ ] Explain parameter effects and valid ranges
  - [ ] Include example configurations

- [ ] **Setup Instructions**
  - [ ] Document installation requirements
  - [ ] Explain credential setup process
  - [ ] Include troubleshooting tips

- [ ] **Update Main Documentation**
  - [ ] Add model to supported models list
  - [ ] Update model comparison tables
  - [ ] Include in usage examples

## Quality Assurance Phase

### ✅ Code Quality
- [ ] **Code Review**
  - [ ] Follow project coding standards
  - [ ] Use consistent naming conventions
  - [ ] Add appropriate error handling
  - [ ] Include logging statements

- [ ] **Performance Verification**
  - [ ] Test response times are reasonable
  - [ ] Verify memory usage is acceptable
  - [ ] Check for memory leaks (local models)
  - [ ] Test concurrent request handling

- [ ] **Security Review**
  - [ ] Ensure API keys are not hardcoded
  - [ ] Validate input sanitization
  - [ ] Check error messages don't leak sensitive info
  - [ ] Verify proper credential handling

### 🛡️ Mental Health Safety
- [ ] **Content Safety**
  - [ ] Test with mental health scenarios
  - [ ] Verify appropriate responses to crisis situations
  - [ ] Check boundary maintenance
  - [ ] Ensure no harmful advice is given

- [ ] **Response Quality**
  - [ ] Test empathy and understanding
  - [ ] Verify therapeutic appropriateness
  - [ ] Check cultural sensitivity
  - [ ] Ensure professional boundaries

## Deployment Phase

### 🚀 Production Readiness
- [ ] **Environment Setup**
  - [ ] Document production requirements
  - [ ] Test in production-like environment
  - [ ] Verify scaling characteristics
  - [ ] Set up monitoring and alerts

- [ ] **Configuration Management**
  - [ ] Create production configuration
  - [ ] Set appropriate rate limits
  - [ ] Configure retry logic
  - [ ] Set up cost monitoring (for paid APIs)

- [ ] **Rollout Planning**
  - [ ] Plan gradual rollout strategy
  - [ ] Set up A/B testing framework
  - [ ] Prepare rollback procedures
  - [ ] Define success metrics

## Post-Deployment Phase

### 📊 Monitoring & Maintenance
- [ ] **Performance Monitoring**
  - [ ] Monitor response times
  - [ ] Track error rates
  - [ ] Monitor resource usage
  - [ ] Set up alerts for issues

- [ ] **Quality Monitoring**
  - [ ] Review generated conversations
  - [ ] Monitor safety flags
  - [ ] Track user feedback
  - [ ] Analyze evaluation scores

- [ ] **Maintenance Planning**
  - [ ] Plan regular model updates
  - [ ] Schedule dependency updates
  - [ ] Plan capacity scaling
  - [ ] Document operational procedures

## Checklist Completion

### ✅ Final Verification
- [ ] **All Tests Pass**
  - [ ] Unit tests: 100% passing
  - [ ] Integration tests: 100% passing
  - [ ] End-to-end tests: 100% passing

- [ ] **Documentation Complete**
  - [ ] Code is fully documented
  - [ ] Setup instructions are clear
  - [ ] Configuration is documented
  - [ ] Troubleshooting guide exists

- [ ] **Production Ready**
  - [ ] Model works in production environment
  - [ ] Performance meets requirements
  - [ ] Security review passed
  - [ ] Mental health safety verified

## Quick Start Commands

Once your model is implemented, test it with these commands:

```bash
# 1. Verify model is registered
python scripts/model_management.py list

# 2. Test model configuration
python scripts/model_management.py info {model_name}

# 3. Run health check
python scripts/model_management.py test {model_name}

# 4. Validate full configuration
python scripts/model_management.py validate-config config/experiment_template.yaml

# 5. Test in pipeline
python scripts/setup_experiment.py --config config/experiment_template.yaml --dry-run

# 6. Run conversation generation
python scripts/run_conversations.py --experiment test --models {model_name} --dry-run
```

## Troubleshooting Common Issues

### Import Errors
- Verify all dependencies are installed
- Check Python path includes src directory
- Ensure model file is in correct location

### Registration Issues
- Verify decorator parameters are correct
- Check model name is unique
- Ensure provider enum includes your provider

### Authentication Errors
- Verify environment variables are set
- Check API key format and permissions
- Test credentials outside framework

### Memory Issues (Local Models)
- Enable quantization (8-bit or 4-bit)
- Reduce batch size or context length
- Check available GPU memory

### Performance Issues
- Optimize model loading time
- Implement response caching if needed
- Consider async processing for slow models

---

**Success Criteria:** Your model is successfully integrated when:
1. ✅ All checklist items are completed
2. ✅ All tests pass consistently
3. ✅ Model generates appropriate mental health responses
4. ✅ Integration with evaluation pipeline works flawlessly
5. ✅ Documentation is complete and accurate

**Estimated Time:** 2-8 hours depending on model complexity and familiarity with the framework.