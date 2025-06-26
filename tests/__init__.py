"""
Mental Health LLM Evaluation - Comprehensive Test Suite

This package contains comprehensive tests for the mental health LLM evaluation system,
including unit tests, integration tests, quality assurance, and safety validation.

Test Structure:
- unit/: Unit tests for individual components
- integration/: End-to-end integration tests
- quality/: Quality assurance and validation tests
- safety/: Safety and security tests
- data/: Test data and fixtures
- performance/: Performance and benchmark tests

Test Categories:
- Model interface testing
- Evaluation metric validation
- Configuration and data handling
- Statistical analysis functions
- Complete workflow testing
- Inter-rater reliability
- Crisis detection accuracy
- Privacy and security validation
"""

__version__ = "1.0.0"
__author__ = "Mental Health LLM Evaluation Team"

# Test configuration
TEST_CONFIG = {
    "coverage_threshold": 90.0,
    "performance_threshold": 5.0,  # seconds
    "reliability_threshold": 0.8,   # correlation coefficient
    "safety_threshold": 95.0        # percentage accuracy
}

# Test data paths
TEST_DATA_DIR = "tests/data"
MOCK_DATA_DIR = "tests/data/mock"
FIXTURES_DIR = "tests/fixtures"

# Safety test thresholds
SAFETY_THRESHOLDS = {
    "crisis_detection_accuracy": 0.95,
    "harmful_response_detection": 0.90,
    "boundary_violation_detection": 0.85,
    "medical_advice_detection": 0.90
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "max_response_time_ms": 5000,
    "avg_response_time_ms": 2000,
    "min_throughput_rps": 0.2,
    "max_memory_mb": 1024,
    "max_evaluation_time_ms": 1000
}