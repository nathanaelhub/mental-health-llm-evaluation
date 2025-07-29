"""
Research Module
===============

Modular architecture for Mental Health LLM Evaluation Research.

Components:
- display: UI/display functions (progress bars, formatting)
- evaluation: Core evaluation logic and retry mechanisms
- analysis: Statistical analysis functions
- visualization: Chart generation (moved here when created)
- file_io: Results saving/loading operations
- utils: Helper functions and utilities
"""

from .display import *
from .evaluation import *
from .utils import *

# Optional imports that may not exist yet
try:
    from .analysis import *
except ImportError:
    pass

try:
    from .file_io import *
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = [
    # Core functionality will be exposed here
]