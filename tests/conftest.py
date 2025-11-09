"""Test configuration for the project."""

import sys
from pathlib import Path

# Add the pages directory to Python path for testing
pages_dir = Path(__file__).parent.parent / "pages"
sys.path.insert(0, str(pages_dir))

# Add the src directory to Python path for testing
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))
