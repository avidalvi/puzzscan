"""Pytest configuration — adds project root to sys.path."""
import sys
from pathlib import Path

# Add project root so that `src`, `utils`, etc. are importable
sys.path.insert(0, str(Path(__file__).parent))
