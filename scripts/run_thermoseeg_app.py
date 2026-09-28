"""
Launch the permanent ThermoSEEG workstation.
"""

from pathlib import Path
import sys

# Add the repository root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.main_window import launch_application


if __name__ == "__main__":
    launch_application()