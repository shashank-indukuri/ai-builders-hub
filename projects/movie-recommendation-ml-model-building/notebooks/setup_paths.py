import sys
from pathlib import Path

# Get the project root directory (two levels up from this file)
project_root = Path(__file__).resolve().parent.parent

# Add the project root to the Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Print the path to verify it's working
print(f"Project root added to Python path: {project_root}")
print(f"Current Python path: {sys.path}")
