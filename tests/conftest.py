import sys
import os
# Ensure the project root (one level up from this file) is on PYTHONPATH
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
