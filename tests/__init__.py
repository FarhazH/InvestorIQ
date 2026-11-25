import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force demo mode for all tests
os.environ["DEMO_MODE"] = "true"
