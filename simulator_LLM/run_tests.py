import sys
import os
import pytest

# Add the project directory to the python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

sys.exit(pytest.main(["-p", "no:pytest11", "/home/ugrp/simulator/simulator_LLM/tests/test_lane_loader.py"]))