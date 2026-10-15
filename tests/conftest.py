"""
Pytest configuration and fixtures for jiuguan-voice-gateway
"""

import os
import sys
import pytest

# Ensure project root is at the head of sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "acceptance_live: marks tests requiring live GPT-SoVITS GPU service on port 9880"
    )
    config.addinivalue_line(
        "markers", "acceptance: marks acceptance and acoustic quality tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests for router and gateway logic"
    )
