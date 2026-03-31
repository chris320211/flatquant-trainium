"""Pytest configuration and fixtures for Trainium2 testing."""

import os
import sys
from pathlib import Path

import pytest
import torch


# Make sure we can import from agent-workflow
sys.path.insert(0, str(Path(__file__).parent / "agent"))


@pytest.fixture
def trainium_available():
    """Check if Trainium/XLA is available."""
    try:
        import neuronx_distributed_inference
        return True
    except ImportError:
        return False


@pytest.fixture
def xla_available():
    """Check if XLA is available for Trainium acceleration."""
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False


@pytest.fixture
def device():
    """Get appropriate device for testing (Trainium or CPU)."""
    if os.environ.get("TRAINIUM_USE_XLA", "").lower() in ("1", "true", "yes"):
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except Exception as e:
            print(f"Warning: Could not get XLA device: {e}. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


@pytest.fixture
def dummy_model_config():
    """Provide a minimal model config for testing."""
    return {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
    }


@pytest.fixture
def dummy_input_ids():
    """Provide dummy input IDs for testing."""
    return torch.randint(0, 32000, (2, 128), dtype=torch.long)


@pytest.fixture
def dummy_attention_mask():
    """Provide dummy attention mask for testing."""
    return torch.ones((2, 128), dtype=torch.long)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "trainium: mark test as requiring Trainium hardware")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "weight: mark test as weight mapping test")
    config.addinivalue_line("markers", "slow: mark test as slow (> 5s)")
    config.addinivalue_line("markers", "xla: mark test as requiring XLA")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on module/function names."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Auto-mark weight tests
        if "weight" in item.nodeid:
            item.add_marker(pytest.mark.weight)

        # Auto-mark Trainium tests
        if "trainium" in item.nodeid:
            item.add_marker(pytest.mark.trainium)
