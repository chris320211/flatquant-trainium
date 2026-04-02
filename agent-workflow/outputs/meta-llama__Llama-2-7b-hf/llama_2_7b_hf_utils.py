"""
Model-specific FlatQuant wrapper classes for Llama-2-7b-hf.

This module imports the FlatQuant wrapper classes from FlatQuantBundled
and re-exports them for use in calibration scripts.
"""

from flatquant.model_tools.llama_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention

__all__ = ['FlatQuantLlamaMLP', 'FlatQuantLlamaAttention']
