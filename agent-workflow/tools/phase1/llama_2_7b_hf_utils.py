"""
Model-specific FlatQuant wrapper classes for Llama-2-7b-hf.

This module imports the FlatQuant wrapper classes from FlatQuantBundled
and re-exports them for use in calibration scripts.

These classes are the ONLY thing needed from FlatQuantBundled to apply
INT4 quantization to Llama-2-7b-hf.
"""

from flatquant.model_tools.llama_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention

__all__ = ['FlatQuantLlamaMLP', 'FlatQuantLlamaAttention']
