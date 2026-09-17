"""
Engine entry point for Parallel Constrained Decoding.

This local port targets Apple Silicon only, so the MLX backend is imported
directly (the upstream repo also shipped a PyTorch/CUDA fallback for
Hugging Face Spaces, which is intentionally not vendored here).
"""

from core.engine_mlx import (
    MODEL_ID,
    get_engine,
    run_parallel_generation,
    run_naive_generation,
    stream_naive_generation,
    run_rlcd_generation,
)

USE_MLX = True

__all__ = [
    "MODEL_ID",
    "get_engine",
    "run_parallel_generation",
    "run_naive_generation",
    "stream_naive_generation",
    "run_rlcd_generation",
    "USE_MLX",
]
