"""
Parallel Constrained Structured Generation Engine.
"""

from core.schema import StructuredSchema, FieldDefinition
from core.engine import (
    get_engine,
    run_parallel_generation,
    run_naive_generation,
    stream_naive_generation,
    # Backward compatibility
    run_rlcd_generation,
)
from core.prompt_builder import (
    build_naive_json_prompt,
    build_parallel_field_prompts,
)

__all__ = [
    "StructuredSchema",
    "FieldDefinition",
    "get_engine",
    "run_parallel_generation",
    "run_naive_generation",
    "stream_naive_generation",
    "build_naive_json_prompt",
    "build_parallel_field_prompts",
    "run_rlcd_generation",
]
