"""
Prompt construction utilities for Autoregressive JSON Generation
vs. Parallel Constrained Decision Batches.
"""

from typing import Dict, Any, List, Tuple
from core.schema import StructuredSchema, FieldDefinition


def build_naive_json_prompt(context: str, schema: StructuredSchema) -> str:
    """
    Builds the baseline prompt instructing the model to generate a full JSON document.
    """
    schema_prompt = schema.to_json_schema_prompt_str()
    prompt = (
        f"<|im_start|>system\n"
        f"You are a precise data extraction system. You must output ONLY a valid, beautifully formatted, indented JSON object with newlines and 2-space indentation matching the schema below. Do not output a single-line string. Do not include markdown tags.\n\n"
        f"JSON Schema:\n{schema_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Analyze the following context and generate the required formatted JSON object:\n\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n  "
    )
    return prompt


def build_parallel_field_prompts(context: str, schema: StructuredSchema) -> List[Tuple[str, FieldDefinition, str]]:
    """
    Builds discrete single-decision prompts for each field in the schema.
    Returns a list of (field_name, field_def, prompt_text).
    """
    prompts = []
    for field_name, field_def in schema.fields.items():
        if field_def.field_type == "boolean":
            options_text = "true, false"
        else:
            if len(field_def.choices) <= 20:
                options_text = ", ".join(field_def.choices)
            else:
                sample = ", ".join(field_def.choices[:8])
                options_text = f"{sample}, ... [{len(field_def.choices)} total options]"

        prompt = (
            f"<|im_start|>system\n"
            f"You are a calibrated decision engine. Select the single most accurate option based on evidence.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{context}\n\n"
            f"Field: {field_name}\n"
            f"Description: {field_def.description}\n"
            f"Allowed choices: {options_text}\n"
            f"Exact choice:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append((field_name, field_def, prompt))
        
    return prompts


# Backward compatibility alias
build_rlcd_field_prompts = build_parallel_field_prompts
