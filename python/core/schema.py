"""
Schema definitions, validation, and sub-vocabulary token mapping for parallel constrained decisions.
Supports booleans and categorical enums with cardinality up to 255.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class FieldDefinition:
    def __init__(self, name: str, field_type: str, description: str, choices: Optional[List[str]] = None):
        self.name = name
        self.field_type = field_type.lower()
        self.description = description
        
        if self.field_type == "boolean":
            self.choices = ["true", "false"]
        elif self.field_type in ("enum", "choice", "selection"):
            if not choices or len(choices) == 0:
                raise ValueError(f"Field '{name}' of type enum must have choices defined.")
            if len(choices) > 255:
                raise ValueError(f"Field '{name}' exceeds maximum cardinality of 255 choices (got {len(choices)}).")
            self.choices = choices
        else:
            raise ValueError(f"Unsupported field type '{field_type}'. Supported types: 'boolean' and 'enum'.")

        self.cached_candidate_token_ids: Optional[List[List[int]]] = None

    @property
    def cardinality(self) -> int:
        return len(self.choices)

    def compile_candidate_tokens(self, tokenizer):
        """Pre-indexes and caches candidate token IDs so inference runs in microseconds."""
        if self.cached_candidate_token_ids is not None:
            return self.cached_candidate_token_ids
            
        candidate_tokens_per_choice = []
        if self.field_type == "boolean":
            true_variants = ['true', ' true', 'True', ' True', 'TRUE', 'yes', ' yes']
            true_ids = []
            for v in true_variants:
                toks = tokenizer.encode(v, add_special_tokens=False)
                if toks:
                    true_ids.append(toks[0])
            candidate_tokens_per_choice.append(list(set(true_ids)))
            
            false_variants = ['false', ' false', 'False', ' False', 'FALSE', 'no', ' no']
            false_ids = []
            for v in false_variants:
                toks = tokenizer.encode(v, add_special_tokens=False)
                if toks:
                    false_ids.append(toks[0])
            candidate_tokens_per_choice.append(list(set(false_ids)))
        else:
            for choice in self.choices:
                c_clean = str(choice).strip()
                variants = [' ' + c_clean, c_clean]
                ids = []
                for v in variants:
                    toks = tokenizer.encode(v, add_special_tokens=False)
                    if toks:
                        ids.append(toks[0])
                candidate_tokens_per_choice.append(list(set(ids)))
                
        self.cached_candidate_token_ids = candidate_tokens_per_choice
        return self.cached_candidate_token_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.field_type,
            "description": self.description,
            "choices": self.choices,
            "cardinality": self.cardinality,
        }


class StructuredSchema:
    def __init__(self, schema_dict: Dict[str, Any], tokenizer=None):
        self.fields: Dict[str, FieldDefinition] = {}
        for field_name, spec in schema_dict.items():
            field_type = spec.get("type", "enum")
            description = spec.get("description", "")
            choices = spec.get("choices", None)
            fdef = FieldDefinition(
                name=field_name,
                field_type=field_type,
                description=description,
                choices=choices
            )
            if tokenizer is not None:
                fdef.compile_candidate_tokens(tokenizer)
            self.fields[field_name] = fdef

    def compile_all_tokens(self, tokenizer):
        for fdef in self.fields.values():
            fdef.compile_candidate_tokens(tokenizer)

    def get_field_names(self) -> List[str]:
        return list(self.fields.keys())

    def __getitem__(self, key: str) -> FieldDefinition:
        return self.fields[key]

    def __len__(self) -> int:
        return len(self.fields)

    def to_json_schema_prompt_str(self) -> str:
        """Returns a clean TypeScript/JSON schema representation for naive LLM prompting."""
        lines = ["{"]
        for name, field in self.fields.items():
            if field.field_type == "boolean":
                lines.append(f'  "{name}": boolean, // {field.description}')
            else:
                choices_limit = 20 if len(field.choices) > 50 else len(field.choices)
                choices_str = " | ".join(f'"{c}"' for c in field.choices[:choices_limit])
                if len(field.choices) > choices_limit:
                    choices_str += f" | ... ({len(field.choices)} total options)"
                lines.append(f'  "{name}": {choices_str}, // {field.description}')
        lines.append("}")
        return "\n".join(lines)

    def to_parallel_schema_str(self) -> str:
        """
        Returns a compact description catalog for the parallel prefill.
        Each line lists the field, its description, and the allowed choices, so the
        model knows the candidate values it is being scored against. Choice lists
        are truncated with the same rule as the naive baseline prompt so both
        engines see identical information.
        """
        lines = []
        for name, field in self.fields.items():
            desc = field.description.split('\n')[0].strip()
            if field.field_type == "boolean":
                choices_str = "true | false"
            else:
                choices_limit = 20 if len(field.choices) > 50 else len(field.choices)
                choices_str = " | ".join(field.choices[:choices_limit])
                if len(field.choices) > choices_limit:
                    choices_str += f" | ... ({len(field.choices)} total options)"
            lines.append(f'  "{name}": {desc} [{choices_str}]')
        return "\n".join(lines)

    to_rlcd_schema_str = to_parallel_schema_str

    def compile_parallel_metadata(self, tokenizer):
        """Pre-indexes and caches compact suffixes, token candidate IDs, and common prefixes."""
        if hasattr(self, "_parallel_metadata") and self._parallel_metadata is not None:
            return self._parallel_metadata
            
        import os
        field_items = list(self.fields.items())
        suffix_tok_lists = []
        suffix_lengths = []
        cands_per_field = []
        prefixes = []
        has_collisions = []
        
        for fname, fdef in field_items:
            if fdef.field_type == "boolean":
                suffix = f'  "{fname}": '
                cands = [
                    tokenizer.encode("true", add_special_tokens=False)[0],
                    tokenizer.encode("false", add_special_tokens=False)[0]
                ]
                prefix = ""
            else:
                prefix = os.path.commonprefix(fdef.choices)
                suffix = f'  "{fname}": "{prefix}'
                cands = []
                for c in fdef.choices:
                    rem = c[len(prefix):]
                    c_toks = tokenizer.encode(rem, add_special_tokens=False)
                    cands.append(c_toks[0] if c_toks else tokenizer.encode('"', add_special_tokens=False)[0])
            toks = tokenizer.encode(suffix, add_special_tokens=False)
            suffix_tok_lists.append(toks)
            suffix_lengths.append(len(toks))
            cands_per_field.append(cands)
            prefixes.append(prefix)
            has_collisions.append(len(set(cands)) < len(cands))
            
        max_s_len = max(suffix_lengths)
        pad_id = tokenizer.pad_token_id or 0
        padded = [s + [pad_id] * (max_s_len - len(s)) for s in suffix_tok_lists]
        try:
            import mlx.core as mx
            suffixes_batch = mx.array(padded, dtype=mx.int32)
        except Exception:
            suffixes_batch = np.array(padded, dtype=np.int32)
        
        self._parallel_metadata = {
            "field_items": field_items,
            "suffix_lengths": suffix_lengths,
            "cands_per_field": cands_per_field,
            "prefixes": prefixes,
            "has_collisions": has_collisions,
            "suffixes_batch": suffixes_batch
        }
        return self._parallel_metadata

    compile_rlcd_metadata = compile_parallel_metadata


def map_candidate_tokens(tokenizer, choices: List[str], is_boolean: bool = False) -> List[List[int]]:
    """Helper fallback when field definition is not pre-compiled."""
    f = FieldDefinition("tmp", "boolean" if is_boolean else "enum", "", choices if not is_boolean else None)
    return f.compile_candidate_tokens(tokenizer)


def extract_calibrated_probabilities(
    next_token_logits: np.ndarray,
    candidate_token_ids_list: List[List[int]],
    temperature: float = 1.0
) -> Tuple[int, float, List[float]]:
    """
    Takes the logits at the decision token position and computes exact
    calibrated probabilities across only the constrained candidate choices (K <= 255).
    """
    choice_scores = []
    for token_ids in candidate_token_ids_list:
        if not token_ids:
            choice_scores.append(-1e9)
            continue
        score = max(float(next_token_logits[tid]) for tid in token_ids)
        choice_scores.append(score)
        
    scores = np.array(choice_scores, dtype=np.float32) / max(temperature, 1e-4)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    probs = exp_scores / (np.sum(exp_scores) + 1e-12)
    
    winner_idx = int(np.argmax(probs))
    winner_prob = float(probs[winner_idx])
    
    return winner_idx, winner_prob, probs.tolist()
