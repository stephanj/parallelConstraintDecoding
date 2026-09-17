"""
Minimal quickstart: define a schema, run parallel constrained decoding on a
context string, and print the extracted JSON with per-field confidence.

    .venv/bin/python example.py
"""

import json

from core.schema import StructuredSchema
from core.engine import run_parallel_generation, run_naive_generation

schema = StructuredSchema({
    "priority": {
        "type": "enum",
        "choices": ["P0_CRITICAL", "P1_HIGH", "P2_NORMAL", "P3_LOW"],
        "description": "Urgency tier based on customer business impact",
    },
    "requires_escalation": {
        "type": "boolean",
        "description": "Whether an on-call engineer must be notified immediately",
    },
    "department": {
        "type": "enum",
        "choices": ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"],
        "description": "Target handling department",
    },
})

context = """
Incident Report: Production database db-primary-01 CPU at 100%.
Payment gateway failing for 40% of checkout requests.
Tier 1 Enterprise customer affected: Acme Global.
"""

if __name__ == "__main__":
    result = run_parallel_generation(context, schema)
    print(f"\nParallel constrained: {result['elapsed_ms']} ms "
          f"(prefill {result['prefill_ms']} ms, suffix eval {result['suffix_eval_ms']} ms, "
          f"forward passes: {result['sequential_forward_passes']})")
    print(json.dumps(result["parsed_json"], indent=2))
    print("\nTop choices per field:")
    for name, tel in result["field_telemetry"].items():
        ranked = ", ".join(f"{c['choice']}={c['probability']}" for c in tel["top_choices"])
        print(f"  {name}: {ranked}")

    naive = run_naive_generation(context, schema)
    print(f"\nAutoregressive baseline: {naive['elapsed_ms']} ms, "
          f"{naive['total_tokens']} tokens, schema_match={naive['schema_match']}")
    print(naive["raw_text"])
