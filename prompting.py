import json
from typing import Dict, List, Optional

from task_schema import CrawlTaskSpec, INPUT_FIELD_LABELS, OutputFieldSpec


def _sample_value(field_spec: OutputFieldSpec) -> str:
    if field_spec.field_type == "number":
        return "0.0"
    if field_spec.field_type == "boolean":
        return "false"
    if field_spec.field_type == "json":
        return "{}"
    return '""'


def build_json_contract_text(field_specs: List[OutputFieldSpec]) -> str:
    lines = ["{"]
    for idx, spec in enumerate(field_specs):
        comma = "," if idx < len(field_specs) - 1 else ""
        lines.append(f'  "{spec.key}": {_sample_value(spec)}{comma}')
    lines.append("}")
    return "\n".join(lines)


def build_task_prompt(
    payload: Dict[str, str],
    task_spec: CrawlTaskSpec,
    runtime_context: Optional[Dict[str, object]] = None,
) -> str:
    input_lines: List[str] = []
    for key, label in INPUT_FIELD_LABELS.items():
        value = str(payload.get(key, "") or "").strip()
        if value:
            input_lines.append(f"- {label}: {value}")

    if not input_lines:
        input_lines.append("- 입력 없음")

    contract_text = build_json_contract_text(task_spec.response_field_specs)
    required_keys = [spec.key for spec in task_spec.response_field_specs if spec.required]
    required_line = ", ".join(required_keys) if required_keys else "없음"

    context = {
        "json_contract": contract_text,
        "required_keys": required_line,
        "min_source_urls": str(task_spec.min_source_urls),
        "input_block": "\n".join(input_lines),
        "input_json": json.dumps(payload, ensure_ascii=False, indent=2),
        "response_keys": ", ".join(task_spec.response_keys),
        "output_columns": ", ".join(task_spec.selected_output_columns),
        "runtime_context_json": "{}",
    }

    runtime_context = runtime_context or {}
    if runtime_context:
        context["runtime_context_json"] = json.dumps(
            runtime_context,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        for key, value in runtime_context.items():
            context[str(key)] = str(value)

    prompt = task_spec.prompt_template
    for key, value in context.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)
    return prompt.strip()


def serialize_field_confidence(value) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return "{}"
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return "{}"
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        return "{}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "{}"
