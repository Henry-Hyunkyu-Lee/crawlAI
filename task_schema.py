import re
from dataclasses import dataclass
from typing import Dict, List

LABEL_NAME = "\uc131\uba85"
LABEL_COMPANY = "\ud68c\uc0ac\uba85"
LABEL_DEPARTMENT = "\ubd80\uc11c\uba85"
LABEL_JOB_TITLE = "\uc9c1\ucc45\uba85"
LABEL_PROJECT = "\uacfc\uc81c\uba85"

INPUT_FIELD_LABELS = {
    "name": LABEL_NAME,
    "company": LABEL_COMPANY,
    "department": LABEL_DEPARTMENT,
    "job_title": LABEL_JOB_TITLE,
    "project": LABEL_PROJECT,
}

STANDARD_OUTPUT_COLUMNS = [
    LABEL_NAME,
    LABEL_COMPANY,
    LABEL_DEPARTMENT,
    LABEL_JOB_TITLE,
    LABEL_PROJECT,
    "email",
    "confidence_score",
    "evidence_urls",
    "evidence_snippets",
    "field_confidence_json",
    "status",
    "error_code",
    "retry_count",
    "validation_flags",
]

REQUIRED_OUTPUT_COLUMNS = ["email", "confidence_score"]
SYSTEM_OUTPUT_COLUMNS = ["status", "error_code", "retry_count", "validation_flags"]

OUTPUT_COLUMN_TO_RESULT_KEY = {
    LABEL_NAME: "name",
    LABEL_COMPANY: "company",
    LABEL_DEPARTMENT: "department",
    LABEL_JOB_TITLE: "job_title",
    LABEL_PROJECT: "project",
    "email": "email",
    "confidence_score": "confidence_score",
    "evidence_urls": "evidence_urls",
    "evidence_snippets": "evidence_snippets",
    "field_confidence_json": "field_confidence_json",
}

RESULT_KEY_TO_OUTPUT_COLUMN = {v: k for k, v in OUTPUT_COLUMN_TO_RESULT_KEY.items()}


@dataclass(frozen=True)
class OutputFieldSpec:
    key: str
    field_type: str = "string"
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class CrawlTaskSpec:
    selected_output_columns: List[str]
    response_field_specs: List[OutputFieldSpec]
    response_keys: List[str]
    custom_field_keys: List[str]
    prompt_template: str
    min_source_urls: int
    validation_level: str = "medium"


@dataclass(frozen=True)
class InputPolicySpec:
    policy_id: str
    label: str
    description: str
    required_keys: List[str]
    recommended_keys: List[str]


BUILTIN_FIELD_SPECS: Dict[str, OutputFieldSpec] = {
    "name": OutputFieldSpec("name", "string", False, "\ub2f4\ub2f9\uc790 \uc131\uba85"),
    "company": OutputFieldSpec("company", "string", False, "\ud68c\uc0ac\uba85"),
    "department": OutputFieldSpec("department", "string", False, "\ubd80\uc11c\uba85"),
    "job_title": OutputFieldSpec("job_title", "string", False, "\uc9c1\ucc45\uba85"),
    "project": OutputFieldSpec("project", "string", False, "\uacfc\uc81c\uba85"),
    "email": OutputFieldSpec("email", "string", True, "\uc774\uba54\uc77c \uc8fc\uc18c"),
    "confidence_score": OutputFieldSpec("confidence_score", "number", True, "\uc2e0\ub8b0\ub3c4(0~1)"),
    "evidence_urls": OutputFieldSpec("evidence_urls", "string", False, "\ucd9c\ucc98 URL(\uc138\ubbf8\ucf5c\ub860 \uad6c\ubd84)"),
    "evidence_snippets": OutputFieldSpec("evidence_snippets", "string", False, "\uadfc\uac70 \uc2a4\ub2c8\ud3ab"),
    "field_confidence_json": OutputFieldSpec(
        "field_confidence_json", "json", False, "\ud544\ub4dc\ubcc4 confidence JSON"
    ),
}


INPUT_POLICY_PRESETS: Dict[str, InputPolicySpec] = {
    "company_required": InputPolicySpec(
        policy_id="company_required",
        label="\ud68c\uc0ac\uba85 \ud544\uc218 (\uae30\ubcf8)",
        description="\ud68c\uc0ac\uba85 \uceec\ub7fc \ub9e4\ud551\uc740 \ubc18\ub4dc\uc2dc \ud544\uc694\ud569\ub2c8\ub2e4.",
        required_keys=["company"],
        recommended_keys=["name"],
    ),
    "contact_focused": InputPolicySpec(
        policy_id="contact_focused",
        label="\ub2f4\ub2f9\uc790 \ud0d0\uc0c9 \uc911\uc2ec",
        description="\ud68c\uc0ac\uba85\uc740 \ud544\uc218, \uc131\uba85/\ubd80\uc11c/\uc9c1\ucc45 \ub9e4\ud551\uc744 \uad8c\uc7a5\ud569\ub2c8\ub2e4.",
        required_keys=["company"],
        recommended_keys=["name", "department", "job_title"],
    ),
    "flexible": InputPolicySpec(
        policy_id="flexible",
        label="\uc790\uc720 \uc785\ub825",
        description="\ud544\uc218 \ub9e4\ud551 \uc5c6\uc774 \uc790\uc720\ub86d\uac8c \uc2e4\ud589\ud569\ub2c8\ub2e4.",
        required_keys=[],
        recommended_keys=["company"],
    ),
}


DEFAULT_PROMPT_TEMPLATE = """
[\uc5ed\ud560]
\ub2f9\uc2e0\uc740 \ucf5c\ub4dc \uba54\uc77c \ubc1c\uc1a1\uc6a9 \uc774\uba54\uc77c \uc8fc\uc18c \uc218\uc9d1 \ubcf4\uc870 \uc5d0\uc774\uc804\ud2b8\ub2e4.

[\ubaa9\ud45c]
\uc785\ub825 \ub370\uc774\ud130\uc640 \uc6f9 \uac80\uc0c9 \uacb0\uacfc\ub97c \uc0ac\uc6a9\ud574 \uc9c0\uc815\ub41c JSON \uc2a4\ud0a4\ub9c8\ub97c \uc815\ud655\ud788 \ucc44\uc6b4\ub2e4.

[\uaddc\uce59]
1. \ubc18\ub4dc\uc2dc JSON \uac1d\uccb4 \ud558\ub098\ub9cc \ubc18\ud658\ud55c\ub2e4. (\ucf54\ub4dc\ube14\ub85d \uae08\uc9c0)
2. \ud655\uc2e4\ud558\uc9c0 \uc54a\uc740 \uc815\ubcf4\ub294 \ucd94\uce21\ud558\uc9c0 \ub9d0\uace0 \ube48 \ubb38\uc790\uc5f4 \ub610\ub294 \ub0ae\uc740 confidence\ub85c \ucc98\ub9ac\ud55c\ub2e4.
3. evidence_urls\uc5d0\ub294 \uc2e4\uc81c \ucc38\uace0\ud55c URL\uc744 \uc138\ubbf8\ucf5c\ub860(;)\uc73c\ub85c \uad6c\ubd84\ud574 \ub123\ub294\ub2e4.
4. field_confidence_json\uc5d0\ub294 \uc8fc\uc694 \ud544\ub4dc\ubcc4 0~1 \uc810\uc218\ub97c \ub123\ub294\ub2e4.
5. \uc544\ub798 \uc2a4\ud0a4\ub9c8\uc5d0 \uc5c6\ub294 \ud0a4\ub294 \uc808\ub300 \ubc18\ud658\ud558\uc9c0 \uc54a\ub294\ub2e4.

[\ud544\uc218 \ud0a4]
{{required_keys}}

[\ucd5c\uc18c \ucd9c\ucc98 URL]
{{min_source_urls}}

[\ubc18\ud658 JSON \uc2a4\ud0a4\ub9c8]
{{json_contract}}

[\uc785\ub825(JSON)]
{{input_json}}

[\uc785\ub825(\uac00\ub3c5\ud615)]
{{input_block}}

[\uc2e4\ud589 \uba54\ud0c0\ub370\uc774\ud130]
- provider: {{provider}}
- model_name: {{model_name}}
- row_index: {{row_index}}
- row_position: {{row_position}}
- total_pending: {{total_pending}}
- run_start: {{run_start}}
- run_end: {{run_end}}
- output_path: {{output_path}}
- timestamp_utc: {{timestamp_utc}}
- runtime_context_json:
{{runtime_context_json}}
""".strip()


def get_default_prompt_template() -> str:
    return DEFAULT_PROMPT_TEMPLATE


def get_input_policy_options() -> List[InputPolicySpec]:
    order = ["company_required", "contact_focused", "flexible"]
    return [INPUT_POLICY_PRESETS[k] for k in order if k in INPUT_POLICY_PRESETS]


def get_input_policy(policy_id: str) -> InputPolicySpec:
    return INPUT_POLICY_PRESETS.get(policy_id) or INPUT_POLICY_PRESETS["company_required"]


def evaluate_input_policy(mapping: Dict[str, str], policy_id: str):
    policy = get_input_policy(policy_id)
    missing_required = [k for k in policy.required_keys if not mapping.get(k)]
    missing_recommended = [k for k in policy.recommended_keys if not mapping.get(k)]
    return policy, missing_required, missing_recommended


def parse_custom_fields_text(raw_text: str) -> List[OutputFieldSpec]:
    specs: List[OutputFieldSpec] = []
    if not raw_text or not raw_text.strip():
        return specs

    allowed_types = {"string", "number", "boolean", "json"}

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        row = line.strip()
        if not row or row.startswith("#"):
            continue

        parts = [p.strip() for p in row.split(",")]
        key = parts[0] if parts else ""
        if not key:
            continue

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
            raise ValueError(f"\ucee4\uc2a4\ud140 \ud544\ub4dc {line_no}\ud589 key \ud615\uc2dd\uc774 \uc62c\ubc14\ub974\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4: {key}")

        field_type = parts[1].lower() if len(parts) >= 2 and parts[1] else "string"
        if field_type not in allowed_types:
            raise ValueError(
                f"\ucee4\uc2a4\ud140 \ud544\ub4dc {line_no}\ud589 type\uc740 {sorted(allowed_types)} \uc911 \ud558\ub098\uc5ec\uc57c \ud569\ub2c8\ub2e4."
            )

        required_raw = parts[2].lower() if len(parts) >= 3 else "false"
        required = required_raw in {"1", "true", "y", "yes", "required"}
        description = parts[3] if len(parts) >= 4 else ""

        if key in BUILTIN_FIELD_SPECS:
            raise ValueError(f"\ucee4\uc2a4\ud140 \ud544\ub4dc {line_no}\ud589 key\uac00 \uae30\ubcf8 \ud0a4\uc640 \ucda9\ub3cc\ud569\ub2c8\ub2e4: {key}")

        specs.append(OutputFieldSpec(key=key, field_type=field_type, required=required, description=description))

    unique = {}
    for spec in specs:
        unique[spec.key] = spec
    return list(unique.values())


def enforce_required_output_columns(columns: List[str]) -> List[str]:
    selected = list(dict.fromkeys(columns or []))
    for required in REQUIRED_OUTPUT_COLUMNS:
        if required not in selected:
            selected.append(required)

    ordered = [c for c in STANDARD_OUTPUT_COLUMNS if c in selected]
    extras = [c for c in selected if c not in STANDARD_OUTPUT_COLUMNS]
    return ordered + extras


def build_task_spec(
    selected_output_columns: List[str],
    custom_specs: List[OutputFieldSpec],
    prompt_template: str,
    min_source_urls: int,
    validation_level: str = "medium",
) -> CrawlTaskSpec:
    final_output_columns = enforce_required_output_columns(selected_output_columns)
    custom_by_key = {spec.key: spec for spec in custom_specs}

    response_field_specs: List[OutputFieldSpec] = []
    response_keys: List[str] = []
    custom_field_keys: List[str] = []

    for column in final_output_columns:
        if column in SYSTEM_OUTPUT_COLUMNS:
            continue

        result_key = OUTPUT_COLUMN_TO_RESULT_KEY.get(column)
        if result_key:
            spec = BUILTIN_FIELD_SPECS[result_key]
        else:
            result_key = column
            spec = custom_by_key.get(column) or OutputFieldSpec(
                key=result_key,
                field_type="string",
                required=False,
            )
            custom_field_keys.append(result_key)

        if result_key in response_keys:
            continue
        response_keys.append(result_key)
        response_field_specs.append(spec)

    final_prompt = (prompt_template or "").strip() or get_default_prompt_template()
    return CrawlTaskSpec(
        selected_output_columns=final_output_columns,
        response_field_specs=response_field_specs,
        response_keys=response_keys,
        custom_field_keys=custom_field_keys,
        prompt_template=final_prompt,
        min_source_urls=max(0, int(min_source_urls)),
        validation_level=validation_level,
    )
