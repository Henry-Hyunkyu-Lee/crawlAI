import pytest

from task_schema import build_task_spec
from validation import ResultValidationError, normalize_and_validate_result


def _build_spec(min_urls=1):
    return build_task_spec(
        selected_output_columns=[
            "email",
            "confidence_score",
            "evidence_urls",
            "field_confidence_json",
        ],
        custom_specs=[],
        prompt_template="",
        min_source_urls=min_urls,
        validation_level="strong",
    )


def test_normalize_and_validate_result_uses_fallback_sources():
    spec = _build_spec(min_urls=1)
    payload = {
        "email": "test@example.com",
        "confidence_score": 0.8,
        "field_confidence_json": {"email": 0.8, "confidence_score": 0.8},
    }
    normalized, flags = normalize_and_validate_result(
        payload,
        task_spec=spec,
        fallback_source_urls=["https://example.com/a"],
    )
    assert normalized["evidence_urls"] == "https://example.com/a"
    assert flags == []


def test_normalize_and_validate_result_raises_when_source_count_insufficient():
    spec = _build_spec(min_urls=2)
    payload = {
        "email": "test@example.com",
        "confidence_score": 0.7,
        "evidence_urls": "https://example.com/a",
        "field_confidence_json": {"email": 0.7, "confidence_score": 0.7},
    }
    with pytest.raises(ResultValidationError):
        normalize_and_validate_result(payload, task_spec=spec, fallback_source_urls=[])


def test_normalize_and_validate_result_raises_on_confidence_range():
    spec = _build_spec(min_urls=0)
    payload = {
        "email": "test@example.com",
        "confidence_score": 1.8,
    }
    with pytest.raises(ResultValidationError):
        normalize_and_validate_result(payload, task_spec=spec, fallback_source_urls=[])
