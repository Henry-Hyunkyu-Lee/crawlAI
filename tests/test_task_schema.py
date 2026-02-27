import pytest

from task_schema import (
    LABEL_COMPANY,
    build_task_spec,
    evaluate_input_policy,
    get_input_policy_options,
    parse_custom_fields_text,
)


def test_parse_custom_fields_text_parses_lines():
    raw = "linkedin_url,string,false,??\nscore_delta,number,true,??"
    specs = parse_custom_fields_text(raw)
    assert len(specs) == 2
    assert specs[0].key == "linkedin_url"
    assert specs[1].field_type == "number"
    assert specs[1].required is True


def test_parse_custom_fields_text_invalid_key_raises():
    with pytest.raises(ValueError):
        parse_custom_fields_text("1bad,string,false")


def test_build_task_spec_enforces_required_columns_and_custom_fields():
    custom_specs = parse_custom_fields_text("linkedin_url,string,false,??")
    spec = build_task_spec(
        selected_output_columns=[LABEL_COMPANY, "linkedin_url"],
        custom_specs=custom_specs,
        prompt_template="",
        min_source_urls=1,
        validation_level="strong",
    )

    assert "email" in spec.selected_output_columns
    assert "confidence_score" in spec.selected_output_columns
    assert "linkedin_url" in spec.response_keys
    assert "company" in spec.response_keys


def test_build_task_spec_uses_default_prompt_when_empty():
    spec = build_task_spec(
        selected_output_columns=["email"],
        custom_specs=[],
        prompt_template="",
        min_source_urls=1,
        validation_level="strong",
    )
    assert "{{json_contract}}" in spec.prompt_template
    assert "{{provider}}" in spec.prompt_template


def test_input_policy_options_have_default():
    options = get_input_policy_options()
    ids = [opt.policy_id for opt in options]
    assert "company_required" in ids


def test_evaluate_input_policy_detects_missing_required():
    _, missing_required, _ = evaluate_input_policy(
        mapping={"name": "name_col"},
        policy_id="company_required",
    )
    assert missing_required == ["company"]
