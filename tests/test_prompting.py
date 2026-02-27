from prompting import build_task_prompt
from task_schema import build_task_spec


def _spec_with_prompt(prompt_template: str):
    return build_task_spec(
        selected_output_columns=["email", "confidence_score"],
        custom_specs=[],
        prompt_template=prompt_template,
        min_source_urls=2,
        validation_level="strong",
    )


def test_build_task_prompt_replaces_runtime_placeholders():
    spec = _spec_with_prompt(
        "provider={{provider}} model={{model_name}} row={{row_index}}"
        " total={{total_pending}} start={{run_start}} end={{run_end}}"
        " out={{output_path}} ts={{timestamp_utc}}"
    )
    prompt = build_task_prompt(
        {"company": "OpenAI"},
        spec,
        runtime_context={
            "provider": "openai",
            "model_name": "gpt-5-nano",
            "row_index": 12,
            "total_pending": 100,
            "run_start": 0,
            "run_end": 100,
            "output_path": "results.csv",
            "timestamp_utc": "2026-02-27T00:00:00+00:00",
        },
    )
    assert "provider=openai" in prompt
    assert "model=gpt-5-nano" in prompt
    assert "row=12" in prompt
    assert "total=100" in prompt
    assert "out=results.csv" in prompt


def test_build_task_prompt_includes_runtime_context_json():
    spec = _spec_with_prompt("meta={{runtime_context_json}}")
    prompt = build_task_prompt(
        {"company": "OpenAI"},
        spec,
        runtime_context={"provider": "openai", "row_index": 7},
    )
    assert '"provider": "openai"' in prompt
    assert '"row_index": 7' in prompt
