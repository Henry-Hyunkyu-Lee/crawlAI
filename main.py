import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from updater import ReleaseInfo, check_for_update, launch_update_script, read_update_status

DEFAULT_OUTPUT = "results.csv"
DEFAULT_HEAD_ROWS = 30
DEFAULT_AUTOSAVE_EVERY = 20
RESULT_COMPLETE_STATES = {"success", "error"}

SUPPORTED_PROVIDERS = ["openai", "gemini"]
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

INPUT_FIELD_SPECS = [
    {"key": "name", "label": "성명", "aliases": ["성명", "이름", "담당자", "name", "person"]},
    {
        "key": "company",
        "label": "회사명",
        "aliases": ["회사명", "기업명", "기관명", "소속", "organization", "company", "org"],
    },
    {
        "key": "department",
        "label": "부서명",
        "aliases": ["부서명", "부서", "소속부서", "department", "dept"],
    },
    {
        "key": "job_title",
        "label": "직책명",
        "aliases": ["직책명", "직급", "직위", "position", "title", "role"],
    },
    {
        "key": "project",
        "label": "과제명",
        "aliases": ["과제명", "프로젝트명", "연구과제", "project", "task"],
    },
]

STANDARD_OUTPUT_COLUMNS = [
    "성명",
    "회사명",
    "부서명",
    "직책명",
    "과제명",
    "email",
    "confidence_score",
    "status",
    "error_code",
]
REQUIRED_OUTPUT_COLUMNS = ["email", "confidence_score"]
UPDATE_PROGRESS_BY_STEP = {
    "queued": 0.05,
    "start": 0.1,
    "stopping_process": 0.18,
    "downloading_release": 0.32,
    "extracting_release": 0.45,
    "applying_files": 0.62,
    "removing_stale_files": 0.76,
    "syncing_dependencies": 0.88,
    "restarting_app": 0.95,
    "success": 1.0,
    "failed": 1.0,
}

INPUT_KEY_TO_OUTPUT_LABEL = {
    "name": "성명",
    "company": "회사명",
    "department": "부서명",
    "job_title": "직책명",
    "project": "과제명",
}

OUTPUT_COLUMN_TO_RESULT_KEY = {
    "성명": "name",
    "회사명": "company",
    "부서명": "department",
    "직책명": "job_title",
    "과제명": "project",
    "email": "email",
    "confidence_score": "confidence_score",
}

PROMPT_INPUT_KEYS = ["name", "company", "department", "job_title", "project"]


class StructuredSearchResult(BaseModel):
    name: str = Field(default="")
    company: str = Field(default="")
    department: str = Field(default="")
    job_title: str = Field(default="")
    project: str = Field(default="")
    email: str = Field(default="")
    confidence_score: float = Field(default=0.0)


def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def normalize_header(text):
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(text or "").lower())


def detect_column_by_alias(columns, aliases):
    normalized_aliases = {normalize_header(alias) for alias in aliases}
    for col in columns:
        if normalize_header(col) in normalized_aliases:
            return col
    return None


def normalize_structured_values(result: StructuredSearchResult):
    data = result.model_dump()
    normalized = {}
    for key, value in data.items():
        if key == "confidence_score":
            normalized[key] = float(value)
        else:
            normalized[key] = str(value or "").strip()

    score = normalized["confidence_score"]
    if not 0 <= score <= 1:
        raise ValueError("confidence_score는 0~1 범위여야 합니다.")
    return normalized


def ensure_result_columns(df):
    df_out = df.copy()

    if "email" not in df_out.columns and "Email" in df_out.columns:
        df_out["email"] = df_out["Email"]
    if "confidence_score" not in df_out.columns and "conf" in df_out.columns:
        df_out["confidence_score"] = df_out["conf"]

    for col in ["성명", "회사명", "부서명", "직책명", "과제명", "email", "status", "error_code"]:
        if col not in df_out.columns:
            df_out[col] = ""

    if "confidence_score" not in df_out.columns:
        df_out["confidence_score"] = np.nan

    return df_out


def preprocess_df(df, dedupe_cols):
    df_out = df.copy()
    if dedupe_cols:
        dedupe_cols = [c for c in dedupe_cols if c in df_out.columns]
        if dedupe_cols:
            df_out = df_out.drop_duplicates(subset=dedupe_cols)
    return df_out


def strip_code_fences(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_structured_result_text(raw_text):
    cleaned = strip_code_fences(raw_text)
    if not cleaned:
        raise ValueError("모델 응답이 비어 있습니다.")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("모델 응답에서 JSON 객체를 찾을 수 없습니다.")
        payload = json.loads(cleaned[start : end + 1])

    result = StructuredSearchResult.model_validate(payload)
    return normalize_structured_values(result)


def build_response_contract(selected_output_columns):
    response_keys = []
    for col in selected_output_columns:
        key = OUTPUT_COLUMN_TO_RESULT_KEY.get(col)
        if key and key not in response_keys:
            response_keys.append(key)
    return response_keys


def enforce_required_output_columns(columns):
    selected = set(columns or [])
    selected.update(REQUIRED_OUTPUT_COLUMNS)
    # Keep a stable, predictable order for prompt/result handling.
    return [c for c in STANDARD_OUTPUT_COLUMNS if c in selected]


def build_json_contract_text(response_keys):
    lines = ["{"]
    for idx, key in enumerate(response_keys):
        if key == "confidence_score":
            value = "0.0"
        else:
            value = '""'
        comma = "," if idx < len(response_keys) - 1 else ""
        lines.append(f'  "{key}": {value}{comma}')
    lines.append("}")
    return "\n".join(lines)


def build_task_prompt(payload, selected_output_columns, user_tweak: str = ""):
    response_keys = build_response_contract(selected_output_columns)
    prompt_input_keys = [k for k in PROMPT_INPUT_KEYS if INPUT_KEY_TO_OUTPUT_LABEL[k] in selected_output_columns]

    input_lines = []
    for key in prompt_input_keys:
        label = INPUT_KEY_TO_OUTPUT_LABEL[key]
        input_lines.append(f"- {label}: {payload[key]}")

    if not input_lines:
        input_lines.append("- 입력 없음")

    contract_text = build_json_contract_text(response_keys)
    base_prompt = (
        "검색을 통해서 아래 정보를 완성하라.\n"
        "아래 입력 항목만 활용하고, 입력에 없는 항목은 웹 검색으로 보완하라.\n"
        "명시되지 않은 컬럼은 절대 반환하지 마라.\n"
        "반드시 JSON 객체만 반환하라. 코드블록은 금지한다.\n"
        f"{contract_text}\n"
        "입력:\n"
        + "\n".join(input_lines)
    )
    tweak = (user_tweak or "").strip()
    if not tweak:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "[추가 지시]\n"
        f"{tweak}\n"
        "단, 위 JSON 스키마와 출력 컬럼 제약은 반드시 준수하라."
    )


def get_info_from_openai(client, prompt_text, model_name):
    response = client.responses.parse(
        model=model_name,
        tools=[{"type": "web_search"}],
        input=prompt_text,
        text_format=StructuredSearchResult,
    )
    return normalize_structured_values(response.output_parsed)


def extract_gemini_text(response_json):
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini 응답에 candidates가 없습니다.")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    text = "\n".join([t for t in texts if t]).strip()
    if text:
        return text

    finish_reason = candidates[0].get("finishReason", "unknown")
    raise ValueError(f"Gemini 텍스트 응답이 비어 있습니다. finishReason={finish_reason}")


def get_info_from_gemini(api_key, prompt_text, model_name, response_keys):
    contract_text = build_json_contract_text(response_keys)
    instruction = (
        f"{prompt_text}\n\n"
        "반드시 아래 JSON 형식만 반환해. 코드블록 없이 JSON만 출력해.\n"
        f"{contract_text}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": instruction}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    url = f"{GEMINI_API_BASE}/{model_name}:generateContent"
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=90)
    response.raise_for_status()
    raw_text = extract_gemini_text(response.json())
    return parse_structured_result_text(raw_text)


def resolve_api_key(input_key, provider):
    if input_key and input_key.strip():
        return input_key.strip()

    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_var_map.get(provider, "OPENAI_API_KEY")
    return os.getenv(env_var, "").strip()


def estimate_update_progress(state: str, step: str, message: str) -> float:
    state_norm = (state or "").strip().lower()
    step_norm = (step or "").strip().lower()
    message_norm = (message or "").strip().lower()

    if step_norm in UPDATE_PROGRESS_BY_STEP:
        return UPDATE_PROGRESS_BY_STEP[step_norm]
    if state_norm in UPDATE_PROGRESS_BY_STEP:
        return UPDATE_PROGRESS_BY_STEP[state_norm]

    # Backward compatibility: infer progress from status message if step is absent.
    if "download" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["downloading_release"]
    if "extract" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["extracting_release"]
    if "sync" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["syncing_dependencies"]
    if "restart" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["restarting_app"]
    return 0.12 if state_norm == "running" else 0.0


def resolve_output_path(raw_output_name):
    output_name = (raw_output_name or DEFAULT_OUTPUT).strip()
    if not output_name:
        output_name = DEFAULT_OUTPUT

    path = Path(output_name).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".csv")

    if path.exists() and path.is_dir():
        raise ValueError("출력 파일명이 디렉터리입니다.")

    if not path.parent.exists():
        raise ValueError(f"출력 경로가 존재하지 않습니다: {path.parent}")

    return path


def get_partial_path(output_path):
    return output_path.with_name(f"{output_path.stem}.partial.csv")


def classify_error(exc):
    status_code = getattr(exc, "status_code", None)
    if status_code is None and getattr(exc, "response", None) is not None:
        status_code = getattr(exc.response, "status_code", None)

    if status_code is not None:
        return f"{type(exc).__name__}:{status_code}"
    return type(exc).__name__


def save_partial(subset, partial_path):
    subset.to_csv(partial_path, index=False, encoding="utf-8-sig")


def merge_partial_results(subset, partial_df):
    partial_df = ensure_result_columns(partial_df.copy())
    if "_row_index" not in partial_df.columns:
        raise ValueError("partial 파일 형식이 올바르지 않습니다. (_row_index 누락)")

    merge_cols = ["_row_index"] + STANDARD_OUTPUT_COLUMNS
    partial_compact = (
        partial_df[merge_cols].drop_duplicates(subset=["_row_index"], keep="last")
    )

    base_cols = [c for c in subset.columns if c not in STANDARD_OUTPUT_COLUMNS]
    merged = subset[base_cols].merge(partial_compact, on="_row_index", how="left")
    merged = ensure_result_columns(merged)
    for col in ["성명", "회사명", "부서명", "직책명", "과제명", "email", "status", "error_code"]:
        merged[col] = merged[col].fillna("")
    return merged


def get_mapping_from_state():
    mapping = {}
    for spec in INPUT_FIELD_SPECS:
        raw_val = st.session_state.get(f"map_{spec['key']}_col")
        mapping[spec["key"]] = None if raw_val in (None, "(없음)") else raw_val
    return mapping


def extract_input_payload(row, mapping):
    payload = {}
    for spec in INPUT_FIELD_SPECS:
        key = spec["key"]
        src_col = mapping.get(key)
        if src_col and src_col in row.index and pd.notna(row[src_col]):
            payload[key] = str(row[src_col]).strip()
        else:
            payload[key] = ""
    return payload


def merge_input_and_result(input_payload, result_payload):
    merged = {}
    for key in ["name", "company", "department", "job_title", "project"]:
        merged[key] = input_payload[key] if input_payload[key] else result_payload.get(key, "")
    merged["email"] = result_payload.get("email", "")
    merged["confidence_score"] = result_payload.get("confidence_score", np.nan)
    return merged


def init_state():
    defaults = {
        "df_raw": None,
        "columns_signature": None,
        "head_rows": DEFAULT_HEAD_ROWS,
        "start": 0,
        "end": 0,
        "output_name": DEFAULT_OUTPUT,
        "dedupe_cols": [],
        "run_requested": False,
        "run_confirmed": False,
        "done_text": "",
        "autosave_every_n_rows": DEFAULT_AUTOSAVE_EVERY,
        "resume_from_partial": True,
        "provider": DEFAULT_PROVIDER,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "selected_output_columns": STANDARD_OUTPUT_COLUMNS.copy(),
        "prompt_user_tweak": "",
        "update_repo": os.getenv("APP_REPO", "").strip(),
        "update_check_result": None,
    }
    for spec in INPUT_FIELD_SPECS:
        defaults[f"map_{spec['key']}_col"] = None

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_loaded_data_state():
    st.session_state.df_raw = None
    st.session_state.columns_signature = None
    st.session_state.dedupe_cols = []
    st.session_state.run_requested = False
    st.session_state.run_confirmed = False
    st.session_state.done_text = ""
    for spec in INPUT_FIELD_SPECS:
        st.session_state[f"map_{spec['key']}_col"] = None


def render_update_sidebar(project_root: Path):
    st.sidebar.subheader("소프트웨어 업데이트")

    st.sidebar.text_input(
        "GitHub Repo (owner/repo)",
        key="update_repo",
        help="예: your-org/your-repo (또는 https://github.com/your-org/your-repo)",
    )

    github_token = os.getenv("GITHUB_TOKEN", "").strip()

    runtime_status = read_update_status(project_root) or {}
    state = (runtime_status.get("state") or "").lower()
    step = runtime_status.get("step") or ""
    message = runtime_status.get("message") or ""
    updated_at = runtime_status.get("updated_at") or ""
    is_update_running = state in {"queued", "running"}

    if st.sidebar.button(
        "최신 버전 확인", key="check_update_btn", disabled=is_update_running
    ):
        repo = st.session_state.update_repo.strip()
        if not repo:
            st.session_state.update_check_result = {
                "error": "APP_REPO 환경변수 또는 Repo 입력값이 필요합니다."
            }
        else:
            status = check_for_update(
                repo=repo,
                pyproject_path=project_root / "pyproject.toml",
                github_token=github_token,
            )
            st.session_state.update_check_result = status.to_dict()

    if state == "success":
        st.sidebar.success(f"최근 업데이트 성공: {message}")
    elif state == "failed":
        st.sidebar.error(f"최근 업데이트 실패: {message}")
    elif is_update_running:
        progress_value = estimate_update_progress(state=state, step=step, message=message)
        st.sidebar.progress(progress_value)
        state_label = step or state
        st.sidebar.info(f"업데이트 진행 중: {state_label} ({message})")
        st.sidebar.button("업데이트 상태 새로고침", key="refresh_update_status")
    if updated_at:
        st.sidebar.caption(f"업데이트 상태 갱신 시각: {updated_at}")

    result = st.session_state.update_check_result
    if not result:
        return

    if result.get("error"):
        st.sidebar.error(result["error"])
        return

    local_version = result.get("local_version")
    latest_version = result.get("latest_version")
    latest_tag = result.get("latest_tag")
    release_url = result.get("release_url")

    st.sidebar.caption(f"현재 버전: {local_version}")
    st.sidebar.caption(f"최신 버전: {latest_version} ({latest_tag})")

    if release_url:
        st.sidebar.markdown(f"[Release Notes]({release_url})")

    if not result.get("update_available"):
        st.sidebar.success("이미 최신 버전입니다.")
        return

    if is_update_running:
        st.sidebar.warning("업데이트가 이미 실행 중입니다. 완료 후 다시 시도해주세요.")
        return

    st.sidebar.warning("새 버전이 있습니다.")
    if st.sidebar.button("업데이트 실행", key="run_update_btn", disabled=is_update_running):
        try:
            # Prevent accidental carry-over into prompt task execution.
            st.session_state.run_requested = False
            st.session_state.run_confirmed = False
            release = ReleaseInfo(
                repo=result.get("repo") or st.session_state.update_repo,
                tag=result.get("latest_tag") or "",
                version=result.get("latest_version") or "",
                zip_url=result.get("zip_url") or "",
                html_url=result.get("release_url") or "",
                published_at=None,
            )
            if not release.zip_url:
                raise ValueError("zip_url 정보가 없어 업데이트를 실행할 수 없습니다.")

            launch_update_script(
                release=release,
                project_root=project_root,
                current_pid=os.getpid(),
                github_token=github_token,
            )
            st.sidebar.warning(
                "업데이트를 시작했습니다. 앱이 잠시 종료되거나 재시작될 수 있습니다."
            )
        except Exception as exc:
            st.sidebar.error(f"업데이트 시작 실패: {type(exc).__name__}: {exc}")


def main():
    st.set_page_config(page_title="AI Mail Collector", layout="wide")
    st.title("AI Mail Collector (Streamlit UI)")

    load_dotenv()
    init_state()

    project_root = Path(__file__).resolve().parent
    render_update_sidebar(project_root)

    st.sidebar.subheader("데이터 불러오기")
    uploaded = st.sidebar.file_uploader("파일 업로드 (csv/xlsx)", type=["csv", "xlsx"])
    if st.sidebar.button("입력 데이터 초기화", key="reset_input_state_btn"):
        clear_loaded_data_state()
        st.rerun()
    st.sidebar.number_input(
        "미리보기 행 수",
        min_value=1,
        max_value=200,
        value=st.session_state.head_rows,
        key="head_rows",
    )

    if uploaded is None:
        if st.session_state.df_raw is not None:
            clear_loaded_data_state()
    else:
        try:
            st.session_state.df_raw = load_data(uploaded)
        except Exception as exc:
            st.error(f"파일 로드 실패: {type(exc).__name__}")
            return

    df_raw = st.session_state.df_raw
    if df_raw is None:
        st.info("파일을 업로드하면 전처리/미리보기/실행 옵션이 표시됩니다.")
        return

    columns = list(df_raw.columns)
    columns_signature = tuple(columns)
    if st.session_state.get("columns_signature") != columns_signature:
        st.session_state.columns_signature = columns_signature

        auto_mapped = []
        for spec in INPUT_FIELD_SPECS:
            detected = detect_column_by_alias(columns, spec["aliases"])
            st.session_state[f"map_{spec['key']}_col"] = detected
            if detected:
                auto_mapped.append(detected)

        st.session_state.dedupe_cols = list(dict.fromkeys(auto_mapped))

    st.sidebar.subheader("입력 컬럼 설정")
    for spec in INPUT_FIELD_SPECS:
        key = spec["key"]
        field_key = f"map_{key}_col"
        options = ["(없음)"] + columns

        current = st.session_state.get(field_key)
        if current not in columns:
            current = "(없음)"

        st.sidebar.selectbox(
            f"{spec['label']} 입력 컬럼",
            options=options,
            index=options.index(current),
            key=field_key,
        )

    st.sidebar.multiselect(
        "중복 제거 기준 컬럼",
        options=columns,
        default=[c for c in st.session_state.dedupe_cols if c in columns],
        key="dedupe_cols",
    )

    mapping = get_mapping_from_state()
    if mapping.get("company") is None:
        st.warning("회사명 입력 컬럼을 설정해주세요. (회사명은 필수)")

    if st.session_state.dedupe_cols:
        df_processed = preprocess_df(df_raw, st.session_state.dedupe_cols)
    else:
        df_processed = df_raw.copy()

    st.subheader("미리보기")
    preview_df = df_processed.head(st.session_state.head_rows)
    row_height = 35
    header_height = 35
    preview_height = header_height + row_height * len(preview_df)
    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
        height=preview_height,
    )

    st.sidebar.subheader("실행 옵션")
    st.session_state.start = st.sidebar.number_input(
        "시작 인덱스",
        min_value=0,
        value=st.session_state.start,
    )
    st.session_state.end = st.sidebar.number_input(
        "끝 인덱스 (0이면 전체)",
        min_value=0,
        value=st.session_state.end,
    )
    st.session_state.output_name = st.sidebar.text_input(
        "출력 파일명", value=st.session_state.output_name
    )

    st.sidebar.multiselect(
        "출력 컬럼 선택",
        options=STANDARD_OUTPUT_COLUMNS,
        default=[
            c for c in st.session_state.selected_output_columns if c in STANDARD_OUTPUT_COLUMNS
        ],
        key="selected_output_columns",
        help="선택에서 제외한 컬럼은 프롬프트/응답/저장에서 모두 제외됩니다.",
    )

    raw_selected_output_columns = [
        c for c in st.session_state.selected_output_columns if c in STANDARD_OUTPUT_COLUMNS
    ]
    selected_output_columns = enforce_required_output_columns(raw_selected_output_columns)
    if set(selected_output_columns) != set(raw_selected_output_columns):
        st.sidebar.info("`email`, `confidence_score`는 필수 컬럼으로 자동 포함됩니다.")
    selected_prompt_columns = [
        c for c in selected_output_columns if c not in ("status", "error_code")
    ]

    st.sidebar.text_area(
        "프롬프트 추가 지시문(선택)",
        key="prompt_user_tweak",
        height=90,
        help="기본 프롬프트에 덧붙일 보조 지시입니다. JSON 형식/컬럼 제약은 자동 유지됩니다.",
        placeholder="예: 한국어 이름 표기는 원문 그대로 유지하고, 불확실하면 confidence_score를 낮춰줘.",
    )

    sample_payload = {k: "" for k in PROMPT_INPUT_KEYS}
    if len(df_processed) > 0:
        sample_payload = extract_input_payload(df_processed.iloc[0], mapping)

    prompt_preview = build_task_prompt(
        sample_payload, selected_prompt_columns, st.session_state.prompt_user_tweak
    )
    st.sidebar.text_area(
        "프롬프트(출력 컬럼 반영)",
        value=prompt_preview,
        height=220,
        disabled=True,
    )

    st.session_state.provider = st.sidebar.selectbox(
        "AI Provider",
        options=SUPPORTED_PROVIDERS,
        index=SUPPORTED_PROVIDERS.index(st.session_state.provider)
        if st.session_state.provider in SUPPORTED_PROVIDERS
        else 0,
    )

    if st.session_state.provider == "openai":
        st.session_state.openai_model = st.sidebar.text_input(
            "OpenAI Model", value=st.session_state.openai_model
        )
        selected_model = st.session_state.openai_model
        env_hint = "OPENAI_API_KEY"
    else:
        st.session_state.gemini_model = st.sidebar.text_input(
            "Gemini Model", value=st.session_state.gemini_model
        )
        selected_model = st.session_state.gemini_model
        env_hint = "GEMINI_API_KEY"

    api_key = st.sidebar.text_input(
        f"{st.session_state.provider.upper()} API Key",
        type="password",
        help=f"비워두면 {env_hint} 환경변수를 사용합니다.",
        value="",
    )

    st.sidebar.number_input(
        "중간 저장 간격(행)",
        min_value=1,
        max_value=1000,
        value=st.session_state.autosave_every_n_rows,
        key="autosave_every_n_rows",
    )
    st.sidebar.checkbox(
        "partial 파일에서 재개",
        value=st.session_state.resume_from_partial,
        key="resume_from_partial",
    )

    if st.sidebar.button("프롬프트 태스크 시작", type="primary"):
        st.session_state.run_requested = True
        st.session_state.run_confirmed = False
        st.session_state.done_text = ""

    if st.session_state.run_requested:
        def stop_requested(level: str, text: str):
            st.session_state.run_requested = False
            if level == "warning":
                st.warning(text)
            else:
                st.error(text)

        if not selected_output_columns:
            stop_requested("error", "최소 1개 이상의 출력 컬럼을 선택해주세요.")
            return
        if not selected_prompt_columns:
            stop_requested(
                "error",
                "status/error_code만 선택된 상태입니다. 추출할 출력 컬럼을 선택해주세요.",
            )
            return

        response_keys = build_response_contract(selected_prompt_columns)

        provider = st.session_state.provider
        resolved_api_key = resolve_api_key(api_key, provider)
        if not resolved_api_key:
            stop_requested("error", f"API Key를 입력하거나 {env_hint} 환경변수를 설정해주세요.")
            return

        if ("회사명" in selected_prompt_columns) and mapping.get("company") is None:
            stop_requested("error", "회사명 입력 컬럼을 설정해주세요. (회사명은 필수)")
            return

        if not selected_model or not str(selected_model).strip():
            stop_requested("error", "모델명을 입력해주세요.")
            return

        try:
            output_path = resolve_output_path(st.session_state.output_name)
        except ValueError as exc:
            stop_requested("error", str(exc))
            return

        partial_path = get_partial_path(output_path)

        df_run = df_processed.copy()
        df_run = ensure_result_columns(df_run)

        total = len(df_run)
        start = max(0, min(int(st.session_state.start), total))
        end = total if int(st.session_state.end) == 0 else min(int(st.session_state.end), total)

        if start >= end:
            stop_requested("warning", f"처리할 행이 없습니다. start={start}, end={end}")
            return

        subset = df_run.iloc[start:end].copy()
        subset["_row_index"] = subset.index
        subset = ensure_result_columns(subset)
        subset["status"] = subset["status"].fillna("")
        subset["error_code"] = subset["error_code"].fillna("")

        resumed_rows = 0
        if st.session_state.resume_from_partial and partial_path.exists():
            try:
                partial_df = pd.read_csv(partial_path)
                subset = merge_partial_results(subset, partial_df)
                resumed_rows = int(subset["status"].isin(RESULT_COMPLETE_STATES).sum())
                if resumed_rows > 0:
                    st.info(f"partial 재개: {resumed_rows}행은 기존 결과를 사용합니다.")
            except Exception as exc:
                st.warning(f"partial 파일을 읽지 못해 새로 시작합니다: {type(exc).__name__}")

        if not st.session_state.resume_from_partial:
            # Ignore carried status/error markers when partial resume is disabled.
            subset["status"] = ""
            subset["error_code"] = ""

        pending_subset = subset[~subset["status"].isin(RESULT_COMPLETE_STATES)].copy()
        if pending_subset.empty:
            final_df = subset.drop(columns=["_row_index"])
            final_df = final_df[selected_output_columns]
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            st.session_state.done_text = f"완료! 결과 저장: {output_path} (추가 처리 0행)"
            st.session_state.run_requested = False
            st.success(st.session_state.done_text)
            return

        st.info(
            f"처리 범위: {start} ~ {end} (총 {len(subset)}행, 신규 처리 {len(pending_subset)}행)"
        )
        st.caption(f"Provider: {provider}, Model: {selected_model}")

        client = OpenAI(api_key=resolved_api_key) if provider == "openai" else None

        progress = st.progress(0)
        status = st.empty()
        status.info("처리 중...")

        autosave_every = int(st.session_state.autosave_every_n_rows)
        success_count = 0
        error_count = 0
        total_pending = len(pending_subset)

        for i, (_, row) in enumerate(pending_subset.iterrows(), start=1):
            row_index = row["_row_index"]
            row_selector = subset["_row_index"] == row_index
            input_payload = extract_input_payload(row, mapping)

            try:
                prompt_text = build_task_prompt(
                    input_payload, selected_prompt_columns, st.session_state.prompt_user_tweak
                )
                if provider == "openai":
                    result_payload = get_info_from_openai(client, prompt_text, selected_model)
                else:
                    result_payload = get_info_from_gemini(
                        resolved_api_key, prompt_text, selected_model, response_keys
                    )

                merged = merge_input_and_result(input_payload, result_payload)

                subset.loc[row_selector, "성명"] = merged["name"]
                subset.loc[row_selector, "회사명"] = merged["company"]
                subset.loc[row_selector, "부서명"] = merged["department"]
                subset.loc[row_selector, "직책명"] = merged["job_title"]
                subset.loc[row_selector, "과제명"] = merged["project"]
                subset.loc[row_selector, "email"] = merged["email"]
                subset.loc[row_selector, "confidence_score"] = merged["confidence_score"]
                subset.loc[row_selector, "status"] = "success"
                subset.loc[row_selector, "error_code"] = ""
                success_count += 1
            except Exception as exc:
                error_code = classify_error(exc)
                subset.loc[row_selector, "성명"] = input_payload["name"]
                subset.loc[row_selector, "회사명"] = input_payload["company"]
                subset.loc[row_selector, "부서명"] = input_payload["department"]
                subset.loc[row_selector, "직책명"] = input_payload["job_title"]
                subset.loc[row_selector, "과제명"] = input_payload["project"]
                subset.loc[row_selector, "email"] = ""
                subset.loc[row_selector, "confidence_score"] = np.nan
                subset.loc[row_selector, "status"] = "error"
                subset.loc[row_selector, "error_code"] = error_code
                error_count += 1
                status.warning(
                    f"[SKIP] index={int(row_index)} 실패, 다음 행으로 진행합니다. "
                    f"(error_code={error_code})"
                )

            if autosave_every > 0 and i % autosave_every == 0:
                save_partial(subset, partial_path)

            progress.progress(i / total_pending)

        final_df = subset.drop(columns=["_row_index"])
        final_df = ensure_result_columns(final_df)
        final_df = final_df[selected_output_columns]
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        if partial_path.exists():
            try:
                partial_path.unlink()
            except OSError:
                pass

        st.session_state.done_text = (
            f"완료! 결과 저장: {output_path} "
            f"(성공 {success_count}행, 실패 {error_count}행, 재개 {resumed_rows}행)"
        )
        st.session_state.run_requested = False
        st.session_state.run_confirmed = False
        status.success(st.session_state.done_text)


def _running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if _running_with_streamlit():
        main()
    else:
        from streamlit.web import cli as stcli
        import sys

        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
