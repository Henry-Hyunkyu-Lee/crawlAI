import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from runner import RetryPolicy, backoff_seconds, classify_exception
from updater import (
    ReleaseInfo,
    check_for_update,
    get_local_version,
    launch_update_script,
    read_update_status,
    reset_update_runtime,
)

DEFAULT_OUTPUT = "results.csv"
DEFAULT_HEAD_ROWS = 30
DEFAULT_AUTOSAVE_EVERY = 20
RESULT_COMPLETE_STATES = {"success", "error"}

SUPPORTED_PROVIDERS = ["openai", "gemini"]
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

OUTPUT_COLUMNS = [
    "input_company",
    "input_domain",
    "input_context",
    "response_id",
    "response_text",
    "source_urls",
    "response_json",
    "error",
]
FALLBACK_BASE_PROMPT = """
[역할]
당신은 콜드 메일 발송을 위한 이메일 주소 조사 에이전트입니다.

[목표]
입력된 회사와 문맥 정보를 바탕으로 실제로 연락 가능한 담당자 이메일을 찾고,
결과를 반환하세요.

[규칙]
1. 확인되지 않은 값은 비워 둡니다.
2. 가능한 경우 기업 공식 사이트/신뢰 가능한 출처를 우선 사용합니다.
""".strip()

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
UPDATE_STATUS_STALE_SECONDS = 15 * 60


def get_default_prompt_from_test() -> str:
    try:
        from test import BASE_PROMPT as test_base_prompt  # type: ignore

        prompt = str(test_base_prompt or "").strip()
        if prompt:
            return prompt
    except Exception:
        pass
    return FALLBACK_BASE_PROMPT


def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    filename = (uploaded_file.name or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def normalize_header(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(text or "").lower())


def detect_column_by_alias(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized_aliases = {normalize_header(alias) for alias in aliases}
    for col in columns:
        if normalize_header(col) in normalized_aliases:
            return col
    return None


def to_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def resolve_api_key(input_key: str, provider: str) -> str:
    if input_key and input_key.strip():
        return input_key.strip()

    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_var_map.get(provider, "OPENAI_API_KEY")
    return os.getenv(env_var, "").strip()


def resolve_output_path(raw_output_name: str) -> Path:
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


def get_partial_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial.csv")


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS + ["status", "error_code", "retry_count"]:
        if col not in out.columns:
            out[col] = "" if col != "retry_count" else 0
    return out


def save_partial(subset: pd.DataFrame, partial_path: Path):
    subset.to_csv(partial_path, index=False, encoding="utf-8-sig")


def merge_partial_results(subset: pd.DataFrame, partial_df: pd.DataFrame) -> pd.DataFrame:
    merged = subset.copy()
    if "_row_index" not in partial_df.columns:
        raise ValueError("partial 파일 형식이 올바르지 않습니다. (_row_index 누락)")

    partial_compact = partial_df.drop_duplicates(subset=["_row_index"], keep="last")
    partial_compact = partial_compact.set_index("_row_index")

    for col in OUTPUT_COLUMNS + ["status", "error_code", "retry_count"]:
        if col not in partial_compact.columns:
            continue
        merged[col] = merged.apply(
            lambda r: partial_compact.at[r["_row_index"], col]
            if r["_row_index"] in partial_compact.index
            else r[col],
            axis=1,
        )

    merged["status"] = merged["status"].fillna("")
    merged["error_code"] = merged["error_code"].fillna("")
    merged["retry_count"] = merged["retry_count"].fillna(0)
    return merged

def extract_input_payload(row: pd.Series, mapping: Dict[str, object]) -> Dict[str, str]:
    company_col = str(mapping.get("company_col") or "")
    domain_col = str(mapping.get("domain_col") or "")
    context_cols = list(mapping.get("context_cols") or [])

    company = to_text(row.get(company_col, "")) if company_col else ""
    domain = to_text(row.get(domain_col, "")) if domain_col else ""

    context_items: List[str] = []
    for col in context_cols:
        if col not in row.index:
            continue
        value = to_text(row.get(col, ""))
        if value:
            context_items.append(f"{col}: {value}")

    return {
        "company": company,
        "domain": domain,
        "context": " | ".join(context_items),
    }


def build_prompt_text(prompt_template: str, payload: Dict[str, str]) -> str:
    lines: List[str] = [str(prompt_template or "").strip()]
    lines.append("")
    lines.append("[입력 데이터]")
    lines.append(f"- 회사명: {payload.get('company', '')}")

    domain = payload.get("domain", "")
    if domain:
        lines.append(f"- 웹사이트/도메인: {domain}")

    context = payload.get("context", "")
    if context:
        lines.append(f"- 참고 컨텍스트: {context}")

    return "\n".join(lines).strip()


def extract_web_sources_from_openai_response(response_obj) -> List[str]:
    try:
        payload = response_obj.model_dump()
    except Exception:
        return []

    urls: List[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        for source in action.get("sources") or []:
            if not isinstance(source, dict):
                continue
            url = str(source.get("url") or "").strip()
            if url and url not in urls:
                urls.append(url)
    return urls


def get_info_from_openai(client, prompt_text: str, model_name: str) -> Dict[str, object]:
    response = client.responses.create(
        model=model_name,
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        input=prompt_text,
    )
    response_text = str(getattr(response, "output_text", "") or "").strip()
    response_id = str(getattr(response, "id", "") or "").strip()
    sources = extract_web_sources_from_openai_response(response)
    try:
        response_json = response.model_dump_json()
    except Exception:
        response_json = str(response)
    return {
        "response_id": response_id,
        "response_text": response_text,
        "source_urls": sources,
        "response_json": response_json,
    }


def extract_gemini_text(response_json: Dict) -> str:
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


def get_info_from_gemini(api_key: str, prompt_text: str, model_name: str) -> Dict[str, object]:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    url = f"{GEMINI_API_BASE}/{model_name}:generateContent"
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=90)
    response.raise_for_status()
    response_obj = response.json()
    try:
        response_text = extract_gemini_text(response_obj)
    except Exception:
        response_text = ""
    return {
        "response_id": "",
        "response_text": response_text,
        "source_urls": [],
        "response_json": json.dumps(response_obj, ensure_ascii=False),
    }


def parse_iso_datetime(value: str) -> Optional[datetime]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def is_stale_update_runtime(state: str, updated_at: str) -> bool:
    state_norm = (state or "").strip().lower()
    if state_norm not in {"queued", "running"}:
        return False

    timestamp = parse_iso_datetime(updated_at)
    if not timestamp:
        return True

    age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
    return age_seconds > UPDATE_STATUS_STALE_SECONDS


def estimate_update_progress(state: str, step: str, message: str) -> float:
    state_norm = (state or "").strip().lower()
    step_norm = (step or "").strip().lower()
    message_norm = (message or "").strip().lower()

    if step_norm in UPDATE_PROGRESS_BY_STEP:
        return UPDATE_PROGRESS_BY_STEP[step_norm]
    if state_norm in UPDATE_PROGRESS_BY_STEP:
        return UPDATE_PROGRESS_BY_STEP[state_norm]

    if "download" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["downloading_release"]
    if "extract" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["extracting_release"]
    if "sync" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["syncing_dependencies"]
    if "restart" in message_norm:
        return UPDATE_PROGRESS_BY_STEP["restarting_app"]
    return 0.12 if state_norm == "running" else 0.0


def parse_version_tuple(version: str):
    raw = (version or "").strip()
    raw = raw.split("-", 1)[0].split("+", 1)[0]
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", raw)
    if not match:
        raise ValueError(f"unsupported version format: {version}")
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    return (major, minor, patch)


def is_update_available_from_versions(local_version: str, latest_version: str) -> bool:
    return parse_version_tuple(latest_version) > parse_version_tuple(local_version)

def render_update_sidebar(project_root: Path):
    st.sidebar.divider()
    st.sidebar.subheader("앱 업데이트")
    st.sidebar.caption("GitHub Release 기반 자동 업데이트")

    st.sidebar.text_input(
        "GitHub Repo (owner/repo)",
        key="update_repo",
        help="예: your-org/your-repo 또는 전체 GitHub URL",
    )

    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    pyproject_path = project_root / "pyproject.toml"

    runtime_status = read_update_status(project_root) or {}
    state = (runtime_status.get("state") or "").lower()
    step = runtime_status.get("step") or ""
    message = runtime_status.get("message") or ""
    updated_at = runtime_status.get("updated_at") or ""
    run_id = runtime_status.get("run_id") or ""
    error_code = runtime_status.get("error_code") or ""
    error_hint = runtime_status.get("error_hint") or ""
    started_at = runtime_status.get("started_at") or ""
    finished_at = runtime_status.get("finished_at") or ""
    progress_raw = runtime_status.get("progress")
    progress_value = None
    if progress_raw is not None:
        try:
            progress_value = min(1.0, max(0.0, float(progress_raw)))
        except (TypeError, ValueError):
            progress_value = None
    stale_runtime = is_stale_update_runtime(state=state, updated_at=updated_at)
    is_update_running = state in {"queued", "running"} and not stale_runtime

    top_col1, top_col2 = st.sidebar.columns(2)
    with top_col1:
        check_clicked = st.button("최신 버전 확인", key="check_update_btn", disabled=is_update_running)
    with top_col2:
        if st.button("상태 새로고침", key="refresh_update_btn"):
            st.rerun()

    if check_clicked:
        repo = st.session_state.update_repo.strip()
        if not repo:
            st.session_state.update_check_result = {
                "error": "APP_REPO 환경변수 또는 Repo 입력값이 필요합니다."
            }
        else:
            status = check_for_update(
                repo=repo,
                pyproject_path=pyproject_path,
                github_token=github_token,
            )
            st.session_state.update_check_result = status.to_dict()

    if stale_runtime:
        st.sidebar.warning("업데이트 상태가 오래되었습니다. 필요하면 상태를 초기화하세요.")
        if st.sidebar.button("stale 상태 초기화", key="reset_stale_update_btn"):
            reset_update_runtime(project_root)
            st.session_state.update_check_result = None
            st.rerun()

    if state == "success":
        st.sidebar.success(f"최근 업데이트 성공: {message}")
    elif state == "failed":
        st.sidebar.error(f"최근 업데이트 실패: {message}")
    elif is_update_running:
        effective_progress = (
            progress_value
            if progress_value is not None
            else estimate_update_progress(state=state, step=step, message=message)
        )
        st.sidebar.progress(effective_progress)
        st.sidebar.info(f"업데이트 진행 중: {step or state} ({message})")

    if updated_at:
        st.sidebar.caption(f"상태 갱신 시각: {updated_at}")
    if started_at:
        st.sidebar.caption(f"시작 시각: {started_at}")
    if finished_at:
        st.sidebar.caption(f"종료 시각: {finished_at}")
    if run_id:
        st.sidebar.caption(f"실행 ID: {run_id}")
    if error_code:
        st.sidebar.caption(f"오류 코드: {error_code}")
    if error_hint:
        st.sidebar.warning(f"조치 가이드: {error_hint}")

    result = st.session_state.update_check_result
    if not result:
        return

    live_local_version = None
    try:
        live_local_version = get_local_version(pyproject_path)
    except Exception:
        pass

    if live_local_version and isinstance(result, dict):
        cached_local_version = result.get("local_version")
        if cached_local_version != live_local_version:
            refreshed = dict(result)
            refreshed["local_version"] = live_local_version
            latest_version = refreshed.get("latest_version")
            if latest_version:
                try:
                    refreshed["update_available"] = is_update_available_from_versions(
                        live_local_version, latest_version
                    )
                except Exception:
                    pass
            st.session_state.update_check_result = refreshed
            result = refreshed

    if result.get("error"):
        st.sidebar.error(result["error"])
        return

    local_version = live_local_version or result.get("local_version")
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
            st.session_state.run_requested = False
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
            st.sidebar.warning("업데이트를 시작했습니다. 앱이 잠시 종료되거나 재시작될 수 있습니다.")
        except Exception as exc:
            st.sidebar.error(f"업데이트 시작 실패: {type(exc).__name__}: {exc}")


def init_state():
    defaults = {
        "df_raw": None,
        "columns_signature": None,
        "head_rows": DEFAULT_HEAD_ROWS,
        "map_company_col": None,
        "map_domain_col": None,
        "map_context_cols": [],
        "prompt_template": get_default_prompt_from_test(),
        "provider": DEFAULT_PROVIDER,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "start": 0,
        "end": 0,
        "output_name": DEFAULT_OUTPUT,
        "autosave_every_n_rows": DEFAULT_AUTOSAVE_EVERY,
        "resume_from_partial": True,
        "run_requested": False,
        "done_text": "",
        "update_repo": os.getenv("APP_REPO", "").strip(),
        "update_check_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_loaded_data_state():
    st.session_state.df_raw = None
    st.session_state.columns_signature = None
    st.session_state.map_company_col = None
    st.session_state.map_domain_col = None
    st.session_state.map_context_cols = []
    st.session_state.run_requested = False
    st.session_state.done_text = ""


def run_workflow(
    df_source: pd.DataFrame,
    mapping: Dict[str, object],
    prompt_template: str,
    provider: str,
    model_name: str,
    api_key: str,
    start: int,
    end: int,
    output_name: str,
    autosave_every: int,
    resume_from_partial: bool,
):
    output_path = resolve_output_path(output_name)
    partial_path = get_partial_path(output_path)

    df_run = ensure_result_columns(df_source)
    total = len(df_run)
    start_idx = max(0, min(int(start), total))
    end_idx = total if int(end) == 0 else min(int(end), total)

    if start_idx >= end_idx:
        raise ValueError(f"처리 범위가 비어 있습니다. start={start_idx}, end={end_idx}")

    subset = df_run.iloc[start_idx:end_idx].copy()
    subset["_row_index"] = subset.index
    subset = ensure_result_columns(subset)
    subset["status"] = subset["status"].fillna("")
    subset["error_code"] = subset["error_code"].fillna("")
    subset["retry_count"] = subset["retry_count"].fillna(0)

    resumed_rows = 0
    if resume_from_partial and partial_path.exists():
        try:
            partial_df = pd.read_csv(partial_path)
            subset = merge_partial_results(subset, partial_df)
            resumed_rows = int(subset["status"].isin(RESULT_COMPLETE_STATES).sum())
            if resumed_rows > 0:
                st.info(f"partial 재개: {resumed_rows}행은 기존 결과를 사용합니다.")
        except Exception as exc:
            st.warning(f"partial 파일을 읽지 못해 새로 시작합니다: {type(exc).__name__}")

    if not resume_from_partial:
        subset["status"] = ""
        subset["error_code"] = ""
        subset["retry_count"] = 0

    pending_subset = subset[~subset["status"].isin(RESULT_COMPLETE_STATES)].copy()
    if pending_subset.empty:
        final_df = subset.drop(columns=["_row_index"])
        final_df = ensure_result_columns(final_df)
        final_df = final_df[OUTPUT_COLUMNS]
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        if partial_path.exists():
            try:
                partial_path.unlink()
            except OSError:
                pass
        st.success(f"완료! 결과 저장: {output_path} (추가 처리 0행)")
        return

    st.info(
        f"처리 범위: {start_idx} ~ {end_idx} (총 {len(subset)}행, 신규 처리 {len(pending_subset)}행)"
    )
    st.caption(f"Provider: {provider}, Model: {model_name}")

    client = OpenAI(api_key=api_key) if provider == "openai" else None
    retry_policy = RetryPolicy(network_retries=3, validation_retries=0, backoff_seconds=1.5)

    progress = st.progress(0)
    status_box = st.empty()
    status_box.info("처리 중...")

    autosave = max(1, int(autosave_every))
    success_count = 0
    error_count = 0
    total_pending = len(pending_subset)

    for i, (_, row) in enumerate(pending_subset.iterrows(), start=1):
        row_index = row["_row_index"]
        row_selector = subset["_row_index"] == row_index
        input_payload = extract_input_payload(row, mapping)
        prompt_text = build_prompt_text(prompt_template, input_payload)

        retries_used = 0
        try:
            network_retries_left = retry_policy.network_retries
            while True:
                try:
                    if provider == "openai":
                        response_record = get_info_from_openai(client, prompt_text, model_name)
                    else:
                        response_record = get_info_from_gemini(api_key, prompt_text, model_name)
                    break
                except Exception as exc:
                    _, retryable = classify_exception(exc)
                    if retryable and network_retries_left > 0:
                        retries_used += 1
                        network_retries_left -= 1
                        time.sleep(backoff_seconds(retry_policy, retries_used))
                        continue
                    raise

            subset.loc[row_selector, "input_company"] = input_payload.get("company", "")
            subset.loc[row_selector, "input_domain"] = input_payload.get("domain", "")
            subset.loc[row_selector, "input_context"] = input_payload.get("context", "")
            subset.loc[row_selector, "response_id"] = str(
                response_record.get("response_id", "") or ""
            ).strip()
            subset.loc[row_selector, "response_text"] = str(
                response_record.get("response_text", "") or ""
            ).strip()
            source_urls = response_record.get("source_urls") or []
            if isinstance(source_urls, list):
                source_urls_text = "\n".join(
                    [str(u).strip() for u in source_urls if str(u).strip()]
                )
            else:
                source_urls_text = str(source_urls or "").strip()
            subset.loc[row_selector, "source_urls"] = source_urls_text
            subset.loc[row_selector, "response_json"] = str(
                response_record.get("response_json", "") or ""
            )
            subset.loc[row_selector, "error"] = ""

            subset.loc[row_selector, "status"] = "success"
            subset.loc[row_selector, "error_code"] = ""
            subset.loc[row_selector, "retry_count"] = retries_used
            success_count += 1
        except Exception as exc:
            error_code, _ = classify_exception(exc)
            subset.loc[row_selector, "input_company"] = input_payload.get("company", "")
            subset.loc[row_selector, "input_domain"] = input_payload.get("domain", "")
            subset.loc[row_selector, "input_context"] = input_payload.get("context", "")
            subset.loc[row_selector, "response_id"] = ""
            subset.loc[row_selector, "response_text"] = ""
            subset.loc[row_selector, "source_urls"] = ""
            subset.loc[row_selector, "response_json"] = ""
            subset.loc[row_selector, "error"] = f"{type(exc).__name__}: {exc}"
            subset.loc[row_selector, "status"] = "error"
            subset.loc[row_selector, "error_code"] = error_code
            subset.loc[row_selector, "retry_count"] = retries_used
            error_count += 1
            status_box.warning(
                f"[SKIP] index={int(row_index)} 실패, 다음 행으로 진행합니다 (error_code={error_code}, retries={retries_used})"
            )

        if autosave > 0 and i % autosave == 0:
            save_partial(subset, partial_path)

        progress.progress(i / total_pending)

    final_df = subset.drop(columns=["_row_index"])
    final_df = ensure_result_columns(final_df)
    final_df = final_df[OUTPUT_COLUMNS]
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if partial_path.exists():
        try:
            partial_path.unlink()
        except OSError:
            pass

    status_box.success(
        f"완료! 결과 저장: {output_path} (성공 {success_count}행, 실패 {error_count}행, 재개 {resumed_rows}행)"
    )


def main():
    st.set_page_config(page_title="AI Mail Collector", layout="wide")
    st.title("AI Mail Collector")
    st.caption("목표: 콜드 메일 발송을 위한 이메일 주소 크롤링")

    load_dotenv()
    init_state()

    project_root = Path(__file__).resolve().parent

    st.sidebar.subheader("1) 데이터 준비")
    uploaded = st.sidebar.file_uploader("파일 업로드 (csv/xlsx)", type=["csv", "xlsx"])
    if st.sidebar.button("입력 데이터 초기화", key="reset_input_state_btn"):
        clear_loaded_data_state()
        st.rerun()

    st.sidebar.number_input(
        "미리보기 행 수",
        min_value=1,
        max_value=200,
        value=int(st.session_state.head_rows),
        key="head_rows",
    )

    if uploaded is None:
        if st.session_state.df_raw is not None:
            clear_loaded_data_state()
    else:
        try:
            st.session_state.df_raw = load_data(uploaded)
        except Exception as exc:
            st.error(f"파일 로드 실패: {type(exc).__name__}: {exc}")
            render_update_sidebar(project_root)
            return

    df_raw = st.session_state.df_raw
    if df_raw is None:
        st.info("파일을 업로드하면 3단계 워크플로 UI가 활성화됩니다.")
        render_update_sidebar(project_root)
        return

    columns = list(df_raw.columns)
    columns_signature = tuple(columns)
    if st.session_state.get("columns_signature") != columns_signature:
        st.session_state.columns_signature = columns_signature
        st.session_state.map_company_col = detect_column_by_alias(
            columns,
            ["회사명", "기업명", "company", "organization", "org"],
        )
        st.session_state.map_domain_col = detect_column_by_alias(
            columns,
            ["도메인", "웹사이트", "website", "domain", "site", "url"],
        )

    company_options = ["(없음)"] + columns
    domain_options = ["(없음)"] + columns

    if st.session_state.map_company_col not in company_options:
        st.session_state.map_company_col = "(없음)"
    if st.session_state.map_domain_col not in domain_options:
        st.session_state.map_domain_col = "(없음)"

    selected_company_col = st.sidebar.selectbox(
        "회사명 입력 컬럼",
        options=company_options,
        index=company_options.index(st.session_state.map_company_col),
        key="map_company_col",
    )
    selected_domain_col = st.sidebar.selectbox(
        "웹사이트/도메인 컬럼 (선택)",
        options=domain_options,
        index=domain_options.index(st.session_state.map_domain_col),
        key="map_domain_col",
    )
    selected_context_cols = st.sidebar.multiselect(
        "추가 컨텍스트 컬럼 (선택)",
        options=columns,
        default=[c for c in st.session_state.map_context_cols if c in columns],
        key="map_context_cols",
    )
    if st.sidebar.button("미리보기/프롬프트 갱신", key="refresh_preview_btn"):
        st.rerun()

    mapping = {
        "company_col": None if selected_company_col == "(없음)" else selected_company_col,
        "domain_col": None if selected_domain_col == "(없음)" else selected_domain_col,
        "context_cols": selected_context_cols,
    }

    st.subheader("Step 1. 데이터 준비")
    st.caption("입력 파일 업로드 후, 회사명/도메인/추가 컨텍스트 컬럼을 매핑합니다.")
    preview_df = df_raw.head(int(st.session_state.head_rows))
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    context_cols_text = ", ".join(mapping["context_cols"]) if mapping["context_cols"] else "(없음)"
    st.caption(
        "현재 매핑: "
        f"회사명={mapping['company_col'] or '(없음)'} | "
        f"도메인={mapping['domain_col'] or '(없음)'} | "
        f"컨텍스트={context_cols_text}"
    )

    if not mapping["company_col"]:
        st.warning("회사명 입력 컬럼은 필수입니다.")

    default_base_prompt = get_default_prompt_from_test()
    st.sidebar.subheader("2) 프롬프트 작성")
    st.sidebar.text_area(
        "프롬프트",
        value=default_base_prompt,
        key="prompt_template",
        height=260,
        help="기본값은 test.py의 BASE_PROMPT를 사용합니다.",
    )

    sample_payload = {"company": "", "domain": "", "context": ""}
    if len(df_raw) > 0:
        sample_payload = extract_input_payload(df_raw.iloc[0], mapping)

    rendered_prompt = build_prompt_text(st.session_state.prompt_template, sample_payload)

    st.subheader("Step 2. 프롬프트 작성")
    st.caption("입력값이 프롬프트에 어떻게 반영되고, 어떤 출력이 생성되는지 즉시 확인합니다.")

    schema_df = pd.DataFrame(
        [
            {"출력 키": "input_company", "설명": "입력 회사명(원문)"},
            {"출력 키": "input_domain", "설명": "입력 도메인(원문)"},
            {"출력 키": "input_context", "설명": "입력 컨텍스트(원문)"},
            {"출력 키": "response_id", "설명": "모델 응답 ID"},
            {"출력 키": "response_text", "설명": "모델 응답 텍스트 원문"},
            {"출력 키": "source_urls", "설명": "웹 검색 출처 URL 원문"},
            {"출력 키": "response_json", "설명": "모델 응답 JSON 원문"},
            {"출력 키": "error", "설명": "실패 시 오류 원문"},
        ]
    )
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**저장 CSV 포맷 (원문 보존)**")
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    with col_right:
        st.markdown("**샘플 입력 렌더링**")
        st.code(
            (
                "[입력 샘플]\n"
                f"- 회사명: {sample_payload.get('company','')}\n"
                f"- 도메인: {sample_payload.get('domain','')}\n"
                f"- 컨텍스트: {sample_payload.get('context','')}"
            ),
            language="text",
        )

    st.text_area(
        "렌더링된 프롬프트 (샘플 입력)",
        value=rendered_prompt,
        height=280,
        disabled=True,
    )
    st.caption("실행 메타데이터(row_index 등)는 모델 입력 프롬프트에 포함하지 않습니다.")

    st.sidebar.subheader("3) 실행 옵션 설정")
    st.session_state.provider = st.sidebar.selectbox(
        "Provider",
        options=SUPPORTED_PROVIDERS,
        index=SUPPORTED_PROVIDERS.index(st.session_state.provider)
        if st.session_state.provider in SUPPORTED_PROVIDERS
        else 0,
    )

    if st.session_state.provider == "openai":
        selected_model = st.sidebar.text_input(
            "Model",
            value=st.session_state.openai_model,
            key="openai_model",
        )
        env_hint = "OPENAI_API_KEY"
    else:
        selected_model = st.sidebar.text_input(
            "Model",
            value=st.session_state.gemini_model,
            key="gemini_model",
        )
        env_hint = "GEMINI_API_KEY"

    api_key = st.sidebar.text_input(
        f"{st.session_state.provider.upper()} API Key",
        type="password",
        value="",
        help=f"비워두면 {env_hint} 환경변수를 사용합니다.",
    )

    st.session_state.start = st.sidebar.number_input(
        "시작 인덱스",
        min_value=0,
        value=int(st.session_state.start),
    )
    st.session_state.end = st.sidebar.number_input(
        "종료 인덱스 (0이면 전체)",
        min_value=0,
        value=int(st.session_state.end),
    )
    st.session_state.output_name = st.sidebar.text_input(
        "출력 파일명",
        value=st.session_state.output_name,
    )

    st.sidebar.number_input(
        "중간 저장 간격(행)",
        min_value=1,
        max_value=1000,
        value=int(st.session_state.autosave_every_n_rows),
        key="autosave_every_n_rows",
    )
    st.sidebar.checkbox(
        "partial 파일에서 재개",
        value=bool(st.session_state.resume_from_partial),
        key="resume_from_partial",
    )

    total_rows = len(df_raw)
    preview_start = max(0, min(int(st.session_state.start), total_rows))
    preview_end = total_rows if int(st.session_state.end) == 0 else min(int(st.session_state.end), total_rows)

    execution_blockers: List[str] = []
    if not mapping["company_col"]:
        execution_blockers.append("회사명 입력 컬럼을 선택하세요.")
    if not selected_model or not str(selected_model).strip():
        execution_blockers.append("Model을 입력하세요.")
    resolved_api = resolve_api_key(api_key, st.session_state.provider)
    if not resolved_api:
        execution_blockers.append(f"API Key를 입력하거나 {env_hint} 환경변수를 설정하세요.")
    if preview_start >= preview_end:
        execution_blockers.append(f"처리 범위가 비어 있습니다. start={preview_start}, end={preview_end}")

    st.subheader("Step 3. 실행 옵션 설정")
    st.caption("Provider/Model/API Key와 실행 범위, 중간저장/재개만 설정합니다.")

    if execution_blockers:
        st.sidebar.error("실행 체크리스트 미충족")
        for item in execution_blockers:
            st.sidebar.caption(f"- {item}")
    else:
        st.sidebar.success("실행 체크리스트 충족")

    if st.sidebar.button("크롤링 실행", type="primary", disabled=bool(execution_blockers)):
        st.session_state.run_requested = True
        st.session_state.done_text = ""

    if st.session_state.run_requested:
        try:
            run_workflow(
                df_source=df_raw,
                mapping=mapping,
                prompt_template=st.session_state.prompt_template,
                provider=st.session_state.provider,
                model_name=selected_model,
                api_key=resolved_api,
                start=int(st.session_state.start),
                end=int(st.session_state.end),
                output_name=st.session_state.output_name,
                autosave_every=int(st.session_state.autosave_every_n_rows),
                resume_from_partial=bool(st.session_state.resume_from_partial),
            )
        except Exception as exc:
            st.error(f"실행 실패: {type(exc).__name__}: {exc}")
        finally:
            st.session_state.run_requested = False

    render_update_sidebar(project_root)


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
