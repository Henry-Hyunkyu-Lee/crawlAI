import json
import os
import re
from pathlib import Path
from string import Formatter

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from updater import ReleaseInfo, check_for_update, launch_update_script, read_update_status

PROMPT_TEMPLATE = (
    "다음 정보를 바탕으로 웹검색을 수행해 해당 인물의 공식 이메일을 찾고, "
    "정확도를 confidence_score(0~1)로 답해줘.\n"
    "- feature_1: {feature_1}\n- feature_2: {feature_2}\n- feature_3: {feature_3}"
)

DEFAULT_OUTPUT = "results.csv"
DEFAULT_HEAD_ROWS = 30
DEFAULT_AUTOSAVE_EVERY = 20
RESULT_COMPLETE_STATES = {"success", "error"}
ALLOWED_PROMPT_FIELDS = {"feature_1", "feature_2", "feature_3"}

SUPPORTED_PROVIDERS = ["openai", "gemini"]
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class SearchResult(BaseModel):
    email: str
    confidence_score: float


def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def ensure_result_columns(df):
    if "Email" not in df.columns:
        df["Email"] = ""
    if "conf" not in df.columns:
        df["conf"] = np.nan
    if "status" not in df.columns:
        df["status"] = ""
    if "error_code" not in df.columns:
        df["error_code"] = ""
    return df


def preprocess_df(df, feature_cols):
    df_out = df.copy()
    if feature_cols:
        df_out = df_out[feature_cols]
        df_out = df_out.drop_duplicates(subset=feature_cols)
    return df_out


def validate_prompt_template(template):
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None and field_name != ""
    }
    invalid_fields = sorted(fields - ALLOWED_PROMPT_FIELDS)
    if invalid_fields:
        raise ValueError(f"허용되지 않은 변수: {', '.join(invalid_fields)}")


def build_prompt(template, name, org, extra):
    return template.format(
        feature_1=str(name) if pd.notna(name) else "N/A",
        feature_2=str(org) if pd.notna(org) else "N/A",
        feature_3=str(extra) if pd.notna(extra) else "N/A",
    )


def strip_code_fences(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_search_result_text(raw_text):
    cleaned = strip_code_fences(raw_text)
    if not cleaned:
        raise ValueError("모델 응답이 비어 있습니다.")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("모델 응답에서 JSON을 찾을 수 없습니다.")
        payload = json.loads(cleaned[start : end + 1])

    result = SearchResult.model_validate(payload)
    if not 0 <= result.confidence_score <= 1:
        raise ValueError("confidence_score는 0~1 범위여야 합니다.")
    return result


def get_info_from_openai(client, prompt_text, model_name):
    response = client.responses.parse(
        model=model_name,
        tools=[{"type": "web_search"}],
        input=prompt_text,
        text_format=SearchResult,
    )
    return response.output_parsed


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


def get_info_from_gemini(api_key, prompt_text, model_name):
    instruction = (
        f"{prompt_text}\n\n"
        "반드시 아래 JSON 형식만 반환해. 코드블록 없이 JSON만 출력해.\n"
        '{"email":"...","confidence_score":0.0}'
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
    return parse_search_result_text(raw_text)


def resolve_api_key(input_key, provider):
    if input_key and input_key.strip():
        return input_key.strip()

    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_var_map.get(provider, "OPENAI_API_KEY")
    return os.getenv(env_var, "").strip()


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

    merge_cols = ["_row_index", "Email", "conf", "status", "error_code"]
    partial_compact = (
        partial_df[merge_cols].drop_duplicates(subset=["_row_index"], keep="last")
    )

    base_cols = [
        c for c in subset.columns if c not in ["Email", "conf", "status", "error_code"]
    ]
    merged = subset[base_cols].merge(partial_compact, on="_row_index", how="left")
    merged = ensure_result_columns(merged)
    merged["Email"] = merged["Email"].fillna("")
    merged["status"] = merged["status"].fillna("")
    merged["error_code"] = merged["error_code"].fillna("")
    return merged


def init_state():
    defaults = {
        "df_raw": None,
        "head_rows": DEFAULT_HEAD_ROWS,
        "start": 0,
        "end": 0,
        "output_name": DEFAULT_OUTPUT,
        "name_col": None,
        "org_col": None,
        "extra_col": None,
        "feature_cols": [],
        "prompt_template": PROMPT_TEMPLATE,
        "run_requested": False,
        "run_confirmed": False,
        "done_text": "",
        "autosave_every_n_rows": DEFAULT_AUTOSAVE_EVERY,
        "resume_from_partial": True,
        "provider": DEFAULT_PROVIDER,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "update_repo": os.getenv("APP_REPO", "").strip(),
        "update_check_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_update_sidebar(project_root: Path):
    st.sidebar.subheader("소프트웨어 업데이트")

    st.sidebar.text_input(
        "GitHub Repo (owner/repo)",
        key="update_repo",
        help="예: your-org/your-repo (또는 https://github.com/your-org/your-repo)",
    )

    github_token = os.getenv("GITHUB_TOKEN", "").strip()

    if st.sidebar.button("최신 버전 확인", key="check_update_btn"):
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

    runtime_status = read_update_status(project_root)
    if runtime_status:
        state = (runtime_status.get("state") or "").lower()
        message = runtime_status.get("message") or ""
        updated_at = runtime_status.get("updated_at") or ""
        if state == "success":
            st.sidebar.success(f"최근 업데이트 성공: {message}")
        elif state == "failed":
            st.sidebar.error(f"최근 업데이트 실패: {message}")
        elif state in {"queued", "running"}:
            st.sidebar.info(f"업데이트 상태: {state} ({message})")
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

    st.sidebar.warning("새 버전이 있습니다.")
    if st.sidebar.button("업데이트 실행", key="run_update_btn"):
        try:
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
    st.sidebar.number_input(
        "미리보기 행 수",
        min_value=1,
        max_value=200,
        value=st.session_state.head_rows,
        key="head_rows",
    )

    if uploaded is not None:
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
        st.session_state.feature_cols = columns
        st.session_state.name_col = None
        st.session_state.org_col = None
        st.session_state.extra_col = None

    st.sidebar.subheader("전처리")
    st.session_state.feature_cols = [
        col for col in (st.session_state.feature_cols or []) if col in columns
    ]
    st.sidebar.multiselect(
        "피처 컬럼 선택",
        options=columns,
        default=st.session_state.feature_cols,
        key="feature_cols",
        help="크롤링에 사용할 컬럼입니다.",
    )

    param_options = st.session_state.feature_cols
    if param_options:
        name_options = param_options
        org_options = param_options
    else:
        name_options = ["(선택)"]
        org_options = ["(선택)"]

    st.session_state.name_col = st.sidebar.selectbox(
        "feature_1 컬럼",
        options=name_options,
        index=name_options.index(st.session_state.name_col)
        if st.session_state.name_col in name_options
        else 0,
    )
    st.session_state.org_col = st.sidebar.selectbox(
        "feature_2 컬럼",
        options=org_options,
        index=org_options.index(st.session_state.org_col)
        if st.session_state.org_col in org_options
        else 0,
    )

    extra_options = ["(없음)"] + param_options if param_options else ["(없음)"]
    if st.session_state.extra_col in param_options:
        extra_index = param_options.index(st.session_state.extra_col) + 1
    else:
        extra_index = 0
    st.session_state.extra_col = st.sidebar.selectbox(
        "feature_3 컬럼 (선택)",
        options=extra_options,
        index=extra_index,
    )

    if st.session_state.feature_cols:
        df_processed = preprocess_df(df_raw, st.session_state.feature_cols)
    else:
        df_processed = df_raw.iloc[0:0]

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

    if not st.session_state.feature_cols:
        st.warning("피처 컬럼을 선택해주세요.")

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
    st.session_state.prompt_template = st.sidebar.text_area(
        "프롬프트 템플릿",
        value=st.session_state.prompt_template,
        height=120,
        help="사용 가능 변수: {feature_1}, {feature_2}, {feature_3}",
    )

    if st.sidebar.button("크롤링 실행", type="primary"):
        st.session_state.run_requested = True
        st.session_state.run_confirmed = False
        st.session_state.done_text = ""

    if st.session_state.run_requested:
        try:
            validate_prompt_template(st.session_state.prompt_template)
        except ValueError as exc:
            st.error(f"프롬프트 템플릿 오류: {exc}")
            return

        provider = st.session_state.provider
        resolved_api_key = resolve_api_key(api_key, provider)
        if not resolved_api_key:
            st.error(f"API Key를 입력하거나 {env_hint} 환경변수를 설정해주세요.")
            return

        if (
            not st.session_state.name_col
            or not st.session_state.org_col
            or st.session_state.name_col == "(선택)"
            or st.session_state.org_col == "(선택)"
        ):
            st.error("feature_1/feature_2 컬럼을 선택해주세요.")
            return

        if not st.session_state.feature_cols:
            st.error("피처 컬럼을 선택해주세요.")
            return

        if not selected_model or not str(selected_model).strip():
            st.error("모델명을 입력해주세요.")
            return

        try:
            output_path = resolve_output_path(st.session_state.output_name)
        except ValueError as exc:
            st.error(str(exc))
            return

        partial_path = get_partial_path(output_path)

        df_run = df_processed.copy()
        df_run = ensure_result_columns(df_run)

        total = len(df_run)
        start = max(0, min(int(st.session_state.start), total))
        end = total if int(st.session_state.end) == 0 else min(int(st.session_state.end), total)

        if start >= end:
            st.warning(f"처리할 행이 없습니다. start={start}, end={end}")
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

        pending_subset = subset[~subset["status"].isin(RESULT_COMPLETE_STATES)].copy()
        if pending_subset.empty:
            final_df = subset.drop(columns=["_row_index"])
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

        extra_col = st.session_state.extra_col if st.session_state.extra_col != "(없음)" else None

        autosave_every = int(st.session_state.autosave_every_n_rows)
        success_count = 0
        error_count = 0
        total_pending = len(pending_subset)

        for i, (_, row) in enumerate(pending_subset.iterrows(), start=1):
            name_val = row[st.session_state.name_col]
            org_val = row[st.session_state.org_col]
            extra_val = row[extra_col] if extra_col and extra_col in subset.columns else "N/A"
            row_index = row["_row_index"]
            row_selector = subset["_row_index"] == row_index

            try:
                prompt_text = build_prompt(
                    st.session_state.prompt_template, name_val, org_val, extra_val
                )
                if provider == "openai":
                    result = get_info_from_openai(client, prompt_text, selected_model)
                else:
                    result = get_info_from_gemini(resolved_api_key, prompt_text, selected_model)

                subset.loc[row_selector, "Email"] = result.email
                subset.loc[row_selector, "conf"] = result.confidence_score
                subset.loc[row_selector, "status"] = "success"
                subset.loc[row_selector, "error_code"] = ""
                success_count += 1
            except Exception as exc:
                error_code = classify_error(exc)
                subset.loc[row_selector, "Email"] = ""
                subset.loc[row_selector, "conf"] = np.nan
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
