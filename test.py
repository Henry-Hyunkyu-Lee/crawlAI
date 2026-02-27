import argparse
import io
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_OUTPUT = "beta_results.csv"
DEFAULT_COLUMN = "회사명"

BASE_PROMPT = """
[역할 부여] 너는 신규 고객을 발굴하는 기획팀 멤버야.

[문제 정의] 고객회사명을 확인하여, 임직원을 찾고, C-level에 가까운 사람의 이름, 직책, 이메일 주소를 찾아야 해.

[검색 키워드]
1.필수: 회사 이메일 도메인, 임직원 이름
2.선택 이지만 1개 이상 반드시 포함: "electronic address" "corresponder address" "corresponding address" "Correspondence" "email" "e-mail"

[출력 요구]
-반드시 있는 정보만 기재할 것.
1) 회사명
2) 성명
3) 직책
4) 이메일 주소
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beta: 회사명 열을 읽어 OpenAI Responses API로 임직원 정보를 수집합니다."
    )
    parser.add_argument("--input", required=True, help="입력 파일 경로 (.csv 또는 .xlsx)")
    parser.add_argument("--column", default=DEFAULT_COLUMN, help="회사명이 들어있는 열 이름")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="결과 CSV 파일 경로")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Responses API 모델명")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 행 수 (0은 전체)")
    return parser.parse_args()


def read_csv_fallback_from_path(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def read_csv_fallback_from_bytes(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def load_input_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_fallback_from_path(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def load_input_upload(uploaded_file) -> pd.DataFrame:
    filename = (uploaded_file.name or "").lower()
    if filename.endswith(".csv"):
        raw = uploaded_file.getvalue()
        return read_csv_fallback_from_bytes(raw)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("지원하지 않는 파일 형식입니다. csv/xlsx/xls만 가능합니다.")


def to_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def extract_companies(df: pd.DataFrame, column: str, limit: int = 0) -> List[str]:
    if column not in df.columns:
        raise ValueError(f"선택한 열을 찾을 수 없습니다: {column}")

    companies = [to_text(v) for v in df[column].tolist()]
    companies = [c for c in companies if c]

    if limit > 0:
        companies = companies[:limit]

    if not companies:
        raise ValueError(f"열 '{column}'에서 처리할 회사명이 없습니다.")

    return companies


def extract_web_sources(response_obj) -> str:
    try:
        payload = response_obj.model_dump()
    except Exception:
        return ""

    urls: List[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "web_search_call":
            continue

        action = item.get("action") or {}
        sources = action.get("sources") or []
        for source in sources:
            if not isinstance(source, dict):
                continue
            url = to_text(source.get("url"))
            if url and url not in urls:
                urls.append(url)

    return "\n".join(urls)


def ask_company_contact(client: OpenAI, model: str, company_name: str) -> Dict[str, str]:
    response = client.responses.create(
        model=model,
        instructions=BASE_PROMPT,
        input=company_name,
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
    )

    try:
        response_json = response.model_dump_json()
    except Exception:
        response_json = str(response)

    return {
        "입력회사명": company_name,
        "응답ID": to_text(getattr(response, "id", "")),
        "응답텍스트": to_text(getattr(response, "output_text", "")),
        "웹출처URL": extract_web_sources(response),
        "응답JSON": response_json,
        "오류": "",
    }


def collect_contacts(
    client: OpenAI,
    model: str,
    companies: List[str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    total = len(companies)

    for idx, company_name in enumerate(companies, start=1):
        if progress_cb:
            progress_cb(idx, total, company_name)

        try:
            rows.append(ask_company_contact(client, model, company_name))
        except Exception as exc:
            rows.append(
                {
                    "입력회사명": company_name,
                    "응답ID": "",
                    "응답텍스트": "",
                    "웹출처URL": "",
                    "응답JSON": "",
                    "오류": f"{type(exc).__name__}: {exc}",
                }
            )

    return rows


def run_cli() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    df = load_input_path(input_path)
    companies = extract_companies(df, args.column, args.limit)
    client = OpenAI(api_key=api_key)

    def cli_progress(i: int, total: int, company: str) -> None:
        print(f"[{i}/{total}] 처리 중: {company}")

    rows = collect_contacts(client=client, model=args.model, companies=companies, progress_cb=cli_progress)

    output_path = Path(args.output).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"완료: {output_path} ({len(rows)}건)")


def run_gui() -> None:
    load_dotenv()

    st.set_page_config(page_title="Beta 고객 발굴", layout="wide")
    st.title("Beta 고객 발굴 GUI")
    st.caption("출력 형식을 강제하지 않고, 응답 텍스트와 원문 JSON을 모두 저장합니다.")

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key_input = st.text_input(
        "OpenAI API Key (비우면 .env의 OPENAI_API_KEY 사용)",
        type="password",
        value="",
    ).strip()
    api_key = api_key_input or env_key

    model = st.text_input("Model", value=DEFAULT_MODEL)
    uploaded = st.file_uploader("파일 업로드 (csv/xlsx/xls)", type=["csv", "xlsx", "xls"])

    if not uploaded:
        st.info("파일을 업로드하면 열 선택과 실행 버튼이 나타납니다.")
        return

    try:
        df = load_input_upload(uploaded)
    except Exception as exc:
        st.error(f"파일 로드 실패: {type(exc).__name__}: {exc}")
        return

    if df.empty:
        st.warning("업로드한 파일이 비어 있습니다.")
        return

    columns = [str(c) for c in df.columns]
    default_index = columns.index(DEFAULT_COLUMN) if DEFAULT_COLUMN in columns else 0

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_column = st.selectbox("회사명 열 선택", options=columns, index=default_index)
    with col2:
        limit = int(st.number_input("처리 건수 제한", min_value=0, value=0, step=1))
    with col3:
        output_name = st.text_input("저장 파일명", value=DEFAULT_OUTPUT)

    st.write("미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    run = st.button("실행", type="primary")
    if not run:
        return

    if not api_key:
        st.error("OPENAI_API_KEY를 입력하거나 .env에 설정해주세요.")
        return

    try:
        companies = extract_companies(df, selected_column, limit)
    except Exception as exc:
        st.error(f"입력 검증 실패: {type(exc).__name__}: {exc}")
        return

    client = OpenAI(api_key=api_key)
    progress = st.progress(0.0)
    status = st.empty()

    def ui_progress(i: int, total: int, company: str) -> None:
        progress.progress(i / total)
        status.text(f"[{i}/{total}] 처리 중: {company}")

    with st.spinner("API 요청 실행 중..."):
        rows = collect_contacts(client=client, model=model, companies=companies, progress_cb=ui_progress)

    result_df = pd.DataFrame(rows)
    ok_count = int((result_df["오류"].astype(str).str.strip() == "").sum())
    status.success(f"완료: 총 {len(result_df)}건, 성공 {ok_count}건")

    st.dataframe(result_df[["입력회사명", "응답ID", "응답텍스트", "웹출처URL", "오류"]], use_container_width=True)

    csv_bytes = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "결과 CSV 다운로드",
        data=csv_bytes,
        file_name=output_name if output_name.endswith(".csv") else f"{output_name}.csv",
        mime="text/csv",
    )


def _running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _has_cli_args(argv: List[str]) -> bool:
    cli_flags = {"--input", "--help", "-h"}
    return any(arg in cli_flags or arg.startswith("--input=") for arg in argv)


if __name__ == "__main__":
    if _running_with_streamlit():
        run_gui()
    elif _has_cli_args(sys.argv[1:]):
        run_cli()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        raise SystemExit(stcli.main())
