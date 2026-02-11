import os
import re
import csv
import time
import json
import argparse
import requests

from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple

# ---------- OpenAI API ----------
BASE = "https://api.openai.com/v1"


def get_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {api_key}"})
    return s


def get_json(session: requests.Session, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_response(session: requests.Session, response_id: str) -> Dict[str, Any]:
    return get_json(session, f"{BASE}/responses/{response_id}")


def fetch_all_input_items(session: requests.Session, response_id: str, limit: int = 100, order: str = "asc") -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    after = None

    while True:
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after

        j = get_json(session, f"{BASE}/responses/{response_id}/input_items", params=params)
        batch = j.get("data", []) or []
        items.extend(batch)

        # pagination patterns
        after = j.get("next") or j.get("next_page_token")
        if not after:
            if j.get("has_more") and batch and batch[-1].get("id"):
                after = batch[-1]["id"]
            else:
                break

    return items


# ---------- Text helpers ----------
def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    texts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue

            if isinstance(part.get("text"), str):
                texts.append(part["text"])
                continue

            if isinstance(part.get("text"), dict) and isinstance(part["text"].get("value"), str):
                texts.append(part["text"]["value"])
                continue

            if isinstance(part.get("value"), str):
                texts.append(part["value"])
                continue

            ptype = part.get("type")
            if isinstance(ptype, str) and (("image" in ptype) or ("file" in ptype)):
                texts.append(f"[{ptype}]")

    return "\n".join(t for t in texts if t).strip()


def collect_user_input_text(input_items: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for it in input_items:
        if it.get("type") != "message":
            continue
        if it.get("role") != "user":
            continue
        txt = extract_text_from_content(it.get("content"))
        if txt:
            chunks.append(txt)
    return "\n\n---\n\n".join(chunks).strip()


def collect_assistant_output_text(resp_json: Dict[str, Any]) -> str:
    out = resp_json.get("output_text")
    if isinstance(out, str) and out.strip():
        return out.strip()

    texts: List[str] = []
    for item in resp_json.get("output", []) or []:
        if item.get("type") == "message":
            txt = extract_text_from_content(item.get("content"))
            if txt:
                texts.append(txt)
    return "\n".join(texts).strip()


# ---------- Parsing user_input ----------
FIELD_PATTERNS = {
    "연구기관명": [r"-\s*연구기관명\s*:\s*(.+)"],
    "책임자": [r"-\s*책임자\s*:\s*(.+)"],
    "과제명": [
        r"-\s*과제명\s*(?:\(\s*optional\s*\))?\s*:\s*(.+)",
    ],

    # 새 입력 포맷 : feature_1 문장
    "feature_1": [r"-\s*feature_1\s*:\s*(.+)"],
    # 새 입력 포맷 : feature_2 문장
    "feature_2": [r"-\s*feature_2\s*:\s*(.+)"],
    # 새 입력 포맷: 연구주제 = feature_3 문장
    "연구주제": [
        r"관련된\s*연구(?:의)?\s*주제(?:는|가)\s*(.+?)(?:\s*(?:이야|입니다|이에요|야)\b|[.;\n]|$)",
    ],
}


def _first_match(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            val = re.split(r"[\r\n]+", val)[0].strip()
            val = val.rstrip(" ;,.")
            return val if val else None
    return None


def parse_user_fields(user_input: str) -> Tuple[str, str, str]:
    책임자 = _first_match(user_input, FIELD_PATTERNS["책임자"])
    if not 책임자:
        책임자 = _first_match(user_input, FIELD_PATTERNS["feature_2"]) or ""
    연구기관명 = _first_match(user_input, FIELD_PATTERNS["연구기관명"]) or ""
    if not 연구기관명:
        연구기관명 = _first_match(user_input, FIELD_PATTERNS["feature_1"]) or ""
    # 1순위: 과제명 라인이 있으면 그걸 사용
    # 2순위: 없으면 연구주제(=feature_3)로 대체
    과제명 = _first_match(user_input, FIELD_PATTERNS["과제명"])
    if not 과제명:
        과제명 = _first_match(user_input, FIELD_PATTERNS["연구주제"]) or ""
    return 책임자, 연구기관명, 과제명


# ---------- Parsing assistant_output JSON ----------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_assistant_json(assistant_output: str) -> Tuple[str, str]:
    s = _strip_code_fences(assistant_output)

    start = s.find("{")
    end = s.rfind("}")
    candidate = s[start:end + 1].strip() if (start != -1 and end != -1 and end > start) else s

    email = ""
    conf_score = ""

    try:
        obj = json.loads(candidate)
        if isinstance(obj.get("email"), str):
            email = obj["email"].strip()
        if obj.get("confidence_score") is not None:
            conf_score = str(float(obj["confidence_score"]))
        return email, conf_score
    except Exception:
        # regex fallback
        em = re.search(r'"email"\s*:\s*"([^"]*)"', candidate)
        cm = re.search(r'"confidence_score"\s*:\s*([0-9]*\.?[0-9]+)', candidate)
        if em:
            email = em.group(1).strip()
        if cm:
            conf_score = cm.group(1)
        return email, conf_score


# ---------- CLI ----------
def parse_inputs_arg(inputs_arg: str) -> List[str]:
    """
    Accept:
    - JSON array string: '["resp_..","resp_.."]'
    - Comma-separated: 'resp_..,resp_..'
    - File reference: '@ids.json' (file contains JSON array)
    """
    s = inputs_arg.strip()
    if s.startswith("@"):
        path = s[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Try JSON first
    try:
        arr = json.loads(s)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except Exception:
        pass

    # Fallback: comma-separated
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output CSV path")
    ap.add_argument("--inputs", required=True, help='JSON array string, comma list, or @file')
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between requests")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Put it in .env or environment variables.")

    ids = parse_inputs_arg(args.inputs)
    if not ids:
        raise SystemExit("No ids provided.")

    session = get_session(api_key)

    rows: List[Dict[str, str]] = []

    for i, rid in enumerate(ids, 1):
        try:
            resp = fetch_response(session, rid)
            items = fetch_all_input_items(session, rid)

            user_input = collect_user_input_text(items)
            assistant_output = collect_assistant_output_text(resp)

            책임자, 연구기관명, 과제명 = parse_user_fields(user_input)
            email, conf_score = parse_assistant_json(assistant_output)

            rows.append({
                "책임자": 책임자,
                "연구기관명": 연구기관명,
                "과제명": 과제명,
                "email": email,
                "conf_score": conf_score,
            })
            print(f"[{i}/{len(ids)}] OK {rid}")
        except Exception as e:
            rows.append({
                "책임자": "",
                "연구기관명": "",
                "과제명": "",
                "email": "",
                "conf_score": "",
            })
            print(f"[{i}/{len(ids)}] FAIL {rid}: {e}")

        time.sleep(args.sleep)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["책임자", "연구기관명", "과제명", "email", "conf_score"])
        w.writeheader()
        w.writerows(rows)

    print("wrote:", args.output)


if __name__ == "__main__":
    main()