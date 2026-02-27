import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

from task_schema import CrawlTaskSpec, OutputFieldSpec


class ResultValidationError(ValueError):
    pass


def strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_object(raw_text: str) -> Dict:
    cleaned = strip_code_fences(raw_text)
    if not cleaned:
        raise ResultValidationError("모델 응답이 비어 있습니다.")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ResultValidationError("모델 응답에서 JSON 객체를 찾을 수 없습니다.")
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise ResultValidationError("모델 응답 JSON은 객체(dict)여야 합니다.")
    return payload


def _to_number(key: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ResultValidationError(f"{key}는 number 타입이어야 합니다.")
    return number


def _to_boolean(key: str, value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    raise ResultValidationError(f"{key}는 boolean 타입이어야 합니다.")


def _to_json_obj(key: str, value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise ResultValidationError(f"{key}는 JSON 객체여야 합니다.")
        if isinstance(parsed, dict):
            return parsed
    raise ResultValidationError(f"{key}는 JSON 객체여야 합니다.")


def _serialize_value(field_spec: OutputFieldSpec, value):
    key = field_spec.key
    if field_spec.field_type == "number":
        return _to_number(key, value)
    if field_spec.field_type == "boolean":
        return _to_boolean(key, value)
    if field_spec.field_type == "json":
        return json.dumps(_to_json_obj(key, value), ensure_ascii=False, sort_keys=True)
    return str(value or "").strip()


def _extract_urls(raw_urls: str) -> List[str]:
    if not raw_urls:
        return []
    split_tokens = re.split(r"[;\n,\s]+", raw_urls)
    urls = []
    for token in split_tokens:
        item = token.strip()
        if not item:
            continue
        if item.startswith("http://") or item.startswith("https://"):
            if item not in urls:
                urls.append(item)
    return urls


def normalize_and_validate_result(
    raw_payload: Dict,
    task_spec: CrawlTaskSpec,
    fallback_source_urls: Optional[Iterable[str]] = None,
) -> Tuple[Dict, List[str]]:
    normalized: Dict = {}
    flags: List[str] = []

    for field_spec in task_spec.response_field_specs:
        key = field_spec.key
        value = raw_payload.get(key)
        normalized[key] = _serialize_value(field_spec, value)

    for field_spec in task_spec.response_field_specs:
        key = field_spec.key
        if not field_spec.required:
            continue

        value = normalized.get(key)
        if field_spec.field_type == "string" and not str(value).strip():
            raise ResultValidationError(f"필수 문자열 키 누락: {key}")

    if "confidence_score" in normalized:
        score = _to_number("confidence_score", normalized["confidence_score"])
        if not 0 <= score <= 1:
            raise ResultValidationError("confidence_score는 0~1 범위여야 합니다.")
        normalized["confidence_score"] = score

    source_urls = list(fallback_source_urls or [])
    evidence_raw = str(normalized.get("evidence_urls", "") or "").strip()
    evidence_urls = _extract_urls(evidence_raw)

    if not evidence_urls and source_urls:
        evidence_urls = [u for u in source_urls if isinstance(u, str) and u.startswith("http")]

    if evidence_urls:
        normalized["evidence_urls"] = ";".join(evidence_urls)

    if task_spec.min_source_urls > 0 and len(evidence_urls) < task_spec.min_source_urls:
        raise ResultValidationError(
            f"근거 URL 개수 부족: required={task_spec.min_source_urls}, actual={len(evidence_urls)}"
        )

    if "field_confidence_json" in normalized:
        parsed_map = _to_json_obj("field_confidence_json", normalized["field_confidence_json"])
        sanitized_map = {}
        for k, v in parsed_map.items():
            num = _to_number(f"field_confidence_json.{k}", v)
            if not 0 <= num <= 1:
                raise ResultValidationError(f"field_confidence_json.{k}는 0~1 범위여야 합니다.")
            sanitized_map[k] = num
        normalized["field_confidence_json"] = json.dumps(
            sanitized_map, ensure_ascii=False, sort_keys=True
        )

        for expected_key in task_spec.response_keys:
            if expected_key in {"field_confidence_json", "evidence_urls", "evidence_snippets"}:
                continue
            if expected_key not in sanitized_map:
                flags.append(f"missing_confidence:{expected_key}")

    return normalized, flags
