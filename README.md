# AI Mail Collector (Streamlit)

콜드 메일 발송용 이메일 리서치를 위한 Streamlit 앱입니다.

## 빠른 시작

```bash
uv sync
uv run main.py
```

필수/선택 환경변수(`.env`):

- `OPENAI_API_KEY` (OpenAI 사용 시)
- `GEMINI_API_KEY` (Gemini 사용 시)
- `APP_REPO` (선택, 업데이트 대상 저장소)
- `GITHUB_TOKEN` (선택, GitHub API 한도 완화)

## 핵심 워크플로 (3단계)

### 1) 데이터 준비

사이드바에서 아래 항목을 설정합니다.

- `파일 업로드 (csv/xlsx)`
- `회사명 입력 컬럼` (필수)
- `웹사이트/도메인 컬럼 (선택)`
- `추가 컨텍스트 컬럼 (선택)`
- `미리보기/프롬프트 갱신` (수동 강제 갱신 버튼)

### 2) 프롬프트 작성

- 기본 프롬프트는 `test.py`의 `BASE_PROMPT`를 사용합니다.
- 템플릿 선택 기능은 제공하지 않습니다.
- 렌더링된 프롬프트는 입력 데이터 기반으로 즉시 확인할 수 있습니다.

### 3) 실행 옵션 설정

- `Provider` (`openai` / `gemini`)
- `Model`
- `API Key`
- `시작 인덱스`
- `종료 인덱스 (0이면 전체)`
- `출력 파일명`
- `중간 저장 간격(행)`
- `partial 파일에서 재개`

## 사이드바 파라미터 -> 워크플로 반영 방식

| 파라미터 | 적용 단계 | 코드 반영 위치 | 실제 영향 |
|---|---|---|---|
| 파일 업로드 | 1 | `load_data()` -> `df_raw` | 입력 원본 DataFrame 생성 |
| 회사명 입력 컬럼 | 1 | `mapping['company_col']` -> `extract_input_payload()` | 행별 회사명 입력 구성 |
| 웹사이트/도메인 컬럼 | 1 | `mapping['domain_col']` -> `extract_input_payload()` | 행별 도메인 입력 구성 |
| 추가 컨텍스트 컬럼 | 1 | `mapping['context_cols']` -> `extract_input_payload()` | 행별 컨텍스트 문자열 구성 |
| 프롬프트 | 2 | `st.session_state.prompt_template` -> `build_prompt_text()` | 모델 입력 프롬프트 본문 생성 |
| Provider | 3 | `get_info_from_openai()` 또는 `get_info_from_gemini()` | API 호출 경로 결정 |
| Model | 3 | `model_name` | 모델 식별자 결정 |
| API Key | 3 | `resolve_api_key()` | 인증 토큰 결정 |
| 시작/종료 인덱스 | 3 | `run_workflow()`의 `start_idx/end_idx` | 처리 범위 결정 |
| 출력 파일명 | 3 | `resolve_output_path()` | 저장 경로 결정 |
| 중간 저장 간격 | 3 | `autosave_every` -> `save_partial()` | N행마다 partial 저장 |
| partial 재개 | 3 | `merge_partial_results()` | 완료 행 재사용 후 나머지 처리 |

## 결과 파일 (원문 보존형)

최종 CSV는 아래 컬럼을 저장합니다.

- `input_company`
- `input_domain`
- `input_context`
- `response_id`
- `response_text`
- `source_urls`
- `response_json`
- `error`

## 앱 업데이트 (GitHub Release)

업데이트 UI는 사이드바 하단에 있습니다.

- `최신 버전 확인`
- `상태 새로고침`
- `업데이트 실행`
- `stale 상태 초기화` (상태 파일이 오래되어 실행 중으로 보일 때)

### 업데이트 상태 필드

업데이트 런타임 상태(`.update_runtime/update_status.json`)는 아래 필드를 사용합니다.

- `run_id`: 실행 식별자
- `state`: `queued` / `running` / `success` / `failed`
- `step`: 세부 단계 (`downloading_release`, `syncing_dependencies` 등)
- `progress`: 0.0~1.0 진행률
- `message`: 상태 메시지
- `error_code`: 실패 분류 코드 (`AUTH_ERROR`, `NETWORK_ERROR`, `FS_ERROR`, `DEPENDENCY_ERROR`, `UNKNOWN_ERROR`)
- `error_hint`: 사용자 조치 가이드
- `started_at`, `finished_at`, `updated_at`

### 실패 시 동작

- 파일 교체 실패 시 백업 복원 시도
- 새로 생성된 파일 정리 시도
- 상태 파일에 실패 코드/힌트 기록
- 업데이트 락(`.update_runtime/update.lock`) 해제

## 스모크 체크

```bash
uv run python -c "import main, updater; print('IMPORT_OK')"
uv run main.py
```

## 테스트

```bash
python -m pytest -q
```
