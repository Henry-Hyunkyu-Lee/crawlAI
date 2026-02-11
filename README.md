# AI 크롤링 Streamlit UI

Streamlit 기반 UI에서 파일 업로드, 피처 컬럼 선택, 크롤링 실행을 수행합니다.
`uv run main.py`만으로 Streamlit이 자동 실행되도록 구성되어 있습니다.

## 요구 사항

- Python 3.10+
- uv

## uv 설치

공식 설치 가이드: https://docs.astral.sh/uv/

예시(Windows PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

또는 기타 설치 예시:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 확인:

```
uv --version
```

## 실행

```bash
uv run main.py
```

브라우저가 열리지 않으면 터미널에 표시되는 로컬 URL로 접속하세요.

## 사용 방법

1. 사이드바에서 CSV/XLSX 파일을 업로드합니다.
2. `입력 컬럼 설정`에서 표준 필드 매핑을 지정합니다.
    - 성명, 회사명, 부서명, 직책명, 과제명
    - 회사명은 필수이며, 없는 필드는 `(없음)`으로 두면 빈값 처리됩니다.
3. `중복 제거 기준 컬럼`을 선택해 전처리 기준을 설정합니다.
4. 미리보기에서 데이터 형태를 확인합니다.
5. 실행 옵션에서 범위/Provider/모델/API Key를 지정합니다.
6. `출력 컬럼 선택`에서 필요한 컬럼만 선택합니다.
    - 제외한 컬럼은 프롬프트 요청/모델 응답 계약/최종 저장에서 모두 제외됩니다.
    - `email`, `confidence_score`는 필수 컬럼으로 항상 포함됩니다.
7. `프롬프트 태스크 시작` 버튼으로 실행합니다.

## 프롬프트

- 프롬프트는 선택한 출력 컬럼을 반영해 동적으로 생성됩니다.
- 입력값이 있는 항목은 그대로 사용하고, 빈 항목만 웹검색으로 보완하도록 지시합니다.
- 모델 응답은 JSON 객체로 강제됩니다.

## 출력

- 결과는 지정한 출력 파일명으로 저장됩니다.
- 기본 파일명은 results.csv입니다.
- 출력 컬럼 기본 목록은 아래와 같고, 실행 시 필요한 컬럼만 선택해 저장할 수 있습니다.
  - `성명`, `회사명`, `부서명`, `직책명`, `과제명`, `email`, `confidence_score`, `status`, `error_code`
- `status`/`error_code`는 모델 응답이 아닌 시스템 실행 상태로 기록됩니다.

## 주의 사항

- 실행 중 세션이 끊기면 결과가 저장되지 않을 수 있습니다.
- API Key는 저장되지 않으며 입력값은 노출되지 않도록 주의하세요.

## 환경변수(.env)

- OpenAI: `OPENAI_API_KEY`
- Gemini: `GEMINI_API_KEY`
- 업데이트 대상 저장소: `APP_REPO` (예: `owner/repo`)
- GitHub API 토큰(선택): `GITHUB_TOKEN`

## 소프트웨어 업데이트

1. 사이드바의 `소프트웨어 업데이트` 섹션에서 `GitHub Repo (owner/repo)`를 입력합니다.
2. `최신 버전 확인`을 눌러 현재 버전과 최신 릴리즈 버전을 비교합니다.
3. 새 버전이 있으면 `업데이트 실행` 버튼이 활성화됩니다.
4. 업데이트 실행 시 백업 후 최신 소스를 반영하고, 신버전에 없는 오래된 파일을 자동 삭제한 뒤 `uv sync` 후 앱을 재실행합니다.
