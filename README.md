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
2. 피처 컬럼 선택에서 사용할 컬럼을 선택합니다.
    - 사용할 컬럼은 기호를 포함하지 않은 영문 또는 한글이어야 합니다.
    - 선택된 피처 컬럼 기준으로 중복이 제거됩니다.
3. feature_1, feature_2, feature_3 컬럼을 지정합니다.
    - feature_3는 선택 사항입니다.
4. 미리보기에서 데이터가 원하는 형태인지 확인합니다.
5. 실행 옵션에서 범위를 지정합니다.
6. `AI Provider`에서 `openai` 또는 `gemini`를 선택합니다.
7. 선택한 Provider의 API Key를 입력하거나, `.env` 환경변수를 사용해 실행합니다.

## 프롬프트 템플릿

프롬프트 템플릿은 아래 변수를 사용할 수 있습니다.
요구 크롤링 작업에 맞도록 프롬프트를 수정합니다.

- {feature_1}
- {feature_2}
- {feature_3}

## 출력

- 결과는 지정한 출력 파일명으로 저장됩니다.
- 기본 파일명은 results.csv입니다.

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
4. 업데이트 실행 시 백업 후 최신 소스를 반영하고 `uv sync` 후 앱을 재실행합니다.
