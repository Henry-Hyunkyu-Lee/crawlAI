# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Streamlit app entrypoint. Handles UI, data loading, provider selection (`openai`/`gemini`), crawl execution, and partial save/resume.
- `updater.py`: GitHub release version check and self-update workflow (download, backup, replace, relaunch).
- `against_error.py`: auxiliary CLI-style parsing/recovery script for response/error analysis.
- `pyproject.toml` + `uv.lock`: dependency and environment lock files.
- `samples/`: sample data files used for manual verification.

## Build, Test, and Development Commands
- `uv sync`: install/update dependencies from lock metadata.
- `uv run main.py`: start the Streamlit app locally.
- `uv run python -m py_compile main.py updater.py`: quick syntax validation.
- `uv run python -c "import main, updater"`: import smoke test.
- `uv run python against_error.py --help`: inspect helper script options.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, UTF-8 files, concise functions.
- Naming:
  - `snake_case` for functions/variables.
  - `UPPER_SNAKE_CASE` for constants.
  - `PascalCase` for classes (e.g., `SearchResult`).
- Keep Streamlit `session_state` keys stable and avoid reassigning keys after widget instantiation.
- Prefer small helper functions for API/network logic (`updater.py`) and keep UI concerns in `main.py`.

## Testing Guidelines
- No formal test suite yet. Use smoke checks before pushing:
  - syntax compile
  - import check
  - manual run of `uv run main.py`
- If adding non-trivial logic, create `tests/` with `pytest` and name files `test_*.py`.
- Focus first tests on version parsing/comparison and update safety paths (backup/rollback).

## Commit & Pull Request Guidelines
- Keep commit subjects short and action-oriented.
- Existing history includes release-style commits (e.g., `release: v0.2.0`); follow similar clarity.
- Recommended prefixes: `feat:`, `fix:`, `chore:`, `release:`.
- PRs should include:
  - what changed and why,
  - env/config impact (`.env`, updater behavior),
  - manual verification steps,
  - screenshots for Streamlit UI changes.

## Security & Configuration Tips
- Never hardcode API keys.
- Keep secrets in `.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, optional `APP_REPO`, `GITHUB_TOKEN`).
- Confirm `.env` remains ignored by git before pushing.
