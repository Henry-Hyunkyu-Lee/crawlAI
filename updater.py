import json
import os
import re
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

GITHUB_API_BASE = "https://api.github.com"
UPDATE_RUNTIME_DIR = ".update_runtime"
UPDATE_BACKUP_DIR = ".update_backups"
UPDATE_STATUS_FILE = "update_status.json"
UPDATE_LOCK_FILE = "update.lock"


@dataclass
class ReleaseInfo:
    repo: str
    tag: str
    version: str
    zip_url: str
    html_url: str
    published_at: Optional[str] = None


@dataclass
class UpdateStatus:
    repo: str
    local_version: Optional[str]
    latest_version: Optional[str]
    latest_tag: Optional[str]
    release_url: Optional[str]
    zip_url: Optional[str]
    update_available: bool
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class UpdateLaunchResult:
    script_path: Path
    status_path: Path
    backup_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_repo(repo: str) -> str:
    value = (repo or "").strip()
    if not value:
        raise ValueError("repository 값이 비어 있습니다. owner/repo 형식이어야 합니다.")

    value = value.replace("\\", "/")
    if value.startswith("https://github.com/"):
        value = value[len("https://github.com/") :]
    if value.startswith("http://github.com/"):
        value = value[len("http://github.com/") :]
    if value.endswith(".git"):
        value = value[:-4]
    value = value.strip("/")

    pattern = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
    if not re.match(pattern, value):
        raise ValueError("repository 형식은 owner/repo 이어야 합니다.")
    return value


def normalize_tag_to_version(tag: str) -> str:
    value = (tag or "").strip()
    if value.lower().startswith("v"):
        value = value[1:]
    if not value:
        raise ValueError("빈 태그는 버전으로 사용할 수 없습니다.")
    return value


def get_local_version(pyproject_path: Path) -> str:
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_path}")

    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^version\s*=\s*\"([^\"]+)\"", text)
    if not match:
        raise ValueError("pyproject.toml에서 version 항목을 찾을 수 없습니다.")
    return match.group(1).strip()


def _github_headers(github_token: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _extract_github_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            message = str(payload.get("message") or "").strip()
            if message:
                return message
    except Exception:
        pass

    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text[:200]
    return "unknown error"


def get_latest_release(repo: str, github_token: str = "", timeout: int = 20) -> ReleaseInfo:
    normalized_repo = normalize_repo(repo)
    url = f"{GITHUB_API_BASE}/repos/{normalized_repo}/releases/latest"
    response = requests.get(url, headers=_github_headers(github_token), timeout=timeout)

    if response.status_code == 404:
        raise ValueError("해당 저장소의 latest release를 찾을 수 없습니다.")
    if response.status_code == 401:
        raise ValueError("GitHub 인증에 실패했습니다. GITHUB_TOKEN 권한을 확인하세요.")
    if response.status_code == 403:
        details = _extract_github_error_message(response)
        raise ValueError(f"GitHub API 요청이 제한되었습니다. {details}")
    if response.status_code >= 400:
        details = _extract_github_error_message(response)
        raise ValueError(f"GitHub API 오류({response.status_code}): {details}")

    response.raise_for_status()
    payload = response.json()

    tag = payload.get("tag_name")
    zip_url = payload.get("zipball_url")
    html_url = payload.get("html_url")
    if not tag or not zip_url or not html_url:
        raise ValueError("release 응답에 필요한 필드(tag/zip/html)가 없습니다.")

    return ReleaseInfo(
        repo=normalized_repo,
        tag=tag,
        version=normalize_tag_to_version(tag),
        zip_url=zip_url,
        html_url=html_url,
        published_at=payload.get("published_at"),
    )


def _is_update_available(local_version: str, latest_version: str) -> bool:
    local = _parse_numeric_version(local_version)
    latest = _parse_numeric_version(latest_version)
    return latest > local


def _parse_numeric_version(version: str):
    """
    Minimal semantic-ish parser:
    - Accepts x, x.y, x.y.z
    - Ignores suffixes like -rc1, +build
    """
    raw = (version or "").strip()
    raw = raw.split("-", 1)[0].split("+", 1)[0]
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", raw)
    if not match:
        raise ValueError(f"지원하지 않는 버전 형식: {version}")

    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    return (major, minor, patch)


def check_for_update(repo: str, pyproject_path: Path, github_token: str = "") -> UpdateStatus:
    try:
        normalized_repo = normalize_repo(repo)
        local_version = get_local_version(pyproject_path)
        latest_release = get_latest_release(normalized_repo, github_token=github_token)
        update_available = _is_update_available(local_version, latest_release.version)

        return UpdateStatus(
            repo=normalized_repo,
            local_version=local_version,
            latest_version=latest_release.version,
            latest_tag=latest_release.tag,
            release_url=latest_release.html_url,
            zip_url=latest_release.zip_url,
            update_available=update_available,
            error=None,
        )
    except Exception as exc:
        return UpdateStatus(
            repo=(repo or "").strip(),
            local_version=None,
            latest_version=None,
            latest_tag=None,
            release_url=None,
            zip_url=None,
            update_available=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def get_status_path(project_root: Path) -> Path:
    runtime_dir = project_root / UPDATE_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / UPDATE_STATUS_FILE


def get_lock_path(project_root: Path) -> Path:
    runtime_dir = project_root / UPDATE_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / UPDATE_LOCK_FILE


def classify_update_error(message: str) -> Tuple[str, str]:
    text = str(message or "").strip().lower()
    if not text:
        return ("UNKNOWN_ERROR", "오류 메시지가 없습니다. 상태 파일을 새로고침 후 다시 시도하세요.")

    if any(token in text for token in ["401", "403", "token", "bad credentials", "unauthorized"]):
        return ("AUTH_ERROR", "GITHUB_TOKEN 또는 저장소 접근 권한을 확인한 뒤 다시 시도하세요.")
    if any(token in text for token in ["timeout", "timed out", "connection", "download", "invoke-webrequest"]):
        return ("NETWORK_ERROR", "네트워크 상태를 확인한 뒤 업데이트를 다시 시도하세요.")
    if any(token in text for token in ["access is denied", "permission", "used by another process", "cannot find path"]):
        return ("FS_ERROR", "앱/파일 점유를 해제하고 관리자 권한으로 다시 시도하세요.")
    if any(token in text for token in ["uv sync", "dependency", "module", "pip", "importerror"]):
        return ("DEPENDENCY_ERROR", "의존성 동기화에 실패했습니다. `uv sync`를 수동 실행해 확인하세요.")
    return ("UNKNOWN_ERROR", "오류 메시지를 확인하고 환경을 점검한 뒤 다시 시도하세요.")


def read_update_lock(project_root: Path) -> Optional[Dict[str, Any]]:
    lock_path = get_lock_path(project_root)
    if not lock_path.exists():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def acquire_update_lock(project_root: Path, run_id: str, force: bool = False) -> Path:
    lock_path = get_lock_path(project_root)
    if force and lock_path.exists():
        try:
            lock_path.unlink()
        except OSError:
            pass

    payload = {
        "run_id": str(run_id or "").strip(),
        "created_at": _utc_now_iso(),
        "pid": os.getpid(),
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
    except FileExistsError as exc:
        active = read_update_lock(project_root) or {}
        active_run_id = str(active.get("run_id") or "").strip()
        suffix = f" (active_run_id={active_run_id})" if active_run_id else ""
        raise RuntimeError(f"업데이트가 이미 실행 중입니다{suffix}.") from exc

    return lock_path


def release_update_lock(project_root: Path, run_id: str = "") -> None:
    lock_path = get_lock_path(project_root)
    if not lock_path.exists():
        return

    if run_id:
        active = read_update_lock(project_root) or {}
        active_run_id = str(active.get("run_id") or "").strip()
        if active_run_id and active_run_id != str(run_id).strip():
            return

    try:
        lock_path.unlink()
    except OSError:
        pass


def reset_update_runtime(project_root: Path) -> None:
    status_path = get_status_path(project_root)
    lock_path = get_lock_path(project_root)
    for path in (status_path, lock_path):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def read_update_status(project_root: Path):
    status_path = get_status_path(project_root)
    if not status_path.exists():
        return None

    try:
        raw = status_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = status_path.read_text(encoding="utf-8-sig")
        except Exception:
            return None
    except Exception:
        return None

    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")

    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_update_status(project_root: Path, status: dict):
    status_path = get_status_path(project_root)
    prev = read_update_status(project_root) or {}
    payload: Dict[str, Any] = dict(prev)
    payload.update(dict(status or {}))
    payload["updated_at"] = _utc_now_iso()

    if "progress" in payload:
        try:
            progress = float(payload["progress"])
            payload["progress"] = min(1.0, max(0.0, progress))
        except (TypeError, ValueError):
            payload.pop("progress", None)

    temp_path = status_path.with_name(f"{status_path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(status_path)


def _powershell_update_script() -> str:
    return textwrap.dedent(
        r'''param(
    [Parameter(Mandatory=$true)][string]$ZipUrl,
    [Parameter(Mandatory=$true)][string]$WorkDir,
    [Parameter(Mandatory=$true)][string]$BackupDir,
    [Parameter(Mandatory=$true)][string]$TempDir,
    [Parameter(Mandatory=$true)][string]$StatusPath,
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$LockPath,
    [int]$TargetPid = 0,
    [string]$GitHubToken = ""
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param(
        [string]$State,
        [string]$Message,
        [string]$Step = "",
        [double]$Progress = -1,
        [string]$ErrorCode = "",
        [string]$ErrorHint = "",
        [string]$FinishedAt = ""
    )

    $obj = @{}
    if (Test-Path $StatusPath) {
        try {
            $existing = Get-Content -Path $StatusPath -Raw | ConvertFrom-Json -ErrorAction Stop
            if ($existing -ne $null) {
                $existing.PSObject.Properties | ForEach-Object { $obj[$_.Name] = $_.Value }
            }
        } catch {}
    }

    $obj["run_id"] = $RunId
    $obj["state"] = $State
    $obj["message"] = $Message
    $obj["step"] = $Step
    $obj["updated_at"] = (Get-Date).ToString("o")
    if (-not $obj.ContainsKey("started_at")) {
        $obj["started_at"] = $obj["updated_at"]
    }
    if ($Progress -ge 0) {
        $obj["progress"] = [Math]::Min(1.0, [Math]::Max(0.0, $Progress))
    }
    if ($ErrorCode) {
        $obj["error_code"] = $ErrorCode
    }
    if ($ErrorHint) {
        $obj["error_hint"] = $ErrorHint
    }
    if ($FinishedAt) {
        $obj["finished_at"] = $FinishedAt
    }

    $json = $obj | ConvertTo-Json -Depth 4
    Set-Content -Path $StatusPath -Value $json -Encoding UTF8
}

function Ensure-Dir {
    param([string]$PathValue)
    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

function Is-Excluded {
    param(
        [string]$RelativePath,
        [string[]]$ExcludedTopDirs,
        [string[]]$ExcludedFiles,
        [string[]]$ExcludedGlobs
    )

    if (-not $RelativePath) { return $true }
    $normalized = $RelativePath.Replace('/', '\\')
    $firstSegment = $normalized.Split('\\')[0]

    if ($ExcludedTopDirs -contains $firstSegment) { return $true }
    if ($ExcludedFiles -contains $normalized) { return $true }

    foreach ($pattern in $ExcludedGlobs) {
        if ($normalized -like $pattern) { return $true }
    }

    return $false
}

function Classify-Error {
    param([string]$Message)
    $m = ($Message | Out-String).ToLowerInvariant()
    if ($m -match "401|403|token|bad credentials|unauthorized") {
        return @("AUTH_ERROR", "GITHUB_TOKEN 또는 저장소 권한을 확인한 뒤 다시 시도하세요.")
    }
    if ($m -match "timeout|timed out|connection|download|invoke-webrequest") {
        return @("NETWORK_ERROR", "네트워크 상태를 확인한 뒤 업데이트를 다시 시도하세요.")
    }
    if ($m -match "access is denied|permission|used by another process|cannot find path") {
        return @("FS_ERROR", "파일 점유를 해제하거나 관리자 권한으로 다시 시도하세요.")
    }
    if ($m -match "uv sync|dependency|module|importerror|health check") {
        return @("DEPENDENCY_ERROR", "의존성 또는 앱 실행 검증에 실패했습니다. `uv sync` 후 재시도하세요.")
    }
    return @("UNKNOWN_ERROR", "오류 메시지를 확인하고 환경을 점검한 뒤 다시 시도하세요.")
}

$newFiles = New-Object System.Collections.Generic.List[string]
$sourceRelativeSet = @{}

try {
    Ensure-Dir -PathValue $TempDir
    Ensure-Dir -PathValue $BackupDir

    Write-Status -State "running" -Message "update script started" -Step "start" -Progress 0.1

    if ($TargetPid -gt 0) {
        Write-Status -State "running" -Message "stopping current app process" -Step "stopping_process" -Progress 0.18
        Start-Sleep -Seconds 2
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    $zipPath = Join-Path $TempDir "release.zip"
    $extractDir = Join-Path $TempDir "extract"

    if (Test-Path $zipPath) { Remove-Item -Path $zipPath -Force }
    if (Test-Path $extractDir) { Remove-Item -Path $extractDir -Recurse -Force }

    $headers = @{}
    if ($GitHubToken) { $headers["Authorization"] = "Bearer $GitHubToken" }

    Write-Status -State "running" -Message "downloading release archive" -Step "downloading_release" -Progress 0.32
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zipPath -Headers $headers -UseBasicParsing
    Write-Status -State "running" -Message "extracting release archive" -Step "extracting_release" -Progress 0.45
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $extractedDirs = Get-ChildItem -Path $extractDir -Directory
    if ($extractedDirs.Count -eq 1) {
        $sourceRoot = $extractedDirs[0].FullName
    }
    else {
        $sourceRoot = $extractDir
    }

    $excludedTopDirs = @(".git", ".venv", "__pycache__", ".update_runtime", ".update_backups")
    $excludedFiles = @(".env")
    $excludedGlobs = @("*.csv", "*.xlsx", "*.xls", "*.parquet")

    Write-Status -State "running" -Message "applying release files" -Step "applying_files" -Progress 0.62
    $sourceFiles = Get-ChildItem -Path $sourceRoot -Recurse -File
    foreach ($src in $sourceFiles) {
        $relative = $src.FullName.Substring($sourceRoot.Length).TrimStart('\\','/')
        if (-not $relative) { continue }

        $normalizedRelative = $relative.Replace('/', '\')
        if (Is-Excluded -RelativePath $normalizedRelative -ExcludedTopDirs $excludedTopDirs -ExcludedFiles $excludedFiles -ExcludedGlobs $excludedGlobs) {
            continue
        }
        $sourceRelativeSet[$normalizedRelative] = $true

        $dest = Join-Path $WorkDir $normalizedRelative
        if (Test-Path $dest) {
            $backupFile = Join-Path $BackupDir $normalizedRelative
            Ensure-Dir -PathValue (Split-Path $backupFile -Parent)
            Copy-Item -Path $dest -Destination $backupFile -Force
        }
        else {
            $newFiles.Add($dest) | Out-Null
        }

        Ensure-Dir -PathValue (Split-Path $dest -Parent)
        Copy-Item -Path $src.FullName -Destination $dest -Force
    }

    Write-Status -State "running" -Message "removing stale files" -Step "removing_stale_files" -Progress 0.76
    # Delete stale files that are not present in the new release.
    $workFiles = Get-ChildItem -Path $WorkDir -Recurse -File
    foreach ($wf in $workFiles) {
        $relative = $wf.FullName.Substring($WorkDir.Length).TrimStart('\\','/')
        if (-not $relative) { continue }

        $normalizedRelative = $relative.Replace('/', '\')
        if (Is-Excluded -RelativePath $normalizedRelative -ExcludedTopDirs $excludedTopDirs -ExcludedFiles $excludedFiles -ExcludedGlobs $excludedGlobs) {
            continue
        }
        if ($sourceRelativeSet.ContainsKey($normalizedRelative)) {
            continue
        }

        $backupFile = Join-Path $BackupDir $normalizedRelative
        Ensure-Dir -PathValue (Split-Path $backupFile -Parent)
        Copy-Item -Path $wf.FullName -Destination $backupFile -Force
        Remove-Item -Path $wf.FullName -Force -ErrorAction Stop
    }

    Write-Status -State "running" -Message "syncing dependencies" -Step "syncing_dependencies" -Progress 0.88
    Push-Location $WorkDir
    uv sync
    & uv run python -c "import main, updater"
    if ($LASTEXITCODE -ne 0) {
        throw "health check failed after update"
    }
    Pop-Location

    Write-Status -State "running" -Message "restarting application" -Step "restarting_app" -Progress 0.95
    Start-Process -FilePath "uv" -ArgumentList @("run", "main.py") -WorkingDirectory $WorkDir

    Write-Status -State "success" -Message "update completed" -Step "success" -Progress 1.0 -FinishedAt (Get-Date).ToString("o")
}
catch {
    $err = $_.Exception.Message

    if (Test-Path $BackupDir) {
        $backupFiles = Get-ChildItem -Path $BackupDir -Recurse -File
        foreach ($bk in $backupFiles) {
            $relative = $bk.FullName.Substring($BackupDir.Length).TrimStart('\\','/')
            if (-not $relative) { continue }

            $dest = Join-Path $WorkDir $relative
            Ensure-Dir -PathValue (Split-Path $dest -Parent)
            Copy-Item -Path $bk.FullName -Destination $dest -Force
        }
    }

    foreach ($nf in $newFiles) {
        if (Test-Path $nf) {
            Remove-Item -Path $nf -Force -ErrorAction SilentlyContinue
        }
    }

    $classified = Classify-Error -Message $err
    $code = $classified[0]
    $hint = $classified[1]
    Write-Status -State "failed" -Message $err -Step "failed" -Progress 1.0 -ErrorCode $code -ErrorHint $hint -FinishedAt (Get-Date).ToString("o")
}
finally {
    if (Test-Path $LockPath) {
        Remove-Item -Path $LockPath -Force -ErrorAction SilentlyContinue
    }
}
'''
    )


def launch_update_script(
    release: ReleaseInfo,
    project_root: Path,
    current_pid: int,
    github_token: str = "",
) -> UpdateLaunchResult:
    normalized_repo = normalize_repo(release.repo)
    if not str(release.tag or "").strip():
        raise ValueError("release tag가 비어 있습니다.")
    if not str(release.version or "").strip():
        raise ValueError("release version이 비어 있습니다.")
    zip_url = str(release.zip_url or "").strip()
    if not zip_url or not zip_url.lower().startswith("https://"):
        raise ValueError("release zip_url은 https URL이어야 합니다.")

    runtime_dir = project_root / UPDATE_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{os.getpid()}"
    lock_path = acquire_update_lock(project_root, run_id=run_id, force=False)
    backup_dir = project_root / UPDATE_BACKUP_DIR / timestamp
    temp_dir = runtime_dir / f"tmp_{timestamp}"
    status_path = runtime_dir / UPDATE_STATUS_FILE
    script_path = runtime_dir / "apply_update.ps1"

    script_path.write_text(_powershell_update_script(), encoding="utf-8")

    write_update_status(
        project_root,
        {
            "run_id": run_id,
            "state": "queued",
            "message": f"queued update: {release.tag}",
            "step": "queued",
            "progress": 0.05,
            "repo": normalized_repo,
            "target_tag": release.tag,
            "target_version": release.version,
            "started_at": _utc_now_iso(),
            "error_code": "",
            "error_hint": "",
        },
    )

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-ZipUrl",
        zip_url,
        "-WorkDir",
        str(project_root),
        "-BackupDir",
        str(backup_dir),
        "-TempDir",
        str(temp_dir),
        "-StatusPath",
        str(status_path),
        "-RunId",
        run_id,
        "-LockPath",
        str(lock_path),
        "-TargetPid",
        str(current_pid),
    ]
    if github_token:
        cmd.extend(["-GitHubToken", github_token])

    creationflags = 0
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "DETACHED_PROCESS"):
        creationflags |= subprocess.DETACHED_PROCESS

    try:
        subprocess.Popen(
            cmd,
            cwd=str(project_root),
            close_fds=True,
            creationflags=creationflags,
        )
    except Exception:
        release_update_lock(project_root, run_id=run_id)
        raise

    return UpdateLaunchResult(
        script_path=script_path,
        status_path=status_path,
        backup_dir=backup_dir,
    )
