import json
import re
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

GITHUB_API_BASE = "https://api.github.com"
UPDATE_RUNTIME_DIR = ".update_runtime"
UPDATE_BACKUP_DIR = ".update_backups"
UPDATE_STATUS_FILE = "update_status.json"


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


def normalize_repo(repo: str) -> str:
    value = (repo or "").strip()
    if not value:
        return ""

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
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def get_latest_release(repo: str, github_token: str = "", timeout: int = 20) -> ReleaseInfo:
    normalized_repo = normalize_repo(repo)
    url = f"{GITHUB_API_BASE}/repos/{normalized_repo}/releases/latest"
    response = requests.get(url, headers=_github_headers(github_token), timeout=timeout)

    if response.status_code == 404:
        raise ValueError("해당 저장소의 latest release를 찾을 수 없습니다.")
    if response.status_code == 403:
        raise ValueError("GitHub API 요청이 제한되었습니다. GITHUB_TOKEN을 설정하세요.")

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


def read_update_status(project_root: Path):
    status_path = get_status_path(project_root)
    if not status_path.exists():
        return None

    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_update_status(project_root: Path, status: dict):
    status_path = get_status_path(project_root)
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _powershell_update_script() -> str:
    return textwrap.dedent(
        r'''param(
    [Parameter(Mandatory=$true)][string]$ZipUrl,
    [Parameter(Mandatory=$true)][string]$WorkDir,
    [Parameter(Mandatory=$true)][string]$BackupDir,
    [Parameter(Mandatory=$true)][string]$TempDir,
    [Parameter(Mandatory=$true)][string]$StatusPath,
    [int]$TargetPid = 0,
    [string]$GitHubToken = ""
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$State, [string]$Message)
    $obj = @{
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString("o")
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

$newFiles = New-Object System.Collections.Generic.List[string]
$sourceRelativeSet = @{}

try {
    Ensure-Dir -PathValue $TempDir
    Ensure-Dir -PathValue $BackupDir

    Write-Status -State "running" -Message "update script started"

    if ($TargetPid -gt 0) {
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

    Invoke-WebRequest -Uri $ZipUrl -OutFile $zipPath -Headers $headers -UseBasicParsing
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

    Push-Location $WorkDir
    uv sync
    Pop-Location

    Start-Process -FilePath "uv" -ArgumentList @("run", "main.py") -WorkingDirectory $WorkDir

    Write-Status -State "success" -Message "update completed"
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

    Write-Status -State "failed" -Message $err
}
'''
    )


def launch_update_script(
    release: ReleaseInfo,
    project_root: Path,
    current_pid: int,
    github_token: str = "",
) -> UpdateLaunchResult:
    runtime_dir = project_root / UPDATE_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / UPDATE_BACKUP_DIR / timestamp
    temp_dir = runtime_dir / f"tmp_{timestamp}"
    status_path = runtime_dir / UPDATE_STATUS_FILE
    script_path = runtime_dir / "apply_update.ps1"

    script_path.write_text(_powershell_update_script(), encoding="utf-8")

    write_update_status(
        project_root,
        {
            "state": "queued",
            "message": f"queued update: {release.tag}",
            "updated_at": datetime.now().isoformat(),
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
        release.zip_url,
        "-WorkDir",
        str(project_root),
        "-BackupDir",
        str(backup_dir),
        "-TempDir",
        str(temp_dir),
        "-StatusPath",
        str(status_path),
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

    subprocess.Popen(
        cmd,
        cwd=str(project_root),
        close_fds=True,
        creationflags=creationflags,
    )

    return UpdateLaunchResult(
        script_path=script_path,
        status_path=status_path,
        backup_dir=backup_dir,
    )
