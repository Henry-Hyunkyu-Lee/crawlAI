from pathlib import Path

import pytest

import updater
from updater import ReleaseInfo


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", raise_exc=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self._raise_exc = raise_exc

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc


def test_normalize_repo_accepts_url_and_git_suffix():
    assert updater.normalize_repo("https://github.com/foo/bar.git") == "foo/bar"


def test_normalize_repo_rejects_empty():
    with pytest.raises(ValueError):
        updater.normalize_repo("")


def test_write_read_update_status_sets_updated_at(tmp_path: Path):
    updater.write_update_status(tmp_path, {"state": "queued", "step": "queued"})
    payload = updater.read_update_status(tmp_path)
    assert payload is not None
    assert payload["state"] == "queued"
    assert isinstance(payload.get("updated_at"), str) and payload["updated_at"]


def test_read_update_status_supports_utf8_sig(tmp_path: Path):
    status_path = updater.get_status_path(tmp_path)
    status_path.write_text('{"state":"running"}', encoding="utf-8-sig")
    payload = updater.read_update_status(tmp_path)
    assert payload == {"state": "running"}


def test_classify_update_error_maps_auth():
    code, hint = updater.classify_update_error("401 Bad credentials")
    assert code == "AUTH_ERROR"
    assert "TOKEN" in hint or "토큰" in hint


def test_update_lock_lifecycle(tmp_path: Path):
    lock_path = updater.acquire_update_lock(tmp_path, run_id="r1")
    assert lock_path.exists()
    with pytest.raises(RuntimeError):
        updater.acquire_update_lock(tmp_path, run_id="r2")
    updater.release_update_lock(tmp_path, run_id="r1")
    assert not lock_path.exists()


def test_reset_update_runtime_clears_status_and_lock(tmp_path: Path):
    updater.write_update_status(tmp_path, {"state": "queued", "run_id": "r1"})
    updater.acquire_update_lock(tmp_path, run_id="r1")
    updater.reset_update_runtime(tmp_path)
    assert updater.read_update_status(tmp_path) is None
    assert not updater.get_lock_path(tmp_path).exists()


def test_get_latest_release_401_has_clear_error(monkeypatch):
    def fake_get(*args, **kwargs):
        return FakeResponse(status_code=401, payload={"message": "Bad credentials"})

    monkeypatch.setattr(updater.requests, "get", fake_get)
    with pytest.raises(ValueError) as exc:
        updater.get_latest_release("owner/repo")
    assert "인증" in str(exc.value)


def test_get_latest_release_403_includes_details(monkeypatch):
    def fake_get(*args, **kwargs):
        return FakeResponse(status_code=403, payload={"message": "API rate limit exceeded"})

    monkeypatch.setattr(updater.requests, "get", fake_get)
    with pytest.raises(ValueError) as exc:
        updater.get_latest_release("owner/repo")
    assert "rate limit" in str(exc.value).lower()


def test_launch_update_script_rejects_non_https_zip(tmp_path: Path):
    bad = ReleaseInfo(
        repo="owner/repo",
        tag="v0.2.0",
        version="0.2.0",
        zip_url="http://example.com/release.zip",
        html_url="https://example.com/release",
    )
    with pytest.raises(ValueError):
        updater.launch_update_script(bad, tmp_path, current_pid=0)


def test_launch_update_script_writes_queued_status_and_spawns(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_popen(cmd, cwd, close_fds, creationflags):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["close_fds"] = close_fds
        calls["creationflags"] = creationflags
        class Dummy:
            pass
        return Dummy()

    monkeypatch.setattr(updater.subprocess, "Popen", fake_popen)

    rel = ReleaseInfo(
        repo="owner/repo",
        tag="v0.2.1",
        version="0.2.1",
        zip_url="https://api.github.com/repos/owner/repo/zipball/v0.2.1",
        html_url="https://github.com/owner/repo/releases/tag/v0.2.1",
    )
    result = updater.launch_update_script(rel, tmp_path, current_pid=123)

    assert result.status_path.exists()
    status_payload = updater.read_update_status(tmp_path)
    assert status_payload["state"] == "queued"
    assert status_payload["target_tag"] == "v0.2.1"
    assert status_payload.get("run_id")
    assert "cmd" in calls
    assert "-ZipUrl" in calls["cmd"]
    assert "-RunId" in calls["cmd"]
    assert "-LockPath" in calls["cmd"]
