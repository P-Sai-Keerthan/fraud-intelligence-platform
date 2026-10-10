"""CORS: allowed browser origins come from the CORS_ALLOW_ORIGINS env var."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import DEFAULT_CORS_ALLOW_ORIGINS, cors_allow_origins

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEV_ORIGIN = "http://localhost:5173"
OTHER_ORIGIN = "https://not-allowed.example.com"


# ---- parsing -------------------------------------------------------------------

def test_default_is_local_vite_dev_and_preview():
    assert cors_allow_origins("") == DEFAULT_CORS_ALLOW_ORIGINS
    assert "http://localhost:5173" in DEFAULT_CORS_ALLOW_ORIGINS
    assert "http://127.0.0.1:5173" in DEFAULT_CORS_ALLOW_ORIGINS
    assert "*" not in DEFAULT_CORS_ALLOW_ORIGINS


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://a.example.com", ["https://a.example.com"]),
        ("https://a.example.com, https://b.example.com", ["https://a.example.com", "https://b.example.com"]),
        (" https://a.example.com/ ,,", ["https://a.example.com"]),
        ("*", ["*"]),
        ("   ", DEFAULT_CORS_ALLOW_ORIGINS),
    ],
)
def test_parse_origins(raw, expected):
    assert cors_allow_origins(raw) == expected


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://dash.example.com")
    assert cors_allow_origins() == ["https://dash.example.com"]
    monkeypatch.delenv("CORS_ALLOW_ORIGINS")
    assert cors_allow_origins() == DEFAULT_CORS_ALLOW_ORIGINS


# ---- the running app (default configuration) ------------------------------------

def _preflight(client, origin):
    return client.options("/predict", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })


def test_preflight_from_local_dev_origin_allowed(client):
    r = _preflight(client, DEV_ORIGIN)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert r.headers["access-control-allow-credentials"] == "true"


def test_preflight_from_other_origin_rejected(client):
    r = _preflight(client, OTHER_ORIGIN)
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_simple_request_headers(client):
    allowed = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    other = client.get("/health", headers={"Origin": OTHER_ORIGIN})
    assert other.status_code == 200  # CORS is enforced by the browser, not by refusing the request
    assert "access-control-allow-origin" not in other.headers


def test_same_origin_requests_unaffected(client):
    # the Vite dev proxy forwards requests without a cross-origin Origin header
    r = client.get("/health")
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


# ---- the environment variable reaches the middleware ----------------------------

def _middleware_config_with_env(value):
    """Imports the app in a fresh interpreter with CORS_ALLOW_ORIGINS set and
    reports the CORS middleware's configuration (no server/models started)."""
    env = {**os.environ, "CORS_ALLOW_ORIGINS": value, "DATABASE_URL": "sqlite://"}
    code = (
        "from app.main import app\n"
        "from fastapi.middleware.cors import CORSMiddleware\n"
        "m = next(m for m in app.user_middleware if m.cls is CORSMiddleware)\n"
        "print(repr((m.kwargs['allow_origins'], m.kwargs['allow_credentials'])))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, env=env,
        capture_output=True, text=True, timeout=180, check=True,
    )
    return eval(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_env_var_configures_app_middleware():
    origins, credentials = _middleware_config_with_env("https://dash.example.com, https://ops.example.com/")
    assert origins == ["https://dash.example.com", "https://ops.example.com"]
    assert credentials is True


@pytest.mark.slow
def test_wildcard_disables_credentials():
    origins, credentials = _middleware_config_with_env("*")
    assert origins == ["*"]
    assert credentials is False
