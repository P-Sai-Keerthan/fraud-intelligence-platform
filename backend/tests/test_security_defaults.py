"""Safe security defaults added in the follow-up audit (workstream A3).

Authentication, TLS and rate limiting are deliberately NOT implemented here: they need deployment decisions
(see docs/deployment-security-requirements.md)."""

import pytest

from app import main as app_main

ORIGIN = "http://localhost:5173"


def _preflight(client, method, headers):
    return client.options("/predict", headers={"Origin": ORIGIN, "Access-Control-Request-Method": method,
                                               "Access-Control-Request-Headers": headers})


def test_preflight_for_the_methods_and_headers_the_dashboard_uses_is_allowed(client):
    r = _preflight(client, "POST", "content-type")
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == ORIGIN


@pytest.mark.parametrize("method", ["DELETE", "PUT", "PATCH"])
def test_preflight_for_other_methods_is_refused(client, method):
    r = _preflight(client, method, "content-type")
    assert r.status_code == 400


def test_preflight_for_unlisted_request_headers_is_refused(client):
    assert _preflight(client, "POST", "content-type, x-api-key").status_code == 400


@pytest.mark.parametrize("path", ["/health", "/customers", "/customer/CUST_0001/profile"])
def test_api_responses_are_not_cacheable_and_not_sniffable(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"


def test_reports_and_predictions_are_not_cacheable(client, normal_txn):
    pred = client.post("/predict", json=normal_txn)
    pdf = client.post("/report/pdf", json={"transaction_id": pred.json()["transaction_id"]})
    for r in (pred, pdf):
        assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"


def test_interactive_docs_stay_available_by_default_and_are_cacheable_assets(client):
    r = client.get("/docs")
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff"
    assert "cache-control" not in r.headers or r.headers["cache-control"] != "no-store"


def test_a_declared_oversized_body_is_refused_before_it_is_read(client, monkeypatch):
    monkeypatch.setattr(app_main, "MAX_BATCH_BYTES", 10)
    monkeypatch.setattr(app_main, "MAX_REQUEST_BYTES_OVERHEAD", 10)
    r = client.post("/predict", content=b"x" * 500, headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json() == {"detail": "Request body is too large."}


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "no"])
def test_docs_can_be_switched_off(value):
    assert app_main.docs_config({"API_DOCS": value}) == {"docs_url": None, "redoc_url": None, "openapi_url": None}


@pytest.mark.parametrize("env", [{}, {"API_DOCS": "on"}, {"API_DOCS": "anything else"}])
def test_docs_default_to_on(env):
    assert app_main.docs_config(env) == {}
