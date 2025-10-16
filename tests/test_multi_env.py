import os
import sys
import importlib


def _clear_src_modules():
    """Ensure src package reloads with new environment."""
    for name in list(sys.modules.keys()):
        if name == "src" or name.startswith("src."):
            sys.modules.pop(name, None)


def _load_app():
    app_module = importlib.import_module("src.main")
    return app_module.app


def test_env_selection_via_env_var(tmp_path, monkeypatch):
    # Create a temporary env file
    env1 = tmp_path / ".env.one"
    env1.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test-1",
                "OPENAI_BASE_URL=http://example1/v1",
                "BIG_MODEL=gpt-test-big-1",
                "SMALL_MODEL=gpt-test-small-1",
            ]
        )
    )

    # Point loader to this file via environment variable
    monkeypatch.setenv("CLAUDE_PROXY_ENV_FILE", str(env1))

    # Reload src to pick up new env
    _clear_src_modules()
    app = _load_app()

    # Use TestClient to call root
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["openai_base_url"] == "http://example1/v1"


def test_env_selection_via_cli_flag(tmp_path, monkeypatch):
    # Ensure env var is not set
    monkeypatch.delenv("CLAUDE_PROXY_ENV_FILE", raising=False)

    # Create a second env file
    env2 = tmp_path / ".env.two"
    env2.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test-2",
                "OPENAI_BASE_URL=http://example2/v1",
                "BIG_MODEL=gpt-test-big-2",
                "SMALL_MODEL=gpt-test-small-2",
            ]
        )
    )

    # Simulate CLI flag
    monkeypatch.setattr(sys, "argv", ["pytest", "--env-file", str(env2)])

    # Reload src to pick up CLI args
    _clear_src_modules()
    app = _load_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["openai_base_url"] == "http://example2/v1"

