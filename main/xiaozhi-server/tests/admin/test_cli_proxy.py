import os

import pytest

from core.admin import config_store
from core.agent import cli_proxy


def test_defaults_toggle_and_restore(tmp_path, monkeypatch):
    path = tmp_path / "user.yaml"
    monkeypatch.setattr(config_store, "custom_config_path", lambda: path)
    monkeypatch.setenv("HTTPS_PROXY", "http://inherited:1080")
    assert cli_proxy.settings({}) == {"enabled": False, "url": "http://127.0.0.1:7890"}
    assert cli_proxy.child_env()["HTTPS_PROXY"] == os.environ["HTTPS_PROXY"]
    value, warning = cli_proxy.save({"enabled": True, "url": "127.0.0.1:1"}, path)
    assert warning
    env = cli_proxy.child_env()
    assert env["https_proxy"] == env["HTTP_PROXY"] == value["url"]
    assert "localhost" in env["no_proxy"]
    assert os.environ["HTTPS_PROXY"] == "http://inherited:1080"
    cli_proxy.save(dict(value, enabled=False), path)
    assert cli_proxy.settings(config_store.load_custom())["url"] == value["url"]
    assert cli_proxy.overrides() == {}


@pytest.mark.parametrize("url", ["", "file:///tmp/test", "https://proxy:65536", "http://proxy:80/path"])
def test_invalid_url_creates_no_config(tmp_path, url):
    path = tmp_path / "user.yaml"
    with pytest.raises(ValueError):
        cli_proxy.save({"enabled": True, "url": url}, path)
    assert not path.exists()
