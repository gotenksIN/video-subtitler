"""Gemini client boundary translation."""

from modules import gemini


def test_client_creation_forwards_key_and_proxy_base_url(monkeypatch):
    received = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(gemini.genai, "Client", RecordingClient)

    gemini.create_client("secret-key", "https://proxy.example.com")

    assert received == {
        "api_key": "secret-key",
        "http_options": {"base_url": "https://proxy.example.com"},
    }


def test_client_creation_omits_unset_options(monkeypatch):
    received = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(gemini.genai, "Client", RecordingClient)

    gemini.create_client(None, None)

    assert received == {}
