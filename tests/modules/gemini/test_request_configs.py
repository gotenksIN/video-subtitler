"""Gemini client boundary translation."""

import pytest
from google.genai import types

from modules import gemini


@pytest.mark.parametrize(
    ("api_key", "base_url", "expected"),
    [
        (
            "secret-key",
            "https://proxy.example.com",
            {
                "api_key": "secret-key",
                "http_options": {
                    "base_url": "https://proxy.example.com",
                    "retry_options": types.HttpRetryOptions(),
                },
            },
        ),
        (None, None, {"http_options": {"retry_options": types.HttpRetryOptions()}}),
    ],
    ids=["forwards key and proxy base url", "always configures retry options"],
)
def test_client_creation_forwards_credentials_and_configures_retries(
    monkeypatch, api_key, base_url, expected
):
    received = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(gemini.genai, "Client", RecordingClient)

    gemini.create_client(api_key, base_url)

    assert received == expected
