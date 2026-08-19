"""Request configuration contracts for the Gemini adapter.

These tests cover repository-controlled request configuration: response
format, response schema, tool selection, and thinking levels. Prompt
wording is not part of the contract and is not asserted.
"""

from google.genai import types

from modules import core, gemini


def test_content_config_always_disables_automatic_function_calling():
    config = gemini.build_content_config(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
    )

    assert config.automatic_function_calling.disable is True


def test_generation_config_requires_structured_json_without_tools():
    config = gemini.generate_content_config("high")

    assert config.response_mime_type == "application/json"
    assert config.response_schema is core.SubtitleResponse
    assert config.automatic_function_calling.disable is True
    assert config.tools is None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert config.temperature == 0.0


def test_generation_config_omits_thinking_config_when_level_is_none():
    config = gemini.generate_content_config(None)

    assert config.thinking_config is None


def test_research_config_always_enables_google_search_as_plain_text():
    config = gemini.build_research_config("medium", [])

    tools = config.tools
    assert any(tool.google_search is not None for tool in tools)
    assert all(tool.url_context is None for tool in tools)
    assert config.response_mime_type is None
    assert config.response_schema is None
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert config.temperature == 0.0


def test_research_config_enables_url_context_only_for_ordinary_urls():
    config = gemini.build_research_config("medium", ["https://example.com/notes"])

    tools = config.tools
    assert any(tool.google_search is not None for tool in tools)
    assert any(tool.url_context is not None for tool in tools)


def test_youtube_analysis_config_is_plain_text_without_tools():
    config = gemini.build_youtube_analysis_config("low")

    assert config.tools is None
    assert config.response_mime_type is None
    assert config.response_schema is None
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.temperature == 0.0


def test_refinement_config_requires_structured_json_without_tools():
    config = gemini.build_refinement_config("medium")

    assert config.response_mime_type == "application/json"
    assert config.response_schema is core.RefinementResponse
    assert config.automatic_function_calling.disable is True
    assert config.tools is None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


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
