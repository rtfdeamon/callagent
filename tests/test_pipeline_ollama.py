import pytest

from src.pipelines.ollama import OllamaLLMAdapter


@pytest.mark.asyncio
async def test_ollama_open_call_marks_unsupported_tool_model():
    adapter = OllamaLLMAdapter(app_config=None, pipeline_defaults={})

    await adapter.open_call(
        "call-1",
        {
            "model": "t-lite-it-2.1",
            "tools": ["save_lead_info"],
            "tools_enabled": True,
        },
    )

    assert adapter._sessions["call-1"]["tools_requested_but_unsupported"] is True


@pytest.mark.asyncio
async def test_ollama_open_call_keeps_tool_capable_model_enabled():
    adapter = OllamaLLMAdapter(app_config=None, pipeline_defaults={})

    await adapter.open_call(
        "call-2",
        {
            "model": "qwen2.5:7b",
            "tools": ["save_lead_info"],
            "tools_enabled": True,
        },
    )

    assert adapter._sessions["call-2"]["tools_requested_but_unsupported"] is False
