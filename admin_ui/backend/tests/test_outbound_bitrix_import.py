import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from api.outbound import (  # noqa: E402
    _bitrix_extract_phone_values,
    _bitrix_records_to_csv_bytes,
    _normalize_bitrix_webhook_base,
)


def test_normalize_bitrix_webhook_base_accepts_direct_method_url(monkeypatch) -> None:
    monkeypatch.delenv("AAVA_HTTP_TOOL_TEST_ALLOW_PRIVATE", raising=False)
    monkeypatch.setattr(
        "api.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("8.8.8.8", 443))],
    )
    normalized = _normalize_bitrix_webhook_base(
        "https://example.bitrix24.ru/rest/7/secret-token/crm.lead.list.json"
    )
    assert normalized == "https://example.bitrix24.ru/rest/7/secret-token/"


def test_normalize_bitrix_webhook_base_rejects_missing_rest_path(monkeypatch) -> None:
    monkeypatch.delenv("AAVA_HTTP_TOOL_TEST_ALLOW_PRIVATE", raising=False)
    with pytest.raises(HTTPException) as exc:
        _normalize_bitrix_webhook_base("https://example.bitrix24.ru/crm/")
    assert exc.value.status_code == 400


def test_bitrix_extract_phone_values_deduplicates_and_preserves_labels() -> None:
    phones = _bitrix_extract_phone_values(
        {
            "PHONE": [
                {"VALUE": "+15550001111", "VALUE_TYPE": "WORK"},
                {"VALUE": "+15550001111", "VALUE_TYPE": "MOBILE"},
                {"VALUE": "8 (999) 000-22-33", "VALUE_TYPE": "MOBILE"},
            ],
            "PHONE_WORK": "+15550003333",
        }
    )

    assert phones == [
        {"value": "+15550001111", "label": "WORK"},
        {"value": "8 (999) 000-22-33", "label": "MOBILE"},
        {"value": "+15550003333", "label": "PHONE_WORK"},
    ]


def test_bitrix_records_to_csv_bytes_expands_each_phone_into_row() -> None:
    csv_bytes, row_count = _bitrix_records_to_csv_bytes(
        webhook_base="https://example.bitrix24.ru/rest/7/secret-token/",
        entity_type="lead",
        context_override="sales-agent",
        records=[
            {
                "ID": "101",
                "TITLE": "Warm lead",
                "STATUS_ID": "NEW",
                "ASSIGNED_BY_ID": "55",
                "PHONE": [
                    {"VALUE": "+15550001111", "VALUE_TYPE": "WORK"},
                    {"VALUE": "+15550002222", "VALUE_TYPE": "MOBILE"},
                ],
            }
        ],
    )

    csv_text = csv_bytes.decode("utf-8")
    assert row_count == 2
    assert "Warm lead" in csv_text
    assert "+15550001111" in csv_text
    assert "+15550002222" in csv_text
    assert "sales-agent" in csv_text
    assert "bitrix24" in csv_text
    assert "/crm/lead/details/101/" in csv_text
