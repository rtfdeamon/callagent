"""Post-call tool: generate lead report (Заявка + Опросный лист) from call data.

Runs after the call ends. Merges in-call lead_data (from save_lead_info tool)
with LLM-extracted facts from the transcript, then formats and logs the report.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from ..base import PostCallTool, ToolCategory, ToolDefinition, ToolPhase
from ..context import PostCallContext
from ...logging_config import get_logger

logger = get_logger(__name__)

# Same fields as save_lead_info
_LEAD_FIELDS = {
    "organization_name": "Организация",
    "organization_type": "Тип учреждения",
    "contact_name": "Контактное лицо",
    "contact_phone": "Телефон",
    "contact_email": "Email",
    "object_type": "Тип объекта",
    "equipment_needed": "Оборудование",
    "project_description": "Описание задачи",
    "budget_range": "Бюджет",
    "timeline": "Сроки",
    "financing_type": "Финансирование",
    "next_action": "Следующий шаг",
}

_EXTRACTION_PROMPT = """Ты анализируешь транскрипт телефонного звонка менеджера по продажам компании MMVS.
Извлеки факты о клиенте из диалога. Верни ТОЛЬКО JSON без пояснений.

Поля (оставь null если информация не прозвучала):
{
  "organization_name": "название организации клиента",
  "organization_type": "тип: музей, школа, библиотека, ДК, вуз, театр, стадион, офис",
  "contact_name": "ФИО контактного лица",
  "contact_phone": "телефон",
  "contact_email": "email",
  "object_type": "тип объекта: зал, класс, музейный зал, фойе, стадион",
  "equipment_needed": "какое оборудование нужно",
  "project_description": "краткое описание задачи",
  "budget_range": "бюджет",
  "timeline": "сроки",
  "financing_type": "44-ФЗ, 223-ФЗ, грант, собственные средства",
  "next_action": "следующий шаг: КП, встреча, перезвонить"
}

ТРАНСКРИПТ:
"""


def _build_transcript_text(history: List[Dict[str, Any]]) -> str:
    """Convert conversation_history to readable transcript."""
    lines = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or role not in ("user", "assistant"):
            continue
        speaker = "Клиент" if role == "user" else "Менеджер"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _format_zayavka(lead: Dict[str, Any], call_id: str) -> str:
    """Format the lead application (Заявка)."""
    date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    short_id = call_id.split(".")[-1] if "." in call_id else call_id[-6:]
    lines = [
        f"{'═' * 50}",
        f"  ЗАЯВКА №{short_id} от {date_str}",
        f"{'═' * 50}",
    ]
    for field_key, label in _LEAD_FIELDS.items():
        value = lead.get(field_key) or "—"
        lines.append(f"  {label}: {value}")
    lines.append(f"{'═' * 50}")
    return "\n".join(lines)


def _format_oprosny_list(lead: Dict[str, Any]) -> str:
    """Format the questionnaire (Опросный лист)."""
    lines = [
        f"{'─' * 50}",
        "  ОПРОСНЫЙ ЛИСТ",
        f"{'─' * 50}",
    ]
    filled = 0
    total = len(_LEAD_FIELDS)
    for field_key, label in _LEAD_FIELDS.items():
        value = lead.get(field_key)
        if value:
            lines.append(f"  [✓] {label}: {value}")
            filled += 1
        else:
            lines.append(f"  [ ] {label}: не указано")
    pct = round(filled / total * 100) if total else 0
    lines.append(f"{'─' * 50}")
    lines.append(f"  Заполнено: {filled}/{total} ({pct}%)")
    lines.append(f"{'─' * 50}")
    return "\n".join(lines)


async def _extract_from_transcript(
    transcript: str,
    ollama_url: str = "http://127.0.0.1:11434",
    model: str = "qwen2.5:7b",
) -> Dict[str, Any]:
    """Use LLM to extract lead fields from transcript (post-call, no latency constraint)."""
    if not transcript.strip():
        return {}
    prompt = _EXTRACTION_PROMPT + transcript
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 400},
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_url}/api/chat", json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                data = await resp.json()
                raw = data.get("message", {}).get("content", "")
                # Extract JSON from response (handle markdown code blocks)
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                extracted = json.loads(raw.strip())
                return {k: v for k, v in extracted.items() if v is not None and v != ""}
    except Exception as exc:
        logger.warning("LLM extraction failed", error=str(exc))
        return {}


class GenerateLeadReportTool(PostCallTool):
    """Generate and log a structured lead report after the call ends."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="generate_lead_report",
            description="Generate lead application and questionnaire from call data",
            category=ToolCategory.BUSINESS,
            phase=ToolPhase.POST_CALL,
            is_global=True,
        )

    async def execute(self, context: PostCallContext) -> None:
        call_id = context.call_id
        logger.info("Generating lead report", call_id=call_id)

        # 1. Start with in-call collected data
        lead = dict(context.lead_data) if context.lead_data else {}

        # 2. Auto-fill contact_phone from caller_number if not set
        if not lead.get("contact_phone") and context.caller_number:
            lead["contact_phone"] = context.caller_number

        # 3. LLM extraction from transcript (fills gaps)
        transcript = _build_transcript_text(context.conversation_history)
        if transcript:
            try:
                extracted = await _extract_from_transcript(transcript)
                for key, value in extracted.items():
                    if key in _LEAD_FIELDS and not lead.get(key):
                        lead[key] = value
                logger.info(
                    "LLM extraction completed",
                    call_id=call_id,
                    extracted_fields=len(extracted),
                )
            except Exception as exc:
                logger.warning("LLM extraction failed", call_id=call_id, error=str(exc))

        # 4. Format reports
        zayavka = _format_zayavka(lead, call_id)
        oprosny = _format_oprosny_list(lead)

        # 5. Log to console (always visible in docker logs)
        filled = len([v for v in lead.values() if v and v != "—"])
        logger.info(
            "LEAD_REPORT",
            call_id=call_id,
            caller=context.caller_number,
            filled_fields=filled,
            total_fields=len(_LEAD_FIELDS),
            duration_sec=context.call_duration_seconds,
            outcome=context.call_outcome,
        )
        # Print formatted reports to stdout for immediate visibility
        print(f"\n{zayavka}\n{oprosny}\n")

        # 6. Save to file
        leads_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "leads",
        )
        os.makedirs(leads_dir, exist_ok=True)
        report_path = os.path.join(leads_dir, f"{call_id}.json")
        try:
            report_json = {
                "call_id": call_id,
                "caller_number": context.caller_number,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": context.call_duration_seconds,
                "outcome": context.call_outcome,
                "lead_data": lead,
                "transcript": context.conversation_history,
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_json, f, ensure_ascii=False, indent=2)
            logger.info("Lead report saved", call_id=call_id, path=report_path)
        except Exception as exc:
            logger.warning("Failed to save lead report file", call_id=call_id, error=str(exc))
