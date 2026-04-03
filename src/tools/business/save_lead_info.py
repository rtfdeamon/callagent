"""In-call tool: save qualifying lead information during sales calls.

The LLM calls this tool silently when it learns a new fact about the caller's needs.
Data is stored in session.lead_data and used by the post-call report generator.
"""

from typing import Any, Dict

from ..base import Tool, ToolCategory, ToolDefinition, ToolParameter, ToolPhase
from ..context import ToolExecutionContext
from ...logging_config import get_logger

logger = get_logger(__name__)

# Fields the LLM can populate
LEAD_FIELDS = [
    "organization_name",       # Название организации
    "organization_type",       # Тип: музей, школа, библиотека, ДК, вуз, театр
    "contact_name",            # ФИО контактного лица
    "contact_phone",           # Телефон
    "contact_email",           # Email для КП
    "object_type",             # Тип объекта: зал, класс, музейный зал, фойе
    "equipment_needed",        # Оборудование: панели, проекторы, звук, свет
    "project_description",     # Описание задачи свободным текстом
    "budget_range",            # Бюджет (ориентировочно)
    "timeline",                # Сроки: когда нужно
    "financing_type",          # 44-ФЗ, 223-ФЗ, грант, собственные средства
    "next_action",             # Следующий шаг: КП, встреча, перезвонить
]


class SaveLeadInfoTool(Tool):
    """Silently save a piece of qualifying information to the call session."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_lead_info",
            description=(
                "Сохрани информацию о клиенте, когда узнаёшь новый факт. "
                "Вызывай для каждого нового факта отдельно. Не упоминай клиенту что записываешь."
            ),
            category=ToolCategory.BUSINESS,
            phase=ToolPhase.IN_CALL,
            is_global=False,
            max_execution_time=2,
            parameters=[
                ToolParameter(
                    name="field",
                    type="string",
                    description=(
                        "Поле для записи: organization_name, organization_type, "
                        "contact_name, contact_phone, contact_email, object_type, "
                        "equipment_needed, project_description, budget_range, "
                        "timeline, financing_type, next_action"
                    ),
                    required=True,
                    enum=LEAD_FIELDS,
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="Значение поля (кратко, по-русски)",
                    required=True,
                ),
            ],
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
        field_name = parameters.get("field", "")
        value = parameters.get("value", "")

        if field_name not in LEAD_FIELDS:
            logger.warning(
                "save_lead_info: unknown field",
                call_id=context.call_id,
                field=field_name,
            )
            return {"status": "error", "message": "", "silent": True}

        if not value:
            return {"status": "error", "message": "", "silent": True}

        try:
            session = await context.get_session()
            lead = dict(session.lead_data) if session.lead_data else {}
            lead[field_name] = value
            await context.update_session(lead_data=lead)

            logger.info(
                "📋 LEAD INFO saved",
                call_id=context.call_id,
                field=field_name,
                value=value,
                filled=len([v for v in lead.values() if v]),
                total=len(LEAD_FIELDS),
            )
        except Exception as exc:
            logger.error("save_lead_info failed", call_id=context.call_id, error=str(exc))
            return {"status": "error", "message": "", "silent": True}

        return {"status": "success", "message": "", "silent": True}
