import asyncio
import json
import logging
import os
import aiohttp
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Ollama API не отвечает или нужная модель не загружена."""


class DialogState(str, Enum):
    INTRO = "INTRO"
    QUALIFY = "QUALIFY"
    VALUE = "VALUE"
    OBJECTION = "OBJECTION"
    CLOSE = "CLOSE"


class LLMResponse(BaseModel):
    response: str
    next_state: DialogState


# ─── Промпты для каждого стейта ───

SYSTEM_PROMPT = """Ты - профессиональный, но дружелюбный телемаркетолог компании ТехПро.
Твоя задача: вести диалог, достичь согласия и перевести на специалиста.

ПРАВИЛА РЕЧИ:
- Реплики короткие (до 10 слов за мысль)
- Используй вводные слова и разговорную речь: "ну", "знаете", "слушайте", "вот"
- Избегай длинных монологов
- Будь естественным, как живой человек

ФОРМАТ ОТВЕТА — строго JSON:
{"response": "твой ответ клиенту", "next_state": "INTRO|QUALIFY|VALUE|OBJECTION|CLOSE"}
"""

STATE_INSTRUCTIONS = {
    DialogState.INTRO: "Стадия INTRO. Поздоровайся, представься (Дмитрий, компания ТехПро). Коротко узнай, удобно ли говорить. next_state→QUALIFY если удобно.",
    DialogState.QUALIFY: "Стадия QUALIFY. Задай ОДИН короткий вопрос: чем занимается компания или есть ли отдел продаж. Если ответ позитивный→VALUE. Негативный→OBJECTION.",
    DialogState.VALUE: "Стадия VALUE. Кратко назови ценность: 'ИИ-помощник для звонков, экономит 50% времени менеджеров'. Если интерес→CLOSE. Нет→OBJECTION.",
    DialogState.OBJECTION: "Стадия OBJECTION. Мягко ответь на возражение. Покажи понимание. Предложи 'просто посмотреть за 5 минут'. Если согласие→CLOSE.",
    DialogState.CLOSE: "Стадия CLOSE. Предложи конкретное время для созвона с техническим специалистом. Поблагодари за время.",
}


class DialogManager:
    def __init__(
        self,
        llm_api_url: str | None = None,
        model: str | None = None,
        request_timeout_sec: float | None = None,
    ):
        # Параметры берём из ENV, чтобы не править код при смене Ollama-хоста.
        self.llm_api_url = llm_api_url or os.getenv(
            "TELEMARKETER_OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        self.model = model or os.getenv("TELEMARKETER_OLLAMA_MODEL", "qwen2.5:1.5b")
        self.request_timeout_sec = float(
            request_timeout_sec
            if request_timeout_sec is not None
            else os.getenv("TELEMARKETER_OLLAMA_TIMEOUT_SEC", "30")
        )
        logger.info(
            "ИНФО: DialogManager инициализирован [model=%s, url=%s, timeout=%.1fs]",
            self.model,
            self.llm_api_url,
            self.request_timeout_sec,
        )

    async def healthcheck(self) -> None:
        """Проверка доступности Ollama и наличия нужной модели.

        Поднимает OllamaUnavailableError, если /api/tags не отвечает или модель
        отсутствует — лучше упасть на старте, чем на первом звонке клиента.
        """
        # /api/generate → /api/tags для пинга
        base = self.llm_api_url.rsplit("/api/", 1)[0]
        tags_url = f"{base}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    tags_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        raise OllamaUnavailableError(
                            f"Ollama /api/tags вернул статус {resp.status}"
                        )
                    data = await resp.json()
        except aiohttp.ClientError as exc:
            raise OllamaUnavailableError(
                f"Не удалось подключиться к Ollama по адресу {tags_url}: {exc}"
            ) from exc
        except Exception as exc:
            raise OllamaUnavailableError(f"Healthcheck Ollama не удался: {exc}") from exc

        installed = {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}
        # Сверяем ИМЕННО полное имя (включая тег), потому что /api/generate
        # вызывается с self.model дословно. Совпадение по базе (qwen2.5:7b
        # вместо qwen2.5:1.5b) пропускает healthcheck, но runtime запрос
        # упадёт с 404 model not found — недопустимо.
        if self.model not in installed:
            raise OllamaUnavailableError(
                f"Модель {self.model} не загружена в Ollama. "
                f"Запустите: ollama pull {self.model}. "
                f"Доступные: {sorted(installed)}"
            )
        logger.info("ИНФО: Ollama доступна, модель %s загружена", self.model)

    async def generate_response(self, user_text: str, current_state: DialogState) -> LLMResponse:
        state_instruction = STATE_INSTRUCTIONS.get(current_state, "")
        prompt = f"{SYSTEM_PROMPT}\n\n{state_instruction}\n\nСлова клиента: \"{user_text}\"\nТекущее состояние: {current_state.value}\n\nОтветь строго в JSON:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_gpu": 1,        # Используем GPU (1.5B модель занимает < 1GB)
                "temperature": 0.7,
                "num_predict": 100,  # Короткие ответы для телефонии
            },
        }

        logger.info(
            "ИНФО: LLM-запрос [state=%s, text='%s']",
            current_state.value,
            user_text[:50],
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.llm_api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout_sec),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    raw = data.get("response", "{}")
                    logger.info("ОТЛАДКА: LLM-ответ [raw=%s]", raw[:200])

                    parsed = json.loads(raw)
                    next_state_str = parsed.get("next_state", current_state.value).upper()
                    if next_state_str not in [s.value for s in DialogState]:
                        next_state_str = current_state.value

                    return LLMResponse(
                        response=parsed.get("response", "Извините, повторите?"),
                        next_state=DialogState(next_state_str),
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(
                "ОШИБКА: LLM недоступен [url=%s, error=%s]",
                self.llm_api_url,
                exc,
                exc_info=True,
            )
            return LLMResponse(
                response="Минуточку, плохо слышно... Повторите, пожалуйста?",
                next_state=current_state,
            )
        except json.JSONDecodeError as exc:
            logger.error(
                "ОШИБКА: LLM вернул невалидный JSON [error=%s]", exc, exc_info=True
            )
            return LLMResponse(
                response="Прошу прощения, можете уточнить?",
                next_state=current_state,
            )
        except Exception as exc:
            logger.error("ОШИБКА: непредвиденный сбой LLM [error=%s]", exc, exc_info=True)
            return LLMResponse(
                response="Ой, секундочку... Повторите, пожалуйста?",
                next_state=current_state,
            )
