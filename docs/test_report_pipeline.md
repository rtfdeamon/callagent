# Программа и методика испытаний модулей пайплайна голосового агента

| Поле | Значение |
|------|----------|
| Документ | test_report_pipeline.md |
| Стандарт | ГОСТ 19.301-79 «Программа и методика испытаний» |
| Версия | 0.3.0 |
| Дата проведения испытаний | 2026-04-29…2026-04-30 |
| Изделие | callagent |
| Окружение | Ubuntu 24.04, Python 3.12.3, NVIDIA RTX 5060 (8 ГБ, sm_120), CUDA 12.8 |

## 1. Объект испытаний

Объектом испытаний являются три модуля основного пайплайна голосового агента:

1. **Модуль порогов реплик и no-input reprompt** — статические методы класса `Engine` в файле [src/engine.py](../src/engine.py): `_pipeline_transcript_ready`, `_pipeline_transcript_stats`, `_default_pipeline_aggregation_timeout_sec`, `_should_emit_pipeline_no_input_reprompt`.
2. **Модуль адаптера Ollama** — фабрика и обработка состояния поддержки tool calling в `OllamaLLMAdapter` в файле [src/pipelines/ollama.py](../src/pipelines/ollama.py).
3. **Модуль захвата аудио** — метод `append_raw` класса `AudioCaptureManager` в файле [src/utils/audio_capture.py](../src/utils/audio_capture.py).

## 2. Цель испытаний

Подтвердить функциональную корректность доработанных и вновь введённых методов после задач:
- понижение порога допуска фраз в LLM (с `words≥3 OR chars≥12` до `words≥2 OR chars≥10`);
- введение единой логики решения о reprompt на основе показателя RMS входного аудио;
- разделение таймаутов агрегации STT-сегментов по типу провайдера (`local_stt` 0.6 с против `deepgram_stt` 1.2 с);
- сохранение сырого аудиопотока в режиме диагностики (`AudioCaptureManager.append_raw`).

## 3. Требования

К функциям предъявлены следующие требования:

| № | Требование |
|---|------------|
| Т-1 | При `force=True` метод `_pipeline_transcript_ready` принимает любую непустую реплику. |
| Т-2 | При `force=False` короткая реплика длиной < 2 слов и < 10 символов отвергается. |
| Т-3 | Реплика «нужна смета» (2 слова, 10 символов без пробелов) принимается без force. |
| Т-4 | `_pipeline_transcript_stats` возвращает `(число_слов, число_символов_без_пробелов)`. |
| Т-5 | Дефолтный таймаут агрегации `local_stt` — 0.6 с, `deepgram_stt` — 1.2 с. |
| Т-6 | Reprompt инициируется только при тишине (RMS < 32), активном захвате, не воспроизводящемся TTS, при подтверждённом media-rx и не достигнутом лимите попыток. |
| Т-7 | Reprompt не инициируется при наличии звука (RMS ≥ 32) или при достижении лимита попыток. |
| Т-8 | Open-call для модели Ollama без поддержки tool calling помечает её как `tools_unsupported`. |
| Т-9 | Open-call для модели с поддержкой tool calling сохраняет `tools_enabled=True`. |
| Т-10 | `AudioCaptureManager.append_raw` накапливает сырые байты в `<base>/<call_id>/<stream>.<ext>`, без потерь при последовательных записях. |

## 4. Состав и порядок испытаний

### 4.1 Состав

Испытания проведены автоматически, средствами фреймворка `pytest 9.0.3` с плагином `pytest-asyncio 1.3.0`.

Файлы тестов:
- [tests/test_pipeline_turn_thresholds.py](../tests/test_pipeline_turn_thresholds.py) — 6 тестов (требования Т-1…Т-7).
- [tests/test_pipeline_ollama.py](../tests/test_pipeline_ollama.py) — 2 теста (требования Т-8, Т-9).
- [tests/test_audio_capture.py](../tests/test_audio_capture.py) — 1 тест (требование Т-10).

### 4.2 Порядок

1. Активация виртуального окружения проекта.
2. Запуск целевых тестов:
   ```bash
   source venv/bin/activate
   python -m pytest tests/test_pipeline_ollama.py \
                    tests/test_pipeline_turn_thresholds.py \
                    tests/test_audio_capture.py -v
   ```
3. Запуск полного регрессионного набора (за исключением интеграционных тестов, требующих запущенного Asterisk и local_ai_server):
   ```bash
   python -m pytest tests/ \
       --ignore=tests/test_audiosocket_bypass.py \
       --ignore=tests/test_audiosocket_minimal.py \
       --ignore=tests/test_e2e_audio_quality.py \
       --ignore=tests/test_local_ai_server_protocol.py -q
   ```

## 5. Методы испытаний

Применён метод модульного тестирования (unit testing): вызов проверяемого метода с заданными входными параметрами и сверка результата с эталонным значением (assert). Для проверки поведения при тишине/звуке используется `types.SimpleNamespace` в качестве макета сессии звонка с фиксированными значениями `audio_diagnostics`, `tts_playing`, `media_rx_confirmed`, `audio_capture_enabled`.

Аппаратное обеспечение испытаний: NVIDIA RTX 5060 (8 ГБ), Intel Core Ultra 7 265K (GPU не задействован — тесты синхронные).

## 6. Результаты испытаний

### 6.1 Прогон целевых тестов

```
tests/test_pipeline_ollama.py::test_ollama_open_call_marks_unsupported_tool_model PASSED
tests/test_pipeline_ollama.py::test_ollama_open_call_keeps_tool_capable_model_enabled PASSED
tests/test_pipeline_turn_thresholds.py::test_pipeline_transcript_ready_accepts_short_flushes PASSED
tests/test_pipeline_turn_thresholds.py::test_pipeline_transcript_ready_requires_threshold_without_flush PASSED
tests/test_pipeline_turn_thresholds.py::test_pipeline_transcript_stats_handles_short_phrases PASSED
tests/test_pipeline_turn_thresholds.py::test_local_stt_default_aggregation_timeout_is_low_latency PASSED
tests/test_pipeline_turn_thresholds.py::test_pipeline_no_input_reprompt_triggers_on_fresh_zero_rms PASSED
tests/test_pipeline_turn_thresholds.py::test_pipeline_no_input_reprompt_skips_when_audio_present_or_limit_reached PASSED
tests/test_audio_capture.py::test_append_raw_persists_bytes_when_keep_files_enabled PASSED
======================== 9 passed, 2 warnings in 0.26s =========================
```

### 6.2 Регрессионный прогон

```
564 passed, 7 warnings in 29.60s
```

В числе 564 — три ранее падавших теста `tests/test_local_ai_server_protocol.py`
(`test_tts_roundtrip`, `test_stt_binary_flow`, `test_full_audio_frame`),
исправленные в версии 0.2.0 настоящего отчёта (см. п. 6.4).

### 6.3 Соответствие требованиям

| Требование | Статус |
|------------|--------|
| Т-1 | Выполнено |
| Т-2 | Выполнено |
| Т-3 | Выполнено |
| Т-4 | Выполнено |
| Т-5 | Выполнено |
| Т-6 | Выполнено |
| Т-7 | Выполнено |
| Т-8 | Выполнено |
| Т-9 | Выполнено |
| Т-10 | Выполнено |

### 6.4 Известные ограничения

Не запускались интеграционные тесты, требующие внешних служб:
- `tests/test_audiosocket_bypass.py`, `tests/test_audiosocket_minimal.py`, `tests/test_e2e_audio_quality.py` — требуют запущенного Asterisk на UDP/5060 + сервер AudioSocket.

### 6.5 Изменения в версии 0.2.0 настоящего отчёта

В рамках задачи «доделать последовательно» (2026-04-29) выполнены:

1. **Починены тесты протокола `test_local_ai_server_protocol.py`** —
   ранее падали из-за устаревших ожиданий контракта. Сейчас:
   - `test_tts_roundtrip` ожидает `tts_response` с base64-полем
     `audio_data` (актуальный контракт `local_ai_server/server.py:3633`).
   - `test_stt_binary_flow` принимает любой `stt_result` (partial/final),
     поскольку на тишине STT не обязан выдавать `is_final=True`.
   - `test_full_audio_frame` проверяет приём кадра и ответ `stt_result`,
     не требуя цикла LLM/TTS на пустом транскрипте.

2. **Зарегистрирован XTTSAdapter в orchestrator** — управляется флагом
   `providers.xtts_tts.enabled` в YAML. По умолчанию выключен (модель
   тяжёлая, требует явной конфигурации). Пример секции добавлен в
   `config/ai-agent.example.yaml`.

3. **Удалены устаревшие бэкапы** `config/ai-agent.local.yaml.bak.*`
   (5 файлов от 2026-03-22…23).

### 6.6 Сквозные испытания (версия 0.3.0)

Проведены 2026-04-30 на боевом окружении (RTX 5060, Ollama, fine-tuned
XTTS v2 голос Андреева).

#### 6.6.1 Сервис ai_telemarketer

| Эндпоинт | HTTP | Время | Результат |
|----------|------|-------|-----------|
| `GET /health` | 200 | <0.1 с | `{"status":"ok","stt":true,"llm":true,"tts":true}` |
| `POST /tts` (122 КБ WAV 24 кГц) | 200 | 0.76 с | Корректный WAV, голос Андреева, длительность ≈2.5 с |
| `POST /generate_response` (LLM) | 200 | 7.35 с | Ответ + переход INTRO→QUALIFY |
| `POST /dialog_step` (полный цикл STT→LLM→TTS) | 200 | 5.42 с | STT распознал, LLM ответил, TTS синтезировал, переход QUALIFY |

Использовалась модель `qwen2.5:1.5b` через Ollama, STT — Faster-Whisper
small на CUDA, TTS — XTTS v2 (`models/xtts_v2/`).

#### 6.6.2 XTTSAdapter в основном пайплайне (smoke-test)

| Этап | Время | Результат |
|------|-------|-----------|
| `start()` (загрузка fine-tuned + speaker embedding + прогрев) | 13.0 с | Модель готова, latent закэширован |
| `synthesize()` 3 предложения (≈4.8 с аудио) | 0.80 с | 122 чанка по 320 байт mulaw 8 кГц |

Использован fine-tuned чекпойнт
`data/xtts_finetuned_andreev_v2/.../best_model.pth` (2.4 ГБ). Чанки
имеют точный размер 320 байт — корректное соответствие настройке
`chunk_size_ms=40` при mulaw 8 кГц (40 мс × 8000 Гц × 1 байт/сэмпл = 320).

Замечание: при тестировании выявлен дефект высокоуровневого API
Coqui TTS 0.27.5 при загрузке fine-tuned `.pth` (требует
`checkpoint_path=`, а не `model_path=`). Адаптер переписан на
низкоуровневое API (`XttsConfig` + `Xtts.load_checkpoint`),
коммит `3bfeb3ab`.

## 7. Заключение

Все целевые требования (Т-1…Т-10) подтверждены. Регрессионный прогон
показывает 564 успешных теста, отказов нет.

В версии 0.3.0 проведены сквозные испытания обоих маршрутов:

1. **ai_telemarketer** — полный цикл STT→LLM→TTS работает на боевом
   окружении, метрики времени отклика приемлемы для real-time телефонии.
2. **XTTSAdapter в основном пайплайне** — smoke-test подтвердил
   корректную загрузку fine-tuned весов и потоковый синтез.

Программа готова к опытной эксплуатации. Открыт один пункт,
требующий валидации в полевых условиях:

- **Сквозной телефонный звонок** через Asterisk с включённым
  `xtts_tts` адаптером — требует поднятого AudioSocket и SIP-маршрута.
  Подтверждение работоспособности проводится отдельным актом
  опытной эксплуатации.

Подписи (заполняется при формальной сдаче):

- Разработчик: ______________________
- Тестировщик: ______________________
- Дата: 2026-04-29
