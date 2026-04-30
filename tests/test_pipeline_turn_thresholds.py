import time
from types import SimpleNamespace

from src.engine import Engine


def test_pipeline_transcript_ready_accepts_short_flushes():
    assert Engine._pipeline_transcript_ready("да", force=True) is True
    assert Engine._pipeline_transcript_ready("ok", force=True) is True


def test_pipeline_transcript_ready_requires_threshold_without_flush():
    assert Engine._pipeline_transcript_ready("да", force=False) is False
    assert Engine._pipeline_transcript_ready("нужна смета", force=False) is True


def test_pipeline_transcript_stats_handles_short_phrases():
    assert Engine._pipeline_transcript_stats("нет") == (1, 3)
    assert Engine._pipeline_transcript_stats("что дальше") == (2, 9)


def test_local_stt_default_aggregation_timeout_is_low_latency():
    assert Engine._default_pipeline_aggregation_timeout_sec("local_stt") == 0.6
    assert Engine._default_pipeline_aggregation_timeout_sec("deepgram_stt") == 1.2


def test_pipeline_no_input_reprompt_triggers_on_fresh_zero_rms():
    session = SimpleNamespace(
        tts_playing=False,
        audio_capture_enabled=True,
        media_rx_confirmed=True,
        audio_diagnostics={
            "transport_in": {
                "rms": 0,
                "updated": time.time(),
            }
        },
    )
    assert Engine._should_emit_pipeline_no_input_reprompt(
        session,
        reprompts_sent=0,
        max_reprompts=1,
    ) is True


def test_pipeline_no_input_reprompt_skips_when_audio_present_or_limit_reached():
    active_audio_session = SimpleNamespace(
        tts_playing=False,
        audio_capture_enabled=True,
        media_rx_confirmed=True,
        audio_diagnostics={
            "transport_in": {
                "rms": 420,
                "updated": time.time(),
            }
        },
    )
    assert Engine._should_emit_pipeline_no_input_reprompt(
        active_audio_session,
        reprompts_sent=0,
        max_reprompts=1,
    ) is False

    exhausted_session = SimpleNamespace(
        tts_playing=False,
        audio_capture_enabled=True,
        media_rx_confirmed=True,
        audio_diagnostics={
            "transport_in": {
                "rms": 0,
                "updated": time.time(),
            }
        },
    )
    assert Engine._should_emit_pipeline_no_input_reprompt(
        exhausted_session,
        reprompts_sent=1,
        max_reprompts=1,
    ) is False
