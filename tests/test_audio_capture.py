from src.utils.audio_capture import AudioCaptureManager


def test_append_raw_persists_bytes_when_keep_files_enabled(tmp_path):
    manager = AudioCaptureManager(base_dir=str(tmp_path), keep_files=True)

    manager.append_raw("call-1", "agent_out_to_caller", b"\x7f\x7f", extension="ulaw")
    manager.append_raw("call-1", "agent_out_to_caller", b"\x01\x02", extension="ulaw")
    manager.close_call("call-1")

    capture_path = tmp_path / "call-1" / "agent_out_to_caller.ulaw"
    assert capture_path.read_bytes() == b"\x7f\x7f\x01\x02"
