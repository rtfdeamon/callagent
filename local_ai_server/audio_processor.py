from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from constants import ULAW_SAMPLE_RATE


class AudioProcessor:
    """Handles audio format conversions for MVP uLaw 8kHz pipeline."""

    @staticmethod
    def resample_audio(
        input_data: bytes,
        input_rate: int,
        output_rate: int,
        input_format: str = "raw",
        output_format: str = "raw",
    ) -> bytes:
        """Resample audio using sox (blocking)."""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{input_format}", delete=False
            ) as input_file:
                input_file.write(input_data)
                input_path = input_file.name

            with tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ) as output_file:
                output_path = output_file.name

            cmd = [
                "sox",
                "-t",
                "raw",
                "-r",
                str(input_rate),
                "-e",
                "signed-integer",
                "-b",
                "16",
                "-c",
                "1",
                input_path,
                "-r",
                str(output_rate),
                "-c",
                "1",
                "-e",
                "signed-integer",
                "-b",
                "16",
                output_path,
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            with open(output_path, "rb") as f:
                resampled_data = f.read()

            os.unlink(input_path)
            os.unlink(output_path)

            return resampled_data

        except Exception as exc:  # pragma: no cover
            logging.error("Audio resampling failed: %s", exc)
            return input_data

    @staticmethod
    def convert_to_ulaw_8k(input_data: bytes, input_rate: int) -> bytes:
        """Convert audio to uLaw 8kHz format for ARI playback (blocking)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
                input_file.write(input_data)
                input_path = input_file.name

            with tempfile.NamedTemporaryFile(suffix=".ulaw", delete=False) as output_file:
                output_path = output_file.name

            cmd = [
                "sox",
                input_path,
                "-r",
                str(ULAW_SAMPLE_RATE),
                "-c",
                "1",
                "-e",
                "mu-law",
                "-t",
                "raw",
                output_path,
                "gain", "-n",  # normalize to 0dBFS before µ-law encoding
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            with open(output_path, "rb") as f:
                ulaw_data = f.read()

            os.unlink(input_path)
            os.unlink(output_path)

            return ulaw_data

        except Exception as exc:  # pragma: no cover
            logging.error("uLaw conversion failed: %s", exc)
            return input_data

