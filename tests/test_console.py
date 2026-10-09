"""Tests de l'affichage console, sur une console rich non interactive."""

from __future__ import annotations

import io
import logging

import pytest
from rich.console import Console

from meeting_transcriber import console as console_module
from meeting_transcriber.console import Steps, adopt_library_loggers, setup_logging


@pytest.fixture
def output(monkeypatch) -> io.StringIO:
    buffer = io.StringIO()
    monkeypatch.setattr(console_module, "console", Console(file=buffer, width=100))
    return buffer


class TestSteps:
    def test_writes_one_final_line_per_step(self, output):
        steps = Steps(total=2)
        with steps.step("Décodage"):
            pass
        with steps.step("Transcription") as report:
            for percent in range(0, 101, 10):
                report(float(percent))

        lines = output.getvalue().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("✓ [1/2] Décodage")
        assert lines[1].startswith("✓ [2/2] Transcription")
        assert lines[0].rstrip().endswith(" s")

    def test_percentage_only_for_measured_steps(self, output):
        steps = Steps(total=2)
        with steps.step("Sans avancement"):
            pass
        with steps.step("Avec avancement") as report:
            report(50.0)

        unmeasured, measured = output.getvalue().splitlines()
        assert "%" not in unmeasured
        assert "100 %" in measured

    def test_durations_are_aligned_across_steps(self, output):
        steps = Steps(total=2)
        with steps.step("Sans avancement"):
            pass
        with steps.step("Avec avancement") as report:
            report(50.0)

        unmeasured, measured = output.getvalue().splitlines()
        assert len(unmeasured.rstrip()) == len(measured.rstrip())

    def test_failure_is_marked_and_propagated(self, output):
        steps = Steps(total=1)
        with pytest.raises(RuntimeError):
            with steps.step("Diarisation"):
                raise RuntimeError("boom")

        assert output.getvalue().startswith("✗ [1/1] Diarisation")


def test_adopt_library_loggers_routes_to_root():
    library = logging.getLogger("whisperx")
    library.addHandler(logging.StreamHandler())
    library.setLevel(logging.INFO)
    library.propagate = False

    adopt_library_loggers()

    assert library.level == logging.WARNING
    assert library.propagate
    # Un logger sans handler pousserait whisperx à réinstaller le sien.
    assert [type(h) for h in library.handlers] == [logging.NullHandler]


def test_library_warning_is_prefixed_once(output):
    setup_logging()
    adopt_library_loggers()
    logging.getLogger("huggingface_hub.file_download").warning("hf_xet absent")

    assert output.getvalue() == "Attention : hf_xet absent\n"
