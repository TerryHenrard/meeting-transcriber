"""Tests du mode interactif."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from meeting_transcriber import prompts
from meeting_transcriber.prompts import clean_path


class TestCleanPath:
    def test_windows_double_quotes(self):
        raw = '"C:\\Users\\moi\\Réunion du lundi.m4a"'
        assert clean_path(raw, posix=False) == "C:\\Users\\moi\\Réunion du lundi.m4a"

    def test_windows_backslashes_kept(self):
        assert clean_path("C:\\audio\\a.m4a", posix=False) == "C:\\audio\\a.m4a"

    def test_linux_single_quotes_and_trailing_space(self):
        assert clean_path("'/home/moi/ma réunion.m4a' ", posix=True) == "/home/moi/ma réunion.m4a"

    def test_macos_escaped_spaces(self):
        assert clean_path("/Users/moi/ma\\ réunion.m4a", posix=True) == "/Users/moi/ma réunion.m4a"


def _namespace(**overrides) -> argparse.Namespace:
    values = {
        "audio": None,
        "language": "fr",
        "speakers": None,
        "min_speakers": None,
        "max_speakers": None,
    }
    return argparse.Namespace(**(values | overrides))


@pytest.fixture
def answers(monkeypatch):
    """Simule les réponses tapées, dans l'ordre des questions."""

    def feed(*lines: str) -> None:
        queue = iter(lines)
        monkeypatch.setattr("builtins.input", lambda *_: next(queue))

    return feed


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "ma réunion.m4a"
    path.write_bytes(b"")
    return path


class TestAsk:
    def test_fills_answers(self, answers, audio):
        answers(f'"{audio}"', "4", "EN")
        args = _namespace()
        prompts.ask(args)
        assert (args.audio, args.speakers, args.language) == (audio, 4, "en")

    def test_defaults_on_enter(self, answers, audio):
        answers(str(audio), "", "")
        args = _namespace()
        prompts.ask(args)
        assert (args.speakers, args.language) == (None, "fr")

    def test_asks_again_until_valid(self, answers, audio, tmp_path):
        answers("", str(tmp_path / "absent.m4a"), str(audio), "deux", "0", "3", "")
        args = _namespace()
        prompts.ask(args)
        assert (args.audio, args.speakers) == (audio, 3)

    def test_skips_speakers_given_as_option(self, answers, audio):
        answers(str(audio), "")
        args = _namespace(min_speakers=2, language="de")
        prompts.ask(args)
        assert (args.speakers, args.language) == (None, "de")

    def test_shows_defaults_in_brackets(self, answers, audio, capsys):
        answers(str(audio), "", "")
        prompts.ask(_namespace(language="de"))
        err = capsys.readouterr().err
        assert "[auto] : " in err
        assert "[de] : " in err

    @pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
    def test_interruption_cancels(self, monkeypatch, interruption):
        def interrupt(*_):
            raise interruption

        monkeypatch.setattr("builtins.input", interrupt)
        with pytest.raises(prompts.Cancelled):
            prompts.ask(_namespace())


class TestConfirmStart:
    @pytest.mark.parametrize(("line", "expected"), [("", True), ("o", True), ("N", False)])
    def test_answers(self, answers, line, expected):
        answers(line)
        assert prompts.confirm_start() is expected

    def test_default_in_uppercase(self, answers, capsys):
        answers("")
        prompts.confirm_start()
        assert "Lancer la transcription ? [O/n] : " in capsys.readouterr().err

    def test_rejects_english_yes(self, answers):
        answers("y", "n")
        assert prompts.confirm_start() is False
