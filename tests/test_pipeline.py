"""Tests du pipeline.

Les fonctions pures sont testées directement. L'enchaînement réel est couvert
par un test d'intégration marqué `slow` : il exige le GPU, le token et les
modèles téléchargés. Pas de mock de torch — ça ne prouverait rien.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from meeting_transcriber.pipeline import Options, _to_turns, run
from meeting_transcriber.render import UNKNOWN_SPEAKER, merge_turns


class TestToTurns:
    def test_maps_segment_fields(self):
        turns = _to_turns([{"start": 1.0, "end": 2.5, "text": " Salut ", "speaker": "SPEAKER_01"}])
        assert len(turns) == 1
        assert (turns[0].start, turns[0].end) == (1.0, 2.5)
        assert turns[0].text == "Salut"
        assert turns[0].speaker == "SPEAKER_01"

    def test_missing_speaker_becomes_unknown(self):
        # assign_word_speakers laisse des segments sans locuteur quand aucun
        # tour de diarisation ne les recouvre.
        turns = _to_turns([{"start": 0, "end": 1, "text": "a"}])
        assert turns[0].speaker == UNKNOWN_SPEAKER

    def test_null_speaker_becomes_unknown(self):
        turns = _to_turns([{"start": 0, "end": 1, "text": "a", "speaker": None}])
        assert turns[0].speaker == UNKNOWN_SPEAKER

    def test_drops_segments_without_timestamps(self):
        # L'alignement peut échouer sur un segment et le rendre sans bornes.
        assert _to_turns([{"text": "a"}]) == []
        assert _to_turns([{"start": 0, "text": "a"}]) == []

    def test_drops_empty_text(self):
        assert _to_turns([{"start": 0, "end": 1, "text": "   "}]) == []


@pytest.mark.slow
def test_end_to_end(tmp_path: Path):
    """Intégration réelle sur un extrait court. Nécessite GPU + HF_TOKEN + modèles."""
    sample = Path(os.environ.get("SAMPLE_AUDIO", "tests/data/sample.m4a"))
    token = os.environ.get("HF_TOKEN", "")
    if not sample.is_file() or not token:
        pytest.skip("SAMPLE_AUDIO et HF_TOKEN requis pour le test d'intégration")

    turns, language = run(sample, token, Options(batch_size=4))

    assert language == "fr"
    assert turns, "le pipeline doit produire au moins un tour"
    assert all(t.end >= t.start for t in turns)
    assert all(t.text.strip() for t in turns)
    merged = merge_turns(turns)
    assert len(merged) <= len(turns)
