"""Tests du rendu. Aucun GPU ni modèle : c'est ici que vivent les vrais bugs."""

from __future__ import annotations

import pytest

from meeting_transcriber.render import (
    UNKNOWN_SPEAKER,
    Turn,
    format_duration,
    format_timestamp,
    merge_turns,
    speakers_in,
    to_markdown,
)


def turn(start: float, end: float, speaker: str = "SPEAKER_00", text: str = "bla") -> Turn:
    return Turn(start=start, end=end, speaker=speaker, text=text)


class TestMergeTurns:
    def test_merges_consecutive_turns_of_same_speaker(self):
        merged = merge_turns([turn(0, 3, text="Bonjour."), turn(3.5, 9, text="On démarre.")])
        assert len(merged) == 1
        assert merged[0] == Turn(0, 9, "SPEAKER_00", "Bonjour. On démarre.")

    def test_keeps_different_speakers_apart(self):
        merged = merge_turns([turn(0, 3), turn(3.1, 9, speaker="SPEAKER_01")])
        assert [t.speaker for t in merged] == ["SPEAKER_00", "SPEAKER_01"]

    def test_breaks_on_long_silence(self):
        merged = merge_turns([turn(0, 3), turn(10, 14)], max_gap=2.0)
        assert len(merged) == 2

    def test_gap_is_measured_from_previous_end(self):
        # 3.0 -> 5.0 : exactement max_gap, donc fusionné (borne inclusive).
        assert len(merge_turns([turn(0, 3), turn(5, 7)], max_gap=2.0)) == 1
        assert len(merge_turns([turn(0, 3), turn(5.01, 7)], max_gap=2.0)) == 2

    def test_overlapping_turns_keep_the_latest_end(self):
        merged = merge_turns([turn(0, 10), turn(8, 12)])
        assert merged[0].end == 12

    def test_drops_empty_and_whitespace_only_turns(self):
        merged = merge_turns([turn(0, 3, text="   "), turn(4, 5, text="Réel")])
        assert len(merged) == 1
        assert merged[0].text == "Réel"

    def test_strips_text_before_joining(self):
        merged = merge_turns([turn(0, 3, text="  A  "), turn(3.5, 5, text="  B  ")])
        assert merged[0].text == "A B"

    def test_empty_input(self):
        assert merge_turns([]) == []


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00:00"), (61, "00:01:01"), (3661, "01:01:01"), (2636.35, "00:43:56")],
    )
    def test_clock_format(self, seconds, expected):
        assert format_timestamp(seconds) == expected

    def test_negative_is_clamped(self):
        assert format_timestamp(-5) == "00:00:00"


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(2636.35, "43 min 56 s"), (4323, "1 h 12 min 03 s"), (5, "0 min 05 s")],
    )
    def test_readable(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestSpeakersIn:
    def test_sorted_with_unknown_last(self):
        turns = [
            turn(0, 1, speaker="SPEAKER_01"),
            turn(1, 2, speaker=UNKNOWN_SPEAKER),
            turn(2, 3, speaker="SPEAKER_00"),
        ]
        assert speakers_in(turns) == ["SPEAKER_00", "SPEAKER_01", UNKNOWN_SPEAKER]

    def test_deduplicates(self):
        assert speakers_in([turn(0, 1), turn(1, 2)]) == ["SPEAKER_00"]


class TestToMarkdown:
    def test_header_carries_metadata(self):
        out = to_markdown(
            [turn(0, 3, text="Salut.")],
            source="reunion.m4a",
            duration=2636.35,
            model="large-v3",
            language="fr",
        )
        assert "`reunion.m4a`" in out
        assert "43 min 56 s" in out
        assert "**Locuteurs détectés** : 1 (SPEAKER_00)" in out
        assert "large-v3 (fr)" in out
        assert out.endswith("Salut.\n")

    def test_turn_layout(self):
        out = to_markdown(
            [turn(0, 3, text="Salut."), turn(10, 12, speaker="SPEAKER_01", text="Yo.")],
            source="x.m4a",
            duration=12,
            model="m",
            language="fr",
        )
        assert out.endswith(
            "**[00:00:00 → 00:00:03] SPEAKER_00**\n\nSalut.\n\n"
            "**[00:00:10 → 00:00:12] SPEAKER_01**\n\nYo.\n"
        )

    def test_no_speaker_reported_when_empty(self):
        out = to_markdown([], source="x.m4a", duration=0, model="m", language="fr")
        assert "0 (aucun)" in out
