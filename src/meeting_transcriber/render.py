"""Rendu des tours de parole en fichiers lisibles.

Ce module ne dépend ni du GPU ni d'un modèle : il ne connaît que des `Turn`.
C'est la frontière qui rend la mise en forme testable sans rien charger.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

UNKNOWN_SPEAKER = "SPEAKER_?"
ARROW = "\u2192"


@dataclass(frozen=True)
class Turn:
    """Un bloc de parole attribué à un locuteur."""

    start: float
    end: float
    speaker: str
    text: str


def merge_turns(turns: Iterable[Turn], max_gap: float = 2.0) -> list[Turn]:
    """Fusionne les tours consécutifs d'un même locuteur séparés de moins de `max_gap`.

    Whisper découpe un monologue continu en segments de 5 à 30 s. Sans fusion, le
    transcript est illisible : un en-tête de locuteur toutes les deux phrases.
    """
    merged: list[Turn] = []
    for turn in turns:
        text = turn.text.strip()
        if not text:
            continue
        if merged:
            previous = merged[-1]
            if previous.speaker == turn.speaker and turn.start - previous.end <= max_gap:
                merged[-1] = Turn(
                    start=previous.start,
                    end=max(previous.end, turn.end),
                    speaker=previous.speaker,
                    text=f"{previous.text} {text}",
                )
                continue
        merged.append(Turn(turn.start, turn.end, turn.speaker, text))
    return merged


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS."""
    whole = int(max(0.0, seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """Durée lisible, p. ex. « 43 min 56 s » ou « 1 h 12 min 03 s »."""
    whole = int(max(0.0, seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    return f"{minutes} min {secs:02d} s"


def speakers_in(turns: Iterable[Turn]) -> list[str]:
    """Locuteurs présents, triés, l'inconnu en dernier."""
    labels = {turn.speaker for turn in turns}
    known = sorted(label for label in labels if label != UNKNOWN_SPEAKER)
    return known + ([UNKNOWN_SPEAKER] if UNKNOWN_SPEAKER in labels else [])


def to_markdown(
    turns: Sequence[Turn],
    *,
    source: str,
    duration: float,
    model: str,
    language: str,
) -> str:
    """Transcript en Markdown, précédé d'un en-tête de métadonnées."""
    speakers = speakers_in(turns)
    header = "\n".join(
        [
            f"# Transcription {ARROW} {source}",
            "",
            f"- **Fichier source** : `{source}`",
            f"- **Durée** : {format_duration(duration)}",
            f"- **Locuteurs détectés** : {len(speakers)} ({', '.join(speakers) or 'aucun'})",
            f"- **Modèle** : {model} ({language})",
            "",
            "---",
            "",
            "",
        ]
    )
    body = "\n\n".join(
        f"**[{format_timestamp(t.start)} {ARROW} {format_timestamp(t.end)}] {t.speaker}**"
        f"\n\n{t.text}"
        for t in turns
    )
    return header + body + "\n"
