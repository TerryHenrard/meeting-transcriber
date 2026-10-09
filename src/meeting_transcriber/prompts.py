"""Mode interactif : les questions posées quand `transcribe` est lancé sans fichier.

Chaque question est un prompt rich qui valide sa réponse et la redemande tant
qu'elle est invalide. Convention commune : indications en gris, valeur prise
par Entrée entre crochets.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.prompt import Confirm, InvalidResponse, Prompt, PromptBase
from rich.text import Text

from meeting_transcriber.console import console
from meeting_transcriber.pipeline import AUTO_LANGUAGE


class Cancelled(Exception):
    """L'utilisateur a quitté le questionnaire (Ctrl+C, Ctrl+D ou Ctrl+Z)."""


def clean_path(raw: str, *, posix: bool = os.name != "nt") -> str:
    """Chemin tapé ou glissé-déposé dans le terminal.

    Le glisser-déposer entoure le chemin de guillemets (Windows, Linux) ou
    échappe ses espaces par des antislashs (macOS).
    """
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if posix:
        return re.sub(r"\\(.)", r"\1", text)
    return text


class _Convention:
    """Valeur par défaut entre crochets, « : » précédé d'une espace."""

    prompt_suffix = " : "

    def render_default(self, default: Any) -> Text:
        return Text(f"[{default}]", style="prompt.default")


class _AudioPrompt(_Convention, PromptBase[Path]):
    def process_response(self, value: str) -> Path:
        text = clean_path(value)
        if not text:
            raise InvalidResponse("[prompt.invalid]Indique le chemin du fichier audio")
        path = Path(text).expanduser()
        if not path.is_file():
            raise InvalidResponse(f"[prompt.invalid]Fichier introuvable : {escape(str(path))}")
        return path


class _SpeakersPrompt(_Convention, PromptBase[int | None]):
    """Entrée vide : estimation automatique, d'où `None` plutôt qu'un défaut rich."""

    def process_response(self, value: str) -> int | None:
        text = value.strip()
        if not text:
            return None
        try:
            count = int(text)
        except ValueError:
            raise InvalidResponse("[prompt.invalid]Entre un nombre entier") from None
        if count < 1:
            raise InvalidResponse("[prompt.invalid]Il faut au moins 1 intervenant")
        return count


class _LanguagePrompt(_Convention, Prompt):
    def process_response(self, value: str) -> str:
        language = value.strip().lower()
        if not language:
            raise InvalidResponse("[prompt.invalid]Indique une langue")
        return language


class _Confirm(_Convention, Confirm):
    """La réponse par défaut en majuscule : [O/n] ou [o/N]."""

    choices = ["o", "n"]
    validate_error_message = "[prompt.invalid]Réponds o ou n"

    def render_default(self, default: Any) -> Text:
        yes, no = self.choices
        label = f"[{yes.upper()}/{no}]" if default else f"[{yes}/{no.upper()}]"
        return Text(label, style="prompt.choices")


def _ask(prompt: type[PromptBase[Any]], text: str, **kwargs: Any) -> Any:
    try:
        return prompt.ask(text, console=console, **kwargs)
    except (KeyboardInterrupt, EOFError):
        # Le curseur est resté au bout de la question.
        console.print()
        raise Cancelled from None


def ask(args: argparse.Namespace) -> None:
    """Complète `args` : fichier audio, intervenants s'ils manquent, langue."""
    console.print("[bold]Transcription d'une réunion[/]")
    console.print("[dim]Entrée = valeur entre \\[crochets] · Ctrl+C pour quitter[/]")
    console.print()
    args.audio = _ask(_AudioPrompt, "Fichier audio [dim](glisse-le ici)[/]")
    if args.speakers is None and args.min_speakers is None and args.max_speakers is None:
        args.speakers = _ask(
            _SpeakersPrompt,
            "Nombre d'intervenants [dim](si tu le connais)[/] [prompt.default]\\[auto][/]",
        )
    args.language = _ask(
        _LanguagePrompt,
        f"Langue [dim](en, de… ou {AUTO_LANGUAGE})[/]",
        default=args.language,
    )
    console.print()


def confirm_start() -> bool:
    return _ask(_Confirm, "Lancer la transcription ?", default=True, show_choices=False)
