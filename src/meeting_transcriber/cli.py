"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from rich.markup import escape

from meeting_transcriber import prompts, render
from meeting_transcriber.console import console, setup_logging
from meeting_transcriber.pipeline import AUTO_LANGUAGE, Options, TranscriberError, run

logger = logging.getLogger(__name__)

TOKEN_HELP = (
    "Jeton Hugging Face absent. Crée-en un (droit « Read ») sur "
    "https://hf.co/settings/tokens, puis définis la variable d'environnement HF_TOKEN :\n"
    "    Windows       : setx HF_TOKEN hf_xxxxxxxx  (puis rouvre le terminal)\n"
    "    macOS / Linux : export HF_TOKEN=hf_xxxxxxxx  (dans ~/.zshrc ou ~/.bashrc)\n"
    "Un fichier .env contenant HF_TOKEN=… dans le dossier courant fonctionne aussi."
)


def _use_utf8_console() -> None:
    """La console Windows est en cp1252 : sans ça, le moindre accent lève une erreur."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcrit une réunion audio en identifiant les locuteurs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="fichier audio (.m4a, .mp3, .wav…) ; sans fichier, les réglages sont demandés",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="dossier de sortie (défaut : à côté du fichier source)",
    )
    parser.add_argument(
        "--language",
        default="fr",
        help=f"langue de la réunion (en, de…), ou « {AUTO_LANGUAGE} » pour la détecter",
    )
    parser.add_argument("--model", default="large-v3", help="modèle Whisper")
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="nombre exact d'intervenants, si tu le connais",
    )
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--compute-type", default="auto", help="auto, float16, int8_float16, int8…"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="réduire en cas de mémoire GPU insuffisante",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=2.0,
        help="silence max en secondes pour fusionner deux tours d'un même locuteur",
    )
    parser.add_argument(
        "--fill-nearest",
        action="store_true",
        help="attribuer au locuteur le plus proche plutôt que de laisser « ? »",
    )
    args = parser.parse_args(argv)
    if args.audio is None and not sys.stdin.isatty():
        parser.error("fichier audio manquant (les questions demandent un terminal interactif)")
    return args


def _check_environment() -> str:
    """Contrôles instantanés, avant les questions et tout chargement de modèle."""
    if shutil.which("ffmpeg") is None:
        raise TranscriberError(
            "ffmpeg introuvable dans le PATH : indispensable pour décoder l'audio."
        )

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise TranscriberError(TOKEN_HELP)

    return token


def _validate(args: argparse.Namespace) -> None:
    if not args.audio.is_file():
        raise TranscriberError(f"Fichier introuvable : {args.audio}")

    if args.speakers is not None and (args.min_speakers or args.max_speakers):
        raise TranscriberError("--speakers et --min/--max-speakers sont exclusifs.")


def _gpu_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Vrai si torch embarque du code exécutable sur ce GPU.

    Un binaire sm_XY tourne sur toute carte de même version majeure X et de
    version mineure >= Y : une sm_89 exécute du sm_86, une sm_120 pas du sm_90.
    """
    major, minor = capability
    for arch in arch_list:
        if arch.startswith("sm_"):
            digits = arch[3:].rstrip("abcdefghijklmnopqrstuvwxyz")
            if int(digits[:-1]) == major and int(digits[-1]) <= minor:
                return True
    return False


def _resolve_device(requested: str, compute_type: str) -> tuple[str, str]:
    import torch

    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise TranscriberError(
            "CUDA demandé mais indisponible. Relance avec --device cpu "
            "(compte plusieurs heures sur une réunion d'une heure)."
        )
    if available and requested in ("auto", "cuda"):
        capability = torch.cuda.get_device_capability()
        if not _gpu_supported(capability, torch.cuda.get_arch_list()):
            message = (
                f"La carte {torch.cuda.get_device_name()} (sm_{capability[0]}{capability[1]}) "
                f"n'est pas prise en charge par PyTorch {torch.__version__}. "
                "Mets à jour l'outil : uv tool install --reinstall "
                "git+https://github.com/TerryHenrard/meeting-transcriber"
            )
            if requested == "cuda":
                raise TranscriberError(message)
            logger.warning(message)
            available = False
    device = "cuda" if available and requested in ("auto", "cuda") else "cpu"
    if requested == "auto" and not available:
        logger.warning("CUDA indisponible : bascule sur le CPU, le traitement sera très lent.")
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def _audio_duration(path: Path) -> float:
    """Durée via ffprobe ; 0.0 si indisponible (métadonnée d'affichage uniquement)."""
    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        output = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return float(output.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _speakers_hint(args: argparse.Namespace) -> str:
    if args.speakers is not None:
        return str(args.speakers)
    if args.min_speakers is not None and args.max_speakers is not None:
        return f"entre {args.min_speakers} et {args.max_speakers}"
    if args.min_speakers is not None:
        return f"au moins {args.min_speakers}"
    if args.max_speakers is not None:
        return f"au plus {args.max_speakers}"
    return "estimés [dim](--speakers N améliore nettement le résultat)[/]"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count > 1 else ''}"


def _print_summary(merged: list[render.Turn], elapsed: float, written: Path) -> None:
    speakers = render.speakers_in(merged)
    known = [speaker for speaker in speakers if speaker != render.UNKNOWN_SPEAKER]
    unknown = " + passages non attribués" if len(known) < len(speakers) else ""
    console.print()
    console.print(f"[bold green]✓ Terminé en {render.format_duration(elapsed)}[/]")
    console.print(
        f"  {_plural(len(merged), 'tour')} de parole · {_plural(len(known), 'locuteur')}{unknown}"
    )
    console.print(f"  → {escape(str(written))}", soft_wrap=True)


def _write_output(
    merged: list[render.Turn], args: argparse.Namespace, duration: float, language: str
) -> Path:
    out_dir = args.out or args.audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.audio.stem}.md"
    content = render.to_markdown(
        merged,
        source=args.audio.name,
        duration=duration,
        model=args.model,
        language=language,
    )
    path.write_text(content, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    setup_logging()
    # Depuis le dossier courant, pas depuis celui du code : installé via
    # `uv tool install`, le paquet vit dans un environnement isolé.
    load_dotenv(find_dotenv(usecwd=True))
    args = _parse_args(argv)
    interactive = args.audio is None

    try:
        token = _check_environment()
        if interactive:
            prompts.ask(args)
        _validate(args)
        duration = _audio_duration(args.audio)
        console.print(
            f"[bold]{escape(args.audio.name)}[/] · "
            + (render.format_duration(duration) if duration else "durée inconnue")
        )
        device, compute_type = _resolve_device(args.device, args.compute_type)
        console.print(
            f"  {escape(args.model)} · {escape(args.language)} · {device} ({compute_type})"
        )
        console.print(f"  Intervenants : {_speakers_hint(args)}")
        if interactive and not prompts.confirm_start():
            console.print("Annulé.", style="dim")
            return 0
        console.print()

        started = time.monotonic()
        raw, language = run(
            args.audio,
            token,
            Options(
                language=args.language,
                model=args.model,
                device=device,
                compute_type=compute_type,
                batch_size=args.batch_size,
                num_speakers=args.speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                fill_nearest=args.fill_nearest,
            ),
        )
        elapsed = time.monotonic() - started

        if not raw:
            logger.warning(
                "Aucune parole détectée. Vérifie que le fichier contient bien de l'audio."
            )
            return 1

        merged = render.merge_turns(raw, max_gap=args.max_gap)
        written = _write_output(merged, args, duration, language)

        _print_summary(merged, elapsed, written)
        return 0

    except TranscriberError as error:
        logger.error("%s", error)
        return 1
    except prompts.Cancelled:
        console.print("Annulé.", style="dim")
        return 130
    except KeyboardInterrupt:
        logger.error("Interrompu.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
