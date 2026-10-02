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

from meeting_transcriber import render
from meeting_transcriber.pipeline import AUTO_LANGUAGE, Options, TranscriberError, run

logger = logging.getLogger("transcribe")

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
    parser.add_argument("audio", type=Path, help="fichier audio (.m4a, .mp3, .wav…)")
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
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> str:
    """Contrôles instantanés, avant tout chargement de modèle."""
    if not args.audio.is_file():
        raise TranscriberError(f"Fichier introuvable : {args.audio}")

    if args.speakers is not None and (args.min_speakers or args.max_speakers):
        raise TranscriberError("--speakers et --min/--max-speakers sont exclusifs.")

    if shutil.which("ffmpeg") is None:
        raise TranscriberError(
            "ffmpeg introuvable dans le PATH : indispensable pour décoder l'audio."
        )

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise TranscriberError(TOKEN_HELP)

    return token


def _resolve_device(requested: str, compute_type: str) -> tuple[str, str]:
    import torch

    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise TranscriberError(
            "CUDA demandé mais indisponible. Relance avec --device cpu "
            "(compte plusieurs heures sur une réunion d'une heure)."
        )
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    # Depuis le dossier courant, pas depuis celui du code : installé via
    # `uv tool install`, le paquet vit dans un environnement isolé.
    load_dotenv(find_dotenv(usecwd=True))
    args = _parse_args(argv)

    try:
        token = _validate(args)
        device, compute_type = _resolve_device(args.device, args.compute_type)

        if args.speakers is None and args.min_speakers is None and args.max_speakers is None:
            logger.info(
                "Nombre d'intervenants non précisé : pyannote l'estimera. "
                "--speakers N améliore nettement le résultat si tu le connais."
            )

        duration = _audio_duration(args.audio)
        logger.info(
            "Fichier : %s (%s)",
            args.audio.name,
            render.format_duration(duration) if duration else "durée inconnue",
        )

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

        speakers = render.speakers_in(merged)
        logger.info("Terminé en %s.", render.format_duration(elapsed))
        logger.info(
            "%d segments, %d tours de parole, %d locuteurs : %s",
            len(raw),
            len(merged),
            len(speakers),
            ", ".join(speakers),
        )
        logger.info("  écrit → %s", written)
        return 0

    except TranscriberError as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrompu.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
