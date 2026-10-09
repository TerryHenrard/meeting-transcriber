"""Pipeline WhisperX : transcription → alignement → diarisation → attribution.

Les imports lourds (torch, whisperx) sont différés dans `run` : le CLI doit
pouvoir échouer sur un fichier manquant ou un token absent en quelques
millisecondes, pas après avoir chargé quatre gigaoctets de modèles.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from meeting_transcriber.console import Steps, adopt_library_loggers, console
from meeting_transcriber.render import UNKNOWN_SPEAKER, Turn

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DIARIZATION_LICENSE_URL = f"https://hf.co/{DIARIZATION_MODEL}"
# Valeur de `language` qui laisse Whisper détecter la langue sur les 30 premières secondes.
AUTO_LANGUAGE = "auto"


class TranscriberError(RuntimeError):
    """Erreur attendue, à présenter à l'utilisateur sans stacktrace."""


class DiarizationAccessError(TranscriberError):
    def __init__(self) -> None:
        super().__init__(
            "Impossible de charger le modèle de diarisation "
            f"« {DIARIZATION_MODEL} ».\n"
            "Causes possibles, dans l'ordre de probabilité :\n"
            f"  1. Conditions du modèle non acceptées : ouvre {DIARIZATION_LICENSE_URL} "
            "et valide le formulaire (compte Hugging Face gratuit requis).\n"
            "  2. HF_TOKEN invalide, expiré, ou sans le droit « Read ».\n"
            "  3. Pas de connexion réseau pour le premier téléchargement."
        )


@dataclass(frozen=True)
class Options:
    """Réglages de transcription. `num_speakers` prime sur min/max.

    `language` accepte `AUTO_LANGUAGE` pour laisser Whisper la détecter.
    """

    language: str = "fr"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    fill_nearest: bool = False


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_turns(segments: list[dict]) -> list[Turn]:
    """Convertit les segments WhisperX en `Turn`, en écartant l'inexploitable."""
    turns: list[Turn] = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        start, end = segment.get("start"), segment.get("end")
        if not text or start is None or end is None:
            continue
        turns.append(
            Turn(
                start=float(start),
                end=float(end),
                speaker=segment.get("speaker") or UNKNOWN_SPEAKER,
                text=text,
            )
        )
    return turns


def run(audio_path: Path, token: str, options: Options) -> tuple[list[Turn], str]:
    """Exécute le pipeline complet.

    Renvoie les tours NON fusionnés et la langue effective (détectée si
    `options.language` vaut `AUTO_LANGUAGE`).
    """
    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    adopt_library_loggers()
    steps = Steps(total=5)

    with steps.step("Décodage audio"):
        audio = whisperx.load_audio(str(audio_path))

    with steps.step("Transcription") as report:
        model = whisperx.load_model(
            options.model,
            options.device,
            compute_type=options.compute_type,
            language=None if options.language == AUTO_LANGUAGE else options.language,
            use_auth_token=token,
        )
        result = model.transcribe(
            audio, batch_size=options.batch_size, progress_callback=report
        )
        del model
        _free_gpu()

    # Le modèle d'alignement wav2vec2 est propre à chaque langue.
    language = result["language"]
    if options.language == AUTO_LANGUAGE:
        console.print(f"  Langue détectée : [bold]{language}[/]")

    with steps.step("Horodatage des mots") as report:
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=options.device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            options.device,
            progress_callback=report,
        )
        del align_model
        _free_gpu()

    with steps.step("Diarisation") as report:
        try:
            diarizer = DiarizationPipeline(
                model_name=DIARIZATION_MODEL, token=token, device=options.device
            )
        except Exception as exc:  # noqa: BLE001 — toute cause remonte le même diagnostic
            raise DiarizationAccessError from exc

        diarization = diarizer(
            audio,
            num_speakers=options.num_speakers,
            min_speakers=options.min_speakers,
            max_speakers=options.max_speakers,
            progress_callback=report,
        )
        del diarizer
        _free_gpu()

    with steps.step("Attribution des locuteurs"):
        result = assign_word_speakers(diarization, result, fill_nearest=options.fill_nearest)
    return _to_turns(result["segments"]), language
