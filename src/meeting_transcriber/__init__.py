"""Transcription locale de réunions audio, avec identification des locuteurs."""

import warnings

# pyannote 4 émet un avertissement de 20 lignes quand torchcodec ne trouve pas
# les DLL FFmpeg partagées. Sans effet ici : le décodage passe par le binaire
# ffmpeg (whisperx.load_audio) et la diarisation reçoit un waveform en mémoire,
# donc torchcodec n'est jamais sollicité.
warnings.filterwarnings("ignore", message=r".*torchcodec.*")

__all__ = ["__version__"]
__version__ = "0.1.0"
