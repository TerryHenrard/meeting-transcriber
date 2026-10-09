"""Transcription locale de réunions audio, avec identification des locuteurs."""

import warnings

# Avertissements de bibliothèques, sans action possible pour l'utilisateur.
# `message` est une regex ancrée en début de message, et `.` ne franchit pas
# les retours à la ligne : d'où les `\s*` en tête.
#
# pyannote 4 : 20 lignes quand torchcodec ne trouve pas les DLL FFmpeg
# partagées. Sans effet ici : le décodage passe par le binaire ffmpeg
# (whisperx.load_audio) et la diarisation reçoit un waveform en mémoire,
# donc torchcodec n'est jamais sollicité.
warnings.filterwarnings("ignore", message=r"\s*torchcodec")
# pyannote désactive TF32 pour la reproductibilité, et le signale.
warnings.filterwarnings("ignore", message=r"\s*TensorFloat-32")
# Un segment d'une seule trame donne un embedding NaN, que la diarisation
# écarte ensuite (pyannote/audio/pipelines/clustering.py).
warnings.filterwarnings("ignore", message=r"\s*std\(\): degrees of freedom")
# huggingface_hub, au premier téléchargement sous Windows sans mode développeur :
# le cache fonctionne sans symlinks, il prend juste un peu plus de place.
warnings.filterwarnings("ignore", message=r"\s*`huggingface_hub` cache-system uses symlinks")

__all__ = ["__version__"]
__version__ = "0.3.0"
