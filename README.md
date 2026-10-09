# meeting-transcriber

Transcrit une réunion enregistrée (`.m4a`, `.mp3`, `.wav`…) en identifiant **qui
parle quand**, puis produit un fichier Markdown prêt à donner à un LLM pour un
résumé, un compte-rendu ou une extraction d'actions.

Tout tourne **en local** : aucun audio n'est envoyé à un service tiers.

```markdown
# Transcription → reunion.m4a

- **Fichier source** : `reunion.m4a`
- **Durée** : 43 min 56 s
- **Locuteurs détectés** : 3 (SPEAKER_00, SPEAKER_01, SPEAKER_02)
- **Modèle** : large-v3 (fr)

---

**[00:00:04 → 00:00:31] SPEAKER_00**

Bon, on commence par le budget. L'idée, c'est de valider l'enveloppe avant vendredi…

**[00:00:32 → 00:00:40] SPEAKER_01**

D'accord, mais il manque encore le devis du prestataire.
```

## Prérequis

| | |
|---|---|
| **Système** | Windows, macOS ou Linux |
| **GPU** | NVIDIA recommandé, avec un pilote à jour. Sans GPU ça fonctionne, mais une réunion d'une heure prend plusieurs heures. Sur Mac, le traitement se fait sur le CPU. |
| **Disque** | ~10 Go (environnement Python + modèles téléchargés au premier lancement) |

## Installation

### 1. Installer ffmpeg et uv

[uv](https://docs.astral.sh/uv/) installe l'outil et la bonne version de Python
(3.12) dans un environnement isolé. Le Python de ton système n'est pas touché.

**Windows** (PowerShell) :

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
```

**macOS** :

```bash
brew install ffmpeg uv
```

**Linux** (Debian / Ubuntu) :

```bash
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ferme puis rouvre le terminal pour que les deux commandes soient reconnues.

### 2. Installer meeting-transcriber

```bash
uv tool install git+https://github.com/TerryHenrard/meeting-transcriber
```

La commande `transcribe` est alors disponible partout. Si le terminal ne la
trouve pas, lance `uv tool update-shell` puis rouvre le terminal.

### 3. Configurer l'accès Hugging Face (une seule fois)

Le modèle qui distingue les voix est hébergé sur Hugging Face. Il est gratuit
mais demande un compte :

1. Crée un compte sur [huggingface.co](https://huggingface.co/join).
2. Ouvre [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
   et accepte les conditions d'utilisation.
3. Crée un jeton avec le droit **Read** sur [hf.co/settings/tokens](https://hf.co/settings/tokens).
4. Enregistre-le dans la variable d'environnement `HF_TOKEN` :

   **Windows** :

   ```powershell
   setx HF_TOKEN hf_xxxxxxxxxxxxxxxx
   ```

   **macOS / Linux** (remplace `~/.zshrc` par `~/.bashrc` si tu utilises bash) :

   ```bash
   echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> ~/.zshrc
   ```

   Rouvre le terminal. Alternative : un fichier `.env` contenant
   `HF_TOKEN=hf_…` dans le dossier depuis lequel tu lances `transcribe`.

## Utilisation

```bash
transcribe "reunion.m4a"
```

Le résultat est écrit à côté du fichier audio : `reunion.md`.

Au premier lancement, les modèles (~4 Go) sont téléchargés : compte quelques
minutes de plus.

```bash
# Avec le nombre d'intervenants connu : nettement meilleur
transcribe "reunion.m4a" --speakers 4

# Fourchette, si le nombre exact est incertain
transcribe "reunion.m4a" --min-speakers 2 --max-speakers 6

# Réunion en anglais, résultat dans un autre dossier
transcribe "meeting.mp3" --language en --out ./transcripts

# Langue inconnue : détection automatique
transcribe "meeting.mp3" --language auto
```

Renseigner `--speakers` est le réglage qui change le plus le résultat.

**Langue.** Par défaut, l'audio est transcrit en français. Si la réunion se
déroule dans une autre langue, précise-la : sinon Whisper risque de *traduire*
en français au lieu de transcrire. Avec `--language auto`, la langue est détectée
sur les 30 premières secondes, puis appliquée à tout le fichier. Cette détection
peut donc se tromper si la réunion commence par un silence, de la musique ou une
autre langue. La langue retenue est affichée dans la console et dans l'en-tête du
fichier `.md`. Une réunion bilingue est transcrite dans une seule langue :
choisis celle qui domine.

### Options

| Option | Défaut | Rôle |
|---|---|---|
| `--out` | dossier du fichier source | dossier de sortie |
| `--language` | `fr` | langue de la réunion (`en`, `de`, `es`…), ou `auto` pour la détecter |
| `--model` | `large-v3` | modèle Whisper (`medium`, `small`… : plus rapide, moins précis) |
| `--speakers` | auto | nombre exact d'intervenants |
| `--min-speakers` / `--max-speakers` | auto | fourchette |
| `--device` | `auto` | forcer `cuda` ou `cpu` |
| `--compute-type` | `auto` | `float16` sur GPU, `int8` sur CPU |
| `--batch-size` | `16` | à réduire si la mémoire GPU sature |
| `--max-gap` | `2.0` | silence max, en secondes, pour fusionner deux tours d'un même locuteur |
| `--fill-nearest` | désactivé | attribue le locuteur le plus proche au lieu de laisser `SPEAKER_?` |

### Sortie

Un fichier `<nom>.md` : un en-tête de métadonnées (source, durée, locuteurs,
modèle), puis le transcript horodaté. Les phrases consécutives d'une même
personne sont regroupées en un seul tour de parole.

Les locuteurs sont nommés `SPEAKER_00`, `SPEAKER_01`… dans l'ordre où le modèle
les distingue. `SPEAKER_?` marque un passage qu'il n'a pas su attribuer.
Astuce : donne au LLM les prénoms des participants, il retrouve généralement
qui est qui à partir du contexte.

## Mettre à jour / désinstaller

```bash
uv tool upgrade meeting-transcriber
uv tool uninstall meeting-transcriber
```

Les modèles téléchargés restent dans le cache Hugging Face
(`~/.cache/huggingface`) : supprime ce dossier pour récupérer l'espace disque.

## Ordres de grandeur

Sur une RTX 4070 Ti SUPER, une réunion de 45 minutes est traitée en une dizaine
de minutes.

## Dépannage

**« Impossible de charger le modèle de diarisation »** : dans 9 cas sur 10, les
conditions de [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
n'ont pas été acceptées avec le compte qui a créé le jeton.

**« CUDA indisponible : bascule sur le CPU »** alors que tu as une carte
NVIDIA : mets à jour le pilote NVIDIA.

**« n'est pas prise en charge par PyTorch »** ou **`no kernel image is available
for execution on the device`** : l'installation date d'avant la prise en charge
des cartes récentes (RTX 50xx). Mets à jour l'outil :
`uv tool install --reinstall git+https://github.com/TerryHenrard/meeting-transcriber`.

**Erreur de mémoire GPU (`CUDA out of memory`)** : relance avec
`--batch-size 4`, ou un modèle plus léger (`--model medium`).

**Avertissement `torchcodec` au démarrage** : sans conséquence. Le décodage
passe par le binaire `ffmpeg`, pas par torchcodec.

**Fichier dans un dossier cloud** (OneDrive, iCloud, Google Drive) : vérifie
qu'il est bien téléchargé en local, et pas seulement « disponible en ligne ».

## Comment ça marche

`ffmpeg` → Whisper `large-v3` (faster-whisper) → alignement wav2vec2 au niveau du
mot → diarisation pyannote → attribution d'un locuteur à chaque mot.

L'alignement au mot **avant** l'attribution est ce qui rend l'outil utilisable
sur un enregistrement fait avec un seul micro posé sur la table : quand deux
personnes se coupent la parole, une attribution au niveau de la phrase se trompe
de locuteur sur toute la phrase.

Le tout repose sur [WhisperX](https://github.com/m-bain/whisperX) et
[pyannote.audio](https://github.com/pyannote/pyannote-audio).

## Développement

```bash
git clone https://github.com/TerryHenrard/meeting-transcriber
cd meeting-transcriber
uv sync
uv run transcribe "reunion.m4a"

uv run pytest -m "not slow"                       # rapide, ni GPU ni modèle
SAMPLE_AUDIO=extrait.m4a uv run pytest -m slow    # intégration réelle
```

## Licence

[MIT](LICENSE)
