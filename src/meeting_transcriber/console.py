"""Affichage console : étapes numérotées avec progression, et journalisation.

Tout passe par la même `Console` rich, sur stderr : un message journalisé
pendant une étape s'affiche au-dessus de la barre au lieu de la casser.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, Task, TextColumn
from rich.table import Column
from rich.text import Text

from meeting_transcriber.render import format_duration

console = Console(stderr=True, highlight=False)

# Ces bibliothèques branchent leur propre handler à l'import, et whisperx et
# lightning forcent en plus le niveau INFO : sans reprise en main, la console
# reçoit leurs messages techniques (VAD, migration de checkpoint…), dans leur
# format, et en double puisqu'ils remontent aussi jusqu'à notre handler.
_LIBRARY_LOGGERS = (
    "whisperx",
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "huggingface_hub",
)


class _ConsoleHandler(logging.Handler):
    _STYLES = {logging.WARNING: "yellow", logging.ERROR: "bold red"}
    _PREFIXES = {logging.WARNING: "Attention : ", logging.ERROR: "Erreur : "}

    def __init__(self) -> None:
        super().__init__()
        # Sans formatter, basicConfig imposerait « NIVEAU:logger:message ».
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = min(record.levelno, logging.ERROR)
            message = self._PREFIXES.get(level, "") + self.format(record)
            console.print(message, style=self._STYLES.get(level), markup=False)
        except Exception:  # noqa: BLE001 — contrat de logging.Handler
            self.handleError(record)


def setup_logging() -> None:
    """Nos messages dès INFO, ceux des bibliothèques à partir de WARNING."""
    logging.basicConfig(level=logging.WARNING, handlers=[_ConsoleHandler()], force=True)
    logging.getLogger("meeting_transcriber").setLevel(logging.INFO)


def adopt_library_loggers() -> None:
    """Ramène les loggers tiers au niveau WARNING et à notre handler.

    À appeler une fois whisperx et pyannote importés : lightning reconfigure
    son logger à l'import. Le `NullHandler` empêche whisperx de réinstaller le
    sien, ce qu'il fait dès qu'il trouve son logger sans handler.
    """
    for name in _LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        library.handlers = [logging.NullHandler()]
        library.setLevel(logging.WARNING)
        library.propagate = True


class _StatusColumn(SpinnerColumn):
    def render(self, task: Task) -> Text:
        if task.fields["failed"]:
            return Text("✗", style="bold red")
        return super().render(task)


class _MeasuredColumn(ProgressColumn):
    """Masque une colonne pour les étapes qui ne rapportent pas leur avancement."""

    def __init__(self, inner: ProgressColumn) -> None:
        super().__init__(table_column=inner.get_table_column())
        self._inner = inner

    def render(self, task: Task):
        return self._inner.render(task) if task.fields["measured"] else Text("")


class _DurationColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        return Text(format_duration(elapsed or 0.0), style="dim")


class Steps:
    """Étapes numérotées « ✓ [2/5] Transcription ━━━━ 100 % 21 s ».

    Une étape = une ligne. Dans un terminal, elle s'anime pendant l'exécution ;
    redirigée vers un fichier, seule sa version finale est écrite.
    """

    def __init__(self, total: int) -> None:
        self._total = total
        self._index = 0

    @contextmanager
    def step(self, label: str) -> Iterator[Callable[[float], None]]:
        """Fournit un rappel `report(pourcentage)`, facultatif.

        Chaque étape est sa propre table rich : les largeurs de colonnes sont
        fixes pour que les lignes restent alignées d'une étape à l'autre.
        """
        self._index += 1
        progress = Progress(
            _StatusColumn(finished_text=Text("✓", style="bold green")),
            TextColumn("{task.description}", table_column=Column(width=32)),
            _MeasuredColumn(BarColumn(bar_width=20, table_column=Column(width=20))),
            _MeasuredColumn(
                TextColumn("{task.percentage:>3.0f} %", table_column=Column(width=5))
            ),
            _DurationColumn(table_column=Column(width=10, justify="right")),
            console=console,
        )
        with progress:
            task = progress.add_task(
                f"[{self._index}/{self._total}] {label}",
                total=100,
                measured=False,
                failed=False,
            )

            def report(percent: float) -> None:
                # À 100 %, rich figerait la durée et cocherait l'étape avant
                # qu'elle ne rende la main (libération du modèle, du GPU…).
                progress.update(task, completed=min(percent, 99.9), measured=True)

            try:
                yield report
            except BaseException:
                progress.update(task, failed=True)
                progress.stop_task(task)
                raise
            progress.update(task, completed=100)
