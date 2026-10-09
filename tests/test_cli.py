"""Tests du CLI."""

from __future__ import annotations

from meeting_transcriber.cli import _gpu_supported

ARCH_CU126 = ["sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


class TestGpuSupported:
    def test_exact_arch(self):
        assert _gpu_supported((8, 6), ARCH_CU126)

    def test_newer_minor_runs_older_binary(self):
        # Une RTX 4070 (sm_89) exécute le code compilé pour sm_86.
        assert _gpu_supported((8, 9), ARCH_CU126)

    def test_newer_major_is_not_supported(self):
        # Une RTX 5070 (sm_120) n'exécute pas le code compilé pour sm_90.
        assert not _gpu_supported((12, 0), ARCH_CU126)

    def test_older_than_every_arch(self):
        assert not _gpu_supported((5, 2), ARCH_CU126)

    def test_three_digit_arch(self):
        assert _gpu_supported((12, 0), ["sm_90", "sm_100", "sm_120"])
