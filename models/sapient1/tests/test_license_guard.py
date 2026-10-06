"""License guards on Modal-based download scripts. Per CODEBASE_MAP §8.

Static analysis tier: every download_*.py file contains the CC0 license
check pattern (or, for the Narratives exception, explicitly acknowledges
the features-only / no-redistribution constraint per spec §3).

The previous shell-script subprocess tests are retired along with the
shell scripts themselves. Downloads now run as Modal CPU jobs that write
directly to the `sapient-data` volume — the same volume that downstream
feature-extraction / training Modal apps read from. Subprocess-testing a
Modal script locally is meaningless (it'd need Modal credentials + a
remote function call, not a local exec).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Narratives (ds002345) is the documented non-CC0 exception per spec §3:
# fMRI is openly shared, stimulus audio is copyrighted. We use it features-
# only and never redistribute raw audio. Its download_*.py logs the license
# but doesn't raise.
NON_CC0_EXCEPTIONS = {"download_narratives.py"}


def _cc0_scripts() -> list[Path]:
    return [
        p for p in sorted(DATA_DIR.glob("download_*.py"))
        if p.name not in NON_CC0_EXCEPTIONS
    ]


@pytest.mark.parametrize("script", _cc0_scripts())
def test_cc0_script_contains_license_guard(script: Path) -> None:
    text = script.read_text()
    assert "CC0" in text, f"{script.name} contains no CC0 string at all"
    assert "License" in text, f"{script.name} doesn't check the License field"
    # Modal scripts must `raise` (not exit 2) on bad license — the function is
    # called from a Modal remote; raising propagates back as a Modal error.
    assert ("raise RuntimeError" in text or "raise " in text), \
        f"{script.name} has no exception path on bad license"
    assert "ABORT" in text, f"{script.name} has no ABORT message"


def test_narratives_acknowledges_non_cc0() -> None:
    """Narratives is the documented exception. Script must explain why."""
    script = DATA_DIR / "download_narratives.py"
    if not script.exists():
        pytest.skip("download_narratives.py not present in this repo")
    text = script.read_text()
    assert "copyrighted" in text or "non-commercial" in text, \
        "narratives.py must acknowledge the copyright/non-commercial constraint"
    assert "feature" in text.lower(), \
        "narratives.py must call out the features-only usage rule"
    assert "redistribute" in text, \
        "narratives.py must mention non-redistribution of the raw audio"


def test_all_downloads_are_modal_apps() -> None:
    """Architectural rule: NO local-disk downloads. Every download_*.py
    declares a `modal.App` and writes through the sapient-data volume."""
    for script in sorted(DATA_DIR.glob("download_*.py")):
        text = script.read_text()
        assert "modal.App(" in text, \
            f"{script.name} is not a Modal app (would write to local disk)"
        assert 'modal.Volume.from_name("sapient-data"' in text, \
            f"{script.name} doesn't mount the shared sapient-data volume"
        assert "volume.commit()" in text, \
            f"{script.name} doesn't volume.commit() its writes"


def test_no_shell_download_scripts() -> None:
    """The shell download scripts were retired when downloads moved to Modal.
    Their presence would imply a local-disk regression."""
    leftovers = sorted(DATA_DIR.glob("download_*.sh"))
    assert not leftovers, (
        f"Expected zero shell download scripts; found {[p.name for p in leftovers]}. "
        "All downloads must be Modal-resident Python apps per the 2026-05-26 "
        "architectural pivot."
    )
