"""License guards on Modal-based download scripts. Per CODEBASE_MAP §8.

Static analysis: every download_*.py file contains the CC0 license check
pattern. Sapient-2's two datasets (ds004996 and ds001740) are both CC0 with
no exceptions.

The previous shell-script subprocess tests are retired along with the
shell scripts themselves. Downloads now run as Modal CPU jobs that write
directly to the `sapient-data` volume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.mark.parametrize("script", sorted(DATA_DIR.glob("download_*.py")))
def test_cc0_script_contains_license_guard(script: Path) -> None:
    text = script.read_text()
    assert "CC0" in text, f"{script.name} contains no CC0 string at all"
    assert "License" in text, f"{script.name} doesn't check the License field"
    assert ("raise RuntimeError" in text or "raise " in text), \
        f"{script.name} has no exception path on bad license"
    assert "ABORT" in text, f"{script.name} has no ABORT message"


def test_ds001740_references_openneuro() -> None:
    """ds001740 sync target points at openneuro.org (S3 mirror).

    Spec §2.1 originally called for v2.1.0 pin, but OpenNeuro's S3 mirror
    only exposes the latest revision (s3://openneuro.org/ds001740/versions/
    2.1.0/ is empty). The script documents this deviation inline and uses
    the dataset root; true v2.1.0 reproducibility would require a
    datalad-based clone of the OpenNeuroDatasets GitHub mirror.
    """
    script = DATA_DIR / "download_ds001740.py"
    text = script.read_text()
    assert "openneuro.org/ds001740" in text
    # The deviation must be explained in a comment so it's not a silent
    # spec violation.
    assert "2.1.0" in text, \
        "Script must reference v2.1.0 (either as pin or as deviation note)"


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
    """The shell download scripts were retired when downloads moved to Modal."""
    leftovers = sorted(DATA_DIR.glob("download_*.sh"))
    assert not leftovers, (
        f"Expected zero shell download scripts; found {[p.name for p in leftovers]}. "
        "All downloads must be Modal-resident Python apps per the 2026-05-26 "
        "architectural pivot."
    )
