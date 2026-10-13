"""
Build the HTML documentation of the ``building_occupancy`` package with pdoc.

Usage (from the repository root)::

    python scripts/build_docs.py            # writes docs/
    python scripts/build_docs.py OUT_DIR    # writes elsewhere (used by tests/test_docs.py)

Open ``docs/index.html`` in a browser. Rebuild after every code or docstring change;
``tests/test_docs.py`` fails if ``docs/`` is out of date.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pdoc  # noqa: E402
import pdoc.render  # noqa: E402

DOCS_DIR = ROOT / "docs"


def build(output_dir=DOCS_DIR):
    """Render the package documentation to ``output_dir`` (replacing its contents).

    Parameters
    ----------
    output_dir : str or Path
        Destination directory.

    Returns
    -------
    list of Path
        The files written, sorted.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    pdoc.render.configure(docformat="numpy", show_source=True,
                          footer_text="MMA3001 building occupancy project")
    pdoc.pdoc("building_occupancy", output_directory=output_dir)
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    for f in files:
        f.write_text(_machine_independent(f.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    return files


def _machine_independent(text):
    """Replace this checkout's absolute path and OS-specific Path reprs in rendered defaults.

    Path settings in ``config`` are shown as their default values; without this the docs
    would contain the local user's directory and differ between machines.
    """
    for form in {ROOT.as_posix(), str(ROOT), str(ROOT).replace("\\", "\\\\")}:
        text = text.replace(form, "&lt;repo&gt;")     # escaped: these strings sit inside HTML
    # pdoc adds a "view value" toggle only when a default value's text is long, and a path
    # default's length depends on where the repository is checked out. Drop the toggles
    # (values are then always shown in full) so the output does not depend on that length.
    text = _VALUE_TOGGLE.sub("", text)
    # The class name may be split from "(" by syntax-highlighting tags, so replace the word.
    return text.replace("WindowsPath", "Path").replace("PosixPath", "Path")


_VALUE_TOGGLE = re.compile(
    r'<input id="[^"]*-view-value" class="view-value-toggle-state" type="checkbox"[^>]*>\s*'
    r'<label class="view-value-button pdoc-button" for="[^"]*-view-value"></label>')


def main(argv):
    """Build the docs to ``argv[0]`` if given, else ``docs/``."""
    files = build(Path(argv[0]) if argv else DOCS_DIR)
    print(f"wrote {len(files)} files to {files[0].parent if files else '?'}")


if __name__ == "__main__":
    main(sys.argv[1:])
