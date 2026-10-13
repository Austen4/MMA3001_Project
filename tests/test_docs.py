"""Documentation checks: docstrings exist and docs/ matches the current code."""
import importlib.util
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import building_occupancy

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _public_objects():
    """Every public function, class and public method defined in the package."""
    for info in pkgutil.iter_modules(building_occupancy.__path__):
        module = importlib.import_module(f"building_occupancy.{info.name}")
        yield module.__name__, module
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                yield f"{module.__name__}.{name}", obj
            if inspect.isclass(obj):
                for meth_name, meth in vars(obj).items():
                    if inspect.isfunction(meth) and not meth_name.startswith("_"):
                        yield f"{module.__name__}.{name}.{meth_name}", meth


def test_every_public_object_has_a_docstring():
    missing = [name for name, obj in _public_objects() if not inspect.getdoc(obj)]
    assert not missing, f"missing docstrings: {missing}"


def _read(path):
    # Git may check HTML out with CRLF line endings on Windows; compare content only.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_html_docs_are_up_to_date(tmp_path):
    pytest.importorskip("pdoc")
    import pdoc

    committed = DOCS / "building_occupancy.html"
    assert committed.exists(), "docs/ missing - run: python scripts/build_docs.py"
    version = re.search(r'name="generator" content="pdoc ([^"]+)"', _read(committed))
    if version and version.group(1) != pdoc.__version__:
        pytest.skip(f"docs built with pdoc {version.group(1)}, installed {pdoc.__version__}")

    spec = importlib.util.spec_from_file_location("build_docs", ROOT / "scripts" / "build_docs.py")
    build_docs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_docs)
    fresh = tmp_path / "docs"
    build_docs.build(fresh)

    rel = lambda base: sorted(p.relative_to(base) for p in base.rglob("*") if p.is_file())  # noqa: E731
    assert rel(fresh) == rel(DOCS), "docs/ file list differs - run: python scripts/build_docs.py"
    stale = [str(f) for f in rel(fresh) if _read(fresh / f) != _read(DOCS / f)]
    assert not stale, f"docs/ out of date ({stale}) - run: python scripts/build_docs.py"
