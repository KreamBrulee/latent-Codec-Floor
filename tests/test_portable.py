"""Portability guard: the core package must import without heavy dependencies.

Enforces D-3 / D-6 (handover section 5.4). Without this test the rule is a
comment, and comments do not hold.

Design, and why:
  * Runs in a fresh subprocess. The pytest process may already have imported
    anything (plugins, other tests), which would hide a violation.
  * Blocks imports with a meta-path hook rather than checking sys.modules
    afterwards. Absence of torch proves nothing on the laptop (it is not
    installed), and `try: import torch except ImportError` would evade a
    sys.modules check on either machine. The hook records every *attempt*.
  * Modules are discovered from the filesystem, not pkgutil, because pkgutil
    skips subpackages that fail to import -- the exact failure we look for.
  * test_guard_can_fail plants violations in a throwaway package. A guard that
    has never been seen to fail has not been shown to work.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = "codecfloor"

# gpu tier + the heavy packages it pulls in, and the viz tier (core-only
# installs are valid per section 5.3). A viz module added later must be
# allowlisted explicitly, not by removing names from this set.
BLOCKED = {
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "peft", "finetrainers", "xformers", "triton",
    "matplotlib", "pandas",
}

_CHECKER = textwrap.dedent(r'''
    import importlib, importlib.abc, json, sys, traceback
    from pathlib import Path

    root, pkg, blocked = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
    pkg_dir = (Path(root) / pkg).resolve()
    preloaded = sorted(m for m in sys.modules if m.partition(".")[0] in blocked)
    attempts = []

    def _culprit():
        f = sys._getframe(2)
        while f is not None:
            p = Path(f.f_code.co_filename)
            if p.is_absolute() and pkg_dir in p.resolve().parents:
                return str(p.resolve().relative_to(pkg_dir.parent))
            f = f.f_back
        return None

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in blocked:
                attempts.append([name, _culprit()])
                raise ImportError(f"blocked by portability guard: {name}")
            return None

    sys.meta_path.insert(0, Block())
    sys.path.insert(0, root)

    modules, errors = [], {}
    for py in sorted(pkg_dir.rglob("*.py")):
        parts = py.relative_to(pkg_dir.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod = ".".join(parts)
        modules.append(mod)
        try:
            importlib.import_module(mod)
        except BaseException as e:
            errors[mod] = traceback.format_exception_only(type(e), e)[-1].strip()

    top = sys.modules.get(pkg)
    print(json.dumps({
        "preloaded": preloaded,
        "modules": modules,
        "errors": errors,
        "attempts": attempts,
        "pkg_file": getattr(top, "__file__", None),
    }))
''')


def _run_checker(root: Path, pkg: str) -> dict:
    # -I: ignore PYTHONPATH and user site, so the environment cannot mask or
    # inject modules; the package is resolved only from `root`.
    out = subprocess.run(
        [sys.executable, "-I", "-c", _CHECKER, str(root), pkg, ",".join(sorted(BLOCKED))],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, f"checker crashed:\n{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_core_imports_without_heavy_dependencies():
    r = _run_checker(REPO, PKG)
    expected = {p for p in (REPO / PKG).rglob("*.py")}

    assert not r["preloaded"], f"blocked modules loaded at interpreter start: {r['preloaded']}"
    # Guards against discovery silently returning nothing (C2).
    assert len(r["modules"]) == len(expected) and len(expected) >= 6, r["modules"]
    # Guards against testing some other copy of the package (C4).
    assert Path(r["pkg_file"]).resolve().parent == (REPO / PKG).resolve(), r["pkg_file"]
    assert not r["attempts"], f"heavy imports attempted at module scope: {r['attempts']}"
    assert not r["errors"], f"modules failed to import: {r['errors']}"


@pytest.mark.parametrize("body", [
    "import torch\n",
    "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
    "from matplotlib import pyplot\n",
])
def test_guard_can_fail(tmp_path: Path, body: str):
    bad = tmp_path / "planted"
    bad.mkdir()
    (bad / "__init__.py").write_text("")
    (bad / "clean.py").write_text("import json\n")
    (bad / "leaky.py").write_text(body)

    r = _run_checker(tmp_path, "planted")

    assert len(r["modules"]) == 3
    culprits = {c for _, c in r["attempts"]}
    assert culprits == {str(Path("planted") / "leaky.py")}, r["attempts"]


def test_lazy_import_inside_function_is_allowed(tmp_path: Path):
    # The pattern section 5.4 prescribes must not be flagged, or the guard
    # would push us away from the correct design.
    ok = tmp_path / "lazy"
    ok.mkdir()
    (ok / "__init__.py").write_text("")
    (ok / "ltx.py").write_text("def load():\n    import torch\n    return torch\n")

    r = _run_checker(tmp_path, "lazy")

    assert r["attempts"] == [] and r["errors"] == {}
