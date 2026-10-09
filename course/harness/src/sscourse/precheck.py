"""The contract pre-check before any test runs (DESIGN 5.4, exit 4).

contracts/  content hash against contracts/VERSION (tree.precheck_contracts)
manifests   Rust workspace and lib targets; Go module name and the
            contracts `replace` (2.15)
Python      every def and class in the unit's .pyi exists in the unit with
            the same parameter names (an AST comparison standing in for
            mypy.stubtest, which needs mypy in the learner env)
C           `cc -fsyntax-only`, then an nm diff: every defined global symbol
            is declared in the contract headers (Mach-O `_` normalized)
"""

from __future__ import annotations

import ast
import platform
import re
import tomllib
from pathlib import Path

from . import ctx, markers
from .registry import Module


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += ["kw:" + x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return names


def _api(tree: ast.Module) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = _params(node)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = {
                n.name: _params(n)
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return out


def python_unit(unit_text: str, pyi_text: str, unit: str) -> list[str]:
    try:
        have = _api(ast.parse(unit_text))
    except SyntaxError as e:
        return [f"{unit}:{e.lineno}: {e.msg}"]
    want = _api(ast.parse(pyi_text))
    errs = []
    for name, sig in want.items():
        if name not in have:
            errs.append(f"{unit}: missing `{name}` from the contract")
        elif isinstance(sig, dict):
            got = have[name]
            if not isinstance(got, dict):
                errs.append(f"{unit}: `{name}` must be a class")
                continue
            for meth, msig in sig.items():
                if meth not in got:
                    errs.append(f"{unit}: `{name}.{meth}` missing")
                elif got[meth] != msig:
                    errs.append(
                        f"{unit}: `{name}.{meth}` parameters {got[meth]} differ from the contract {msig}"
                    )
        elif have[name] != sig:
            errs.append(
                f"{unit}: `{name}` parameters {have[name]} differ from the contract {sig}"
            )
    return errs


def pyi_for(contracts: Path, unit: str) -> Path:
    rel = unit.split("/", 1)[1] if unit.startswith("python/") else unit
    return contracts / "py" / (rel[:-3] + ".pyi")


def header_names(inc: Path) -> set[str]:
    names: set[str] = set()
    for h in inc.rglob("*.h") if inc.is_dir() else []:
        text = re.sub(r"/\*.*?\*/", " ", h.read_text(), flags=re.S)
        names |= set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text))
        names |= set(re.findall(r"\bextern\b[^;(]*?\b([A-Za-z_]\w*)\s*;", text))
    return names


def defined_globals(obj: Path) -> list[str]:
    rc, out = ctx.run(["nm", "-g", str(obj)])
    if rc != 0:
        return []
    syms = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2] not in ("U", "w", "v") and parts[-2].isupper():
            name = parts[-1]
            if platform.system() == "Darwin" and name.startswith("_"):
                name = name[1:]
            syms.append(name)
    return syms


def c_unit(learner: Path, unit: str, scratch: Path) -> list[str]:
    inc = learner / "contracts" / "c" / "include"
    src = learner / unit
    obj = scratch / (unit.replace("/", "_") + ".o")
    obj.parent.mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wno-unused-parameter",
            f"-I{inc}",
            "-c",
            str(src),
            "-o",
            str(obj),
        ]
    )
    if rc != 0:
        return [
            f"{unit} does not compile against the contract headers:\n{ctx.indent(ctx.tail(out, 20))}"
        ]
    declared = header_names(inc)
    extra = [s for s in defined_globals(obj) if s not in declared]
    if extra:
        return [
            f"{unit} exports {', '.join(sorted(extra))}, which no contract header declares; make them static"
        ]
    return []


def rust_manifests(learner: Path, unit_list: list[str], ref_rust: Path) -> list[str]:
    errs = []
    ws = learner / "rust" / "Cargo.toml"
    if not ws.is_file():
        return [
            "rust/Cargo.toml is missing (ss start writes it when absent; restore it)"
        ]
    try:
        doc = tomllib.loads(ws.read_text())
    except tomllib.TOMLDecodeError as e:
        return [f"rust/Cargo.toml: {e}"]
    if "workspace" not in doc:
        errs.append("rust/Cargo.toml must be a [workspace]")
    for u in unit_list:
        parts = u.split("/")
        if len(parts) < 4 or parts[1] != "crates":
            continue
        crate = parts[2]
        man = learner / "rust" / "crates" / crate / "Cargo.toml"
        if not man.is_file():
            errs.append(f"rust/crates/{crate}/Cargo.toml is missing")
            continue
        m = tomllib.loads(man.read_text())
        name = m.get("package", {}).get("name")
        if name != crate:
            errs.append(
                f"rust/crates/{crate}/Cargo.toml: package name must be {crate!r} (contract name), got {name!r}"
            )
        if "lib" not in m and not (man.parent / "src" / "lib.rs").is_file():
            errs.append(f"rust/crates/{crate}: needs a lib target (src/lib.rs)")
        if crate == "tl-py":
            errs += tl_py_contract(learner, m)
    return errs


def tl_py_contract(learner: Path, man: dict) -> list[str]:
    """The tl-py build contract (DESIGN 2.5): a cdylib, PyO3 with `abi3-py311`
    and `extension-module`, and on macOS `-undefined dynamic_lookup` in
    rust/.cargo/config.toml or the crate's build.rs (no maturin needed)."""
    errs = []
    where = "rust/crates/tl-py/Cargo.toml"
    if "cdylib" not in (man.get("lib") or {}).get("crate-type", []):
        errs.append(f'{where}: [lib] crate-type must include "cdylib"')
    py = (man.get("dependencies") or {}).get("pyo3")
    feats = set(py.get("features", [])) if isinstance(py, dict) else set()
    for f in ("abi3-py311", "extension-module"):
        if f not in feats:
            errs.append(f"{where}: the pyo3 dependency needs feature {f!r}")
    cfg = learner / "rust" / ".cargo" / "config.toml"
    build = learner / "rust" / "crates" / "tl-py" / "build.rs"
    text = (cfg.read_text() if cfg.is_file() else "") + (
        build.read_text() if build.is_file() else ""
    )
    if "dynamic_lookup" not in text:
        errs.append(
            "tl-py: on macOS the extension links with `-undefined dynamic_lookup`; put the "
            "link args in rust/.cargo/config.toml ([target.'cfg(target_os = \"macos\")'] rustflags) or build.rs"
        )
    return errs


def go_manifest(learner: Path, ref_go: Path) -> list[str]:
    gm = learner / "go" / "go.mod"
    if not gm.is_file():
        return ["go/go.mod is missing (ss start writes it when absent; restore it)"]
    text = gm.read_text()
    want = "tinyllm"
    ref = ref_go / "go.mod"
    if ref.is_file():
        m = re.search(r"^module\s+(\S+)", ref.read_text(), re.M)
        want = m.group(1) if m else want
    m = re.search(r"^module\s+(\S+)", text, re.M)
    errs = []
    if not m or m.group(1) != want:
        errs.append(f"go/go.mod: module must be {want!r}")
    if (learner / "contracts" / "go" / "go.mod").is_file():
        if not re.search(
            r"supersource\.urmzd\.com/tl/contracts\s*(v\S+\s*)?=>\s*\.\./contracts/go",
            text,
        ):
            errs.append(
                "go/go.mod: needs `replace supersource.urmzd.com/tl/contracts => ../contracts/go`"
            )
    return errs


def module(learner: Path, course: Path, m: Module, scratch: Path) -> list[str]:
    errs: list[str] = []
    contracts = learner / "contracts"
    langs = {markers.lang_of(u) for u in m.owned}
    for u in m.owned:
        p = learner / u
        if not p.is_file():
            continue
        lang = markers.lang_of(u)
        if lang == "python":
            pyi = pyi_for(contracts, u)
            if pyi.is_file():
                errs += python_unit(p.read_text(), pyi.read_text(), u)
        elif lang == "c" and u.endswith(".c"):
            errs += c_unit(learner, u, scratch)
    if "rust" in langs:
        errs += rust_manifests(learner, m.owned, course / "ref" / "rust")
    if "go" in langs:
        errs += go_manifest(learner, course / "ref" / "go")
    return errs
