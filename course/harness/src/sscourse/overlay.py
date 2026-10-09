"""The cumulative overlay (DESIGN 5.4 "Per-language realization").

An Overlay is built for one target module and a `sources` map that says, for
the target and every module in its dependency closure, where that module's
units come from: `learner`, `ref`, or `stub`. Every registry unit outside the
closure is a stub in the compiled languages (C objects, Rust and Go farms);
in Python the learner's own file is used when present and a stub fills the
gap otherwise. Nothing is ever written into the learner's repo or into the
course tree: everything lands under `work` (the learner's .ss/overlay/<ID>).

  Python  PYTHONPATH = pyext : subst : <learner>/python : stubs : <course>/tests
          subst holds every closure unit that is not the learner's own file
          (reference, stub, or mutant), so it shadows the learner's copy.
  C       one object per unit (learner, ref, or stub), compiled twice: an
          ASan+UBSan test binary per test dir, and an -O2 libtinyllm for
          ctypes and Rust. Objects are cached by content hash.
  Rust    a copy farm of rust/ with a generated workspace manifest and the
          `ss-tests` crate; files are rewritten only when their bytes change
          so Cargo's mtime freshness keeps builds warm. There is one farm per
          learner (.ss/rust-farm), shared by every module, because Cargo
          fingerprints by path: per-module farms over one target dir go stale.
  Go      a copy farm of go/ with a generated go.mod and a go.work over the
          farm, contracts/go, and course/tests/go.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, ctx, ids, markers, tomlw, units
from .registry import Registry

SAN_FLAGS = [
    "-std=c11",
    "-O1",
    "-g",
    "-fsanitize=address,undefined",
    "-fno-omit-frame-pointer",
    "-fno-sanitize-recover=undefined",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
]
LIB_FLAGS = ["-std=c11", "-O2", "-fPIC", "-Wall", "-Wextra", "-Wno-unused-parameter"]
# A third C build for modules whose [tests].sanitize lists "thread" (rt.03's
# pool, DESIGN 4.4): ThreadSanitizer cannot share a binary with ASan.
TSAN_FLAGS = [
    "-std=c11",
    "-O1",
    "-g",
    "-fsanitize=thread",
    "-fno-omit-frame-pointer",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
]
PY_TEST_DEPS = ["pytest", "pytest-randomly", "hypothesis", "numpy"]
SKIP_DIRS = {"target", ".git", "__pycache__", ".venv", "node_modules"}
CONTRACTS_GO_MODULE = "supersource.urmzd.com/tl/contracts"


@dataclass
class TestRun:
    lang: str
    ok: bool
    compiled: bool
    output: str


@dataclass
class Resolved:
    text: str | None  # None: use the learner's file in place (Python only)
    origin: str  # learner | ref | stub | patch | learner-outside
    active: bool  # the target or in its closure


def _walk(d: Path):
    if not d.is_dir():
        return
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.is_file():
            yield rel.as_posix(), p


def _sync(farm: Path, desired: dict[str, bytes], keep: set[str] = frozenset()) -> None:
    """Make farm hold exactly `desired`, touching only files whose bytes changed."""
    farm.mkdir(parents=True, exist_ok=True)
    for rel, data in desired.items():
        p = farm / rel
        if p.is_file() and p.read_bytes() == data:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for rel, p in list(_walk(farm)):
        if rel not in desired and rel not in keep:
            p.unlink()


def _sha(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for x in parts:
        h.update(x if isinstance(x, bytes) else x.encode())
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class Overlay:
    reg: Registry
    course: Path
    target: str
    sources: dict[str, str]
    learner: Path | None
    work: Path
    target_dir: Path  # CARGO_TARGET_DIR and the C object cache live here
    patches: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    _files: dict[str, Resolved] = field(default_factory=dict)
    _lib: Path | None = None
    _pyext: Path | None = None

    # -- resolution -----------------------------------------------------------

    @property
    def contracts(self) -> Path:
        return self.learner / "contracts" if self.learner else self.course / "contracts"

    def resolve(self, unit: str) -> Resolved:
        if unit in self._files:
            return self._files[unit]
        chain = self.reg.unit_chain(unit)
        active = [o for o in chain if o in self.sources]
        if active:
            owner = active[-1]
            src = self.sources[owner]
            if src == "learner":
                t = units.learner_text(self.learner, unit)
                r = (
                    Resolved(t, "learner", True)
                    if t is not None
                    else Resolved(
                        units.stub_text(self.course, self.reg, unit, owner),
                        "stub",
                        True,
                    )
                )
            elif src == "ref":
                r = Resolved(
                    units.ref_text(self.course, self.reg, unit, owner), "ref", True
                )
            else:
                r = Resolved(
                    units.stub_text(self.course, self.reg, unit, owner), "stub", True
                )
        else:
            owner = chain[-1]
            if (
                markers.lang_of(unit) == "python"
                and self.learner
                and (self.learner / unit).is_file()
            ):
                r = Resolved(None, "learner-outside", False)
            else:
                r = Resolved(
                    units.stub_text(self.course, self.reg, unit, owner), "stub", False
                )
        if unit in self.patches:
            r = Resolved(self.patches[unit], "patch", r.active)
        self._files[unit] = r
        return r

    def units_in(self, lang: str) -> list[str]:
        """Library units of one language: those under its root (python/, c/,
        rust/, go/). Primer exercises (primers/<ID>/) are checked by their own
        `check` script and never join the shared builds."""
        return [
            u
            for u in self.reg.all_units()
            if markers.lang_of(u) == lang and u.startswith(f"{lang}/")
        ]

    def sources_summary(self) -> dict[str, str]:
        return {k: v for k, v in self.sources.items() if k != self.target}

    def env(self) -> dict:
        env = ctx.base_env()
        env.pop("VIRTUAL_ENV", None)
        env.update(
            {
                "SS_SEED": str(self.seed),
                "SS_MODULE": self.target,
                "SS_OVERLAY": str(self.work),
                "TINYLLM_FIXTURES": str(self.course / "fixtures"),
                "TINYLLM_CACHE": str(ctx.cache_dir()),
            }
        )
        if self._lib:
            env["TINYLLM_LIB"] = str(self._lib)
            env["TINYLLM_C_LIB_DIR"] = str(self._lib.parent)
        return env

    # -- tests on disk --------------------------------------------------------

    def test_dir(self, mid: str) -> Path:
        m = self.reg.get(mid)
        d = m.tests.get("dir")
        if d and not d.startswith("course/tests/go") and not d.endswith(".rs"):
            return (
                self.course.parent / d if d.startswith("course/") else self.course / d
            )
        return self.course / "tests" / mid

    def test_langs(self, mid: str) -> list[str]:
        out = []
        d = self.test_dir(mid)
        if any(d.glob("test_*.py")) or any(d.glob("*_test.py")):
            out.append("python")
        if any(d.glob("*.c")):
            out.append("c")
        if (self.course / "tests" / "rust" / f"{ids.underscore(mid)}.rs").is_file():
            out.append("rust")
        g = self.course / "tests" / "go" / ids.underscore(mid)
        if g.is_dir() and any(g.rglob("*_test.go")):
            out.append("go")
        return out

    # -- compile checks (verify checks 2 and 14, the stubbed-tree lint) -------

    def compile_python(self) -> tuple[bool, str]:
        errs = []
        for u in self.units_in("python"):
            r = self.resolve(u)
            text = r.text if r.text is not None else units.learner_text(self.learner, u)
            try:
                compile(text or "", u, "exec")
            except SyntaxError as e:
                errs.append(f"{u}:{e.lineno}: {e.msg}")
        return (not errs), "\n".join(errs)

    # -- Python ---------------------------------------------------------------

    def build_python(self) -> tuple[Path, Path]:
        subst = self.work / "subst" / "python"
        stubs = self.work / "stubs" / "python"
        for d in (subst, stubs):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        for u in self.units_in("python"):
            r = self.resolve(u)
            if r.origin in ("learner", "learner-outside"):
                continue
            rel = u.split("/", 1)[1] if u.startswith("python/") else u
            dest = (subst if r.active else stubs) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(r.text or "")
        return subst, stubs

    def _py_project(self) -> tuple[Path, dict]:
        extra = {}
        if self.learner and (self.learner / "python" / "pyproject.toml").is_file():
            return self.learner / "python", extra
        src = self.course / "ref" / "python" / "pyproject.toml"
        proj = self.work / "pyproject"
        proj.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            data = src.read_bytes()
        else:
            data = b'[project]\nname = "ss-verify"\nversion = "0.0.0"\nrequires-python = ">=3.11"\ndependencies = ["numpy"]\n'
        if (
            not (proj / "pyproject.toml").is_file()
            or (proj / "pyproject.toml").read_bytes() != data
        ):
            (proj / "pyproject.toml").write_bytes(data)
        extra["UV_PROJECT_ENVIRONMENT"] = str(ctx.cache_dir() / "verify-venv")
        return proj, extra

    def py_env(self) -> tuple[list[str], dict]:
        """The `uv run` prefix and environment every Python run in this overlay
        uses: the learner's project env plus the harness test deps, and
        PYTHONPATH = pyext : subst : learner/python : stubs : course tests :
        python testkit."""
        subst, stubs = self.build_python()
        proj, extra = self._py_project()
        env = self.env()
        env.update(extra)
        if self.needs_pyext():
            ext, err = self.build_pyext(env)
            if ext is None:
                raise HarnessError("building tinyllm_rs (tl-py) failed:\n" + err)
            env["TINYLLM_PYEXT_DIR"] = str(ext)
        path = [str(subst)]
        if self.learner:
            path.append(str(self.learner / "python"))
        path += [str(stubs), str(self.course / "tests")]
        tk = self.course / "testkit" / "python"
        if tk.is_dir():
            path.append(str(tk))
        if env.get("TINYLLM_PYEXT_DIR"):
            path.insert(0, env["TINYLLM_PYEXT_DIR"])
        env["PYTHONPATH"] = os.pathsep.join(path)
        env["PYTHONPYCACHEPREFIX"] = str(self.work / "pycache")
        cmd = ["uv", "run", "--project", str(proj), "--quiet"]
        for dep in PY_TEST_DEPS:
            cmd += ["--with", dep]
        return cmd, env

    def run_python(self, mid: str, names: list[str] | None, timeout: float) -> TestRun:
        try:
            prefix, env = self.py_env()
        except HarnessError as e:
            return TestRun("python", False, False, str(e))
        cmd = prefix + [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--randomly-seed={self.seed}",
            f"--rootdir={self.course / 'tests'}",
            f"--confcutdir={self.course / 'tests'}",
            str(self.test_dir(mid)),
        ]
        if names:
            cmd += ["-k", " or ".join(names)]
        rc, out = ctx.run(cmd, cwd=self.work, env=env, timeout=timeout)
        if rc == 5 and names:  # pytest: no tests collected
            return TestRun("python", False, True, out + f"\nno test matched {names}")
        compiled = not re.search(
            r"(SyntaxError|IndentationError|ImportError|ModuleNotFoundError) while|ERROR collecting",
            out,
        )
        return TestRun("python", rc == 0, compiled, out)

    # -- C ----------------------------------------------------------------------

    def _c_include(self) -> Path:
        return self.contracts / "c" / "include"

    def _c_obj(
        self, unit: str, r: Resolved, flags: list[str], tag: str
    ) -> tuple[Path | None, str]:
        inc = self._c_include()
        key = _sha(unit, r.text or "", " ".join(flags), _headers_hash(inc))
        cache = self.target_dir / "objcache" / tag
        cache.mkdir(parents=True, exist_ok=True)
        obj = cache / f"{key}.o"
        if obj.is_file():
            return obj, ""
        if r.origin == "learner" and self.learner:
            src = self.learner / unit
        else:
            src = self.work / "c-src" / unit
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(r.text or "")
        rc, out = ctx.run(
            ["cc", *flags, f"-I{inc}", "-c", str(src), "-o", str(obj) + ".tmp"]
        )
        if rc != 0:
            return None, f"[{r.origin}] {unit}\n{out}"
        os.replace(str(obj) + ".tmp", obj)
        return obj, ""

    def c_objects(self, flags: list[str], tag: str) -> tuple[list[Path], str]:
        objs, errs = [], []
        for u in self.units_in("c"):
            if not u.endswith(".c"):
                continue
            o, e = self._c_obj(u, self.resolve(u), flags, tag)
            if o:
                objs.append(o)
            else:
                errs.append(e)
        return objs, "\n".join(errs)

    def build_c_lib(self) -> tuple[Path | None, str]:
        objs, err = self.c_objects(LIB_FLAGS, "lib")
        if err:
            return None, err
        out = self.work / "build"
        out.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Darwin":
            lib = out / "libtinyllm.dylib"
            cmd = [
                "cc",
                "-dynamiclib",
                "-install_name",
                "@rpath/libtinyllm.dylib",
                "-o",
                str(lib),
                *map(str, objs),
            ]
        else:
            lib = out / "libtinyllm.so"
            cmd = ["cc", "-shared", "-o", str(lib), *map(str, objs), "-lm", "-lpthread"]
        rc, msg = ctx.run(cmd)
        if rc != 0:
            return None, msg
        static = out / "libtinyllm.a"
        static.unlink(missing_ok=True)
        rc, msg = ctx.run(["ar", "rcs", str(static), *map(str, objs)])
        if rc != 0:
            return None, msg
        self._lib = lib
        return lib, ""

    def run_c(self, mid: str, names: list[str] | None, timeout: float) -> TestRun:
        r = self._run_c_build(mid, names, timeout, SAN_FLAGS, "asan")
        sans = self.reg.get(mid).tests.get("sanitize", [])
        if r.ok and "thread" in sans and os.environ.get("SS_TSAN", "1") != "0":
            t = self._run_c_build(mid, names, timeout, TSAN_FLAGS, "tsan")
            return TestRun("c", t.ok, t.compiled, r.output + "\n[tsan]\n" + t.output)
        return r

    def _run_c_build(
        self,
        mid: str,
        names: list[str] | None,
        timeout: float,
        flags: list[str],
        tag: str,
    ) -> TestRun:
        objs, err = self.c_objects(flags, tag)
        if err:
            return TestRun("c", False, False, err)
        tests = sorted(self.test_dir(mid).glob("*.c"))
        out_dir = self.work / "build" / mid
        out_dir.mkdir(parents=True, exist_ok=True)
        exe = out_dir / f"test-{tag}"
        cmd = [
            "cc",
            *flags,
            "-DSS_COUNTING_ALLOC=1",
            f"-I{self._c_include()}",
            *map(str, tests),
            *map(str, objs),
            "-o",
            str(exe),
        ]
        if platform.system() != "Darwin":
            cmd += ["-lm", "-lpthread"]
        rc, out = ctx.run(cmd)
        if rc != 0:
            return TestRun("c", False, False, out)
        env = self.env()
        env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        env["ASAN_OPTIONS"] = (
            "detect_leaks=1" if platform.system() == "Linux" else "detect_leaks=0"
        )
        env["TSAN_OPTIONS"] = "halt_on_error=1:second_deadlock_stack=1"
        if names:
            env["SS_ONLY"] = ",".join(names)
        rc, out = ctx.run([str(exe)], cwd=self.work, env=env, timeout=timeout)
        return TestRun("c", rc == 0, True, out)

    def compile_c(self) -> tuple[bool, str]:
        _, err = self.c_objects(LIB_FLAGS, "lib")
        return (not err), err

    # -- Rust -------------------------------------------------------------------

    def _fix_paths(self, manifest: dict, orig_dir: Path, rust_root: Path) -> dict:
        def fix(deps: dict) -> None:
            for name, spec in deps.items():
                if isinstance(spec, dict) and "path" in spec:
                    target = (orig_dir / spec["path"]).resolve()
                    try:
                        target.relative_to(rust_root.resolve())
                        continue  # stays inside the farm: keep relative
                    except ValueError:
                        pass
                    try:
                        rest = target.relative_to(
                            (rust_root.parent / "contracts").resolve()
                        )
                        target = self.contracts / rest
                    except ValueError:
                        pass
                    spec["path"] = str(target)

        for sect in ("dependencies", "dev-dependencies", "build-dependencies"):
            if isinstance(manifest.get(sect), dict):
                fix(manifest[sect])
        for tgt in (manifest.get("target") or {}).values():
            for sect in ("dependencies", "dev-dependencies", "build-dependencies"):
                if isinstance(tgt.get(sect), dict):
                    fix(tgt[sect])
        ws = manifest.get("workspace", {})
        if isinstance(ws.get("dependencies"), dict):
            fix(ws["dependencies"])
        return manifest

    def build_rust(self, test_ids: list[str]) -> Path:
        # One farm per learner (not per module): Cargo keys freshness by path
        # and mtime, so two farms sharing CARGO_TARGET_DIR would hand one the
        # other's stale artifacts. Rewriting only changed files in one place
        # keeps builds warm and correct.
        farm = self.target_dir / "rust-farm"
        desired: dict[str, bytes] = {}
        origin_dir: dict[str, Path] = {}
        unit_set = {u for u in self.units_in("rust")}
        roots = [self.course / "ref" / "rust"]
        if self.learner:
            roots.append(self.learner / "rust")
        for root in roots:
            for rel, p in _walk(root):
                if f"rust/{rel}" in unit_set or rel == "Cargo.lock":
                    continue
                desired[rel] = p.read_bytes()
                origin_dir[rel] = root
        for u in unit_set:
            desired[u.split("/", 1)[1]] = (self.resolve(u).text or "").encode()
        # Manifests: fix path deps, then add the ss-tests member.
        crates: list[tuple[str, str]] = []  # (dir, package name) of lib crates
        for rel in sorted(desired):
            if not rel.endswith("Cargo.toml") or rel == "Cargo.toml":
                continue
            d = rel[: -len("Cargo.toml")].rstrip("/")
            man = tomllib.loads(desired[rel].decode())
            root = origin_dir.get(rel, self.course / "ref" / "rust")
            man = self._fix_paths(man, (root / d).resolve(), root)
            desired[rel] = tomlw.dumps(man).encode()
            pkg = man.get("package", {}).get("name")
            if pkg and ("lib" in man or f"{d}/src/lib.rs" in desired):
                crates.append((d, pkg))
        ws = (
            tomllib.loads(desired["Cargo.toml"].decode())
            if "Cargo.toml" in desired
            else {}
        )
        ws.setdefault("workspace", {})
        ws["workspace"].setdefault("resolver", "2")
        members = list(ws["workspace"].get("members", []))
        if not members:
            members = [d for d, _ in crates]
        if "ss-tests" not in members:
            members.append("ss-tests")
        ws["workspace"]["members"] = members
        root = (
            self.learner / "rust"
            if self.learner and (self.learner / "rust").is_dir()
            else self.course / "ref" / "rust"
        )
        desired["Cargo.toml"] = tomlw.dumps(
            self._fix_paths(ws, root.resolve(), root)
        ).encode()
        # tl-py is a Python extension (its symbols resolve inside the
        # interpreter), never a dependency of the course test crate.
        deps = {pkg: {"path": f"../{d}"} for d, pkg in crates if pkg != "tl-py"}
        tlc = self.contracts / "rust" / "tl-contracts"
        if (tlc / "Cargo.toml").is_file():
            deps["tl-contracts"] = {"path": str(tlc)}
        tk = self.course / "testkit" / "rust" / "tl-testkit"
        if (tk / "Cargo.toml").is_file():
            deps["tl-testkit"] = {"path": str(tk)}  # failpoints and the fake clock
        desired["ss-tests/Cargo.toml"] = tomlw.dumps(
            {
                "package": {
                    "name": "ss-tests",
                    "version": "0.0.0",
                    "edition": "2021",
                    "publish": False,
                },
                "dependencies": deps,
            }
        ).encode()
        desired["ss-tests/src/lib.rs"] = (
            b"// generated by ss: course tests live in tests/\n"
        )
        for mid in test_ids:
            name = ids.underscore(mid)
            src = self.course / "tests" / "rust" / f"{name}.rs"
            if src.is_file():
                desired[f"ss-tests/tests/{name}.rs"] = src.read_bytes()
        keep = {"Cargo.lock"}
        _sync(farm, desired, keep)
        lock = self.learner / "rust" / "Cargo.lock" if self.learner else None
        if lock and lock.is_file() and not (farm / "Cargo.lock").is_file():
            shutil.copy2(lock, farm / "Cargo.lock")
        return farm

    def _cargo_env(self) -> dict:
        env = self.env()
        env["CARGO_TARGET_DIR"] = str(self.target_dir / "target")
        env["CARGO_TERM_COLOR"] = "never"
        return env

    def run_rust(
        self,
        mid: str,
        names: list[str] | None,
        timeout: float,
        test_ids: list[str] | None = None,
    ) -> TestRun:
        farm = self.build_rust(test_ids or [mid])
        name = ids.underscore(mid)
        base = [
            "cargo",
            "test",
            "-q",
            "--manifest-path",
            str(farm / "Cargo.toml"),
            "-p",
            "ss-tests",
            "--test",
            name,
        ]
        rc, out = ctx.run(
            base + ["--no-run"], env=self._cargo_env(), timeout=max(timeout, 300)
        )
        if rc != 0:
            return TestRun("rust", False, False, out)
        cmd = base + ["--", "--test-threads=1"]
        if names:
            cmd += ["--exact", *names]
        rc, out = ctx.run(cmd, env=self._cargo_env(), timeout=timeout)
        return TestRun("rust", rc == 0, True, out)

    def compile_rust(self, all_tests: bool = False) -> tuple[bool, str]:
        test_ids = []
        if all_tests:
            test_ids = [
                m
                for m in self.reg.modules
                if (
                    self.course / "tests" / "rust" / f"{ids.underscore(m)}.rs"
                ).is_file()
            ]
        farm = self.build_rust(test_ids)
        cmd = [
            "cargo",
            "test",
            "-q",
            "--no-run",
            "--workspace",
            "--manifest-path",
            str(farm / "Cargo.toml"),
        ]
        ext = (farm / "crates" / "tl-py" / "Cargo.toml").is_file()
        if (
            ext
        ):  # an extension module links only inside Python: check it, do not link tests
            cmd += ["--exclude", "tl-py"]
        rc, out = ctx.run(cmd, env=self._cargo_env(), timeout=600)
        if rc == 0 and ext:
            rc, more = ctx.run(
                [
                    "cargo",
                    "check",
                    "-q",
                    "--manifest-path",
                    str(farm / "Cargo.toml"),
                    "-p",
                    "tl-py",
                ],
                env=self._cargo_env(),
                timeout=600,
            )
            out += more
        return rc == 0, out

    # -- tl-py: the PyO3 extension tinyllm_rs (DESIGN 2.5) ------------------------

    def _tl_py_manifest(self) -> Path | None:
        for root in ([self.learner / "rust"] if self.learner else []) + [
            self.course / "ref" / "rust"
        ]:
            p = root / "crates" / "tl-py" / "Cargo.toml"
            if p.is_file():
                return p
        return None

    def needs_pyext(self) -> bool:
        """Python tests reach Rust through tinyllm_rs: build it when a tl-py
        crate exists and the module's closure holds a Rust unit."""
        if self._tl_py_manifest() is None:
            return False
        return any(self.resolve(u).active for u in self.units_in("rust"))

    def _interpreter(self, env: dict) -> str:
        """The learner's uv interpreter: PYO3_PYTHON for the tl-py build."""
        proj, extra = self._py_project()
        e = dict(env)
        e.update(extra)
        rc, out = ctx.run(
            [
                "uv",
                "run",
                "--project",
                str(proj),
                "--quiet",
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ],
            env=e,
            timeout=300,
        )
        lines = [x for x in out.splitlines() if x.strip()]
        if rc != 0 or not lines:
            raise HarnessError(
                f"cannot find the learner's Python for PYO3_PYTHON:\n{out}"
            )
        return lines[-1].strip()

    def build_pyext(self, env: dict | None = None) -> tuple[Path | None, str]:
        """`cargo rustc -p tl-py --lib --crate-type cdylib` in the shared farm
        with PYO3_PYTHON set to the learner's interpreter and, on macOS, the
        `-undefined dynamic_lookup` link args (no maturin); the library is
        copied to <work>/pyext/tinyllm_rs.so, which goes first on PYTHONPATH."""
        if self._pyext is not None:
            return self._pyext, ""
        farm = self.build_rust([])
        man_path = farm / "crates" / "tl-py" / "Cargo.toml"
        if not man_path.is_file():
            return None, "no rust/crates/tl-py/Cargo.toml in the farm"
        man = tomllib.loads(man_path.read_text())
        pkg = man.get("package", {}).get("name", "tl-py")
        libname = (man.get("lib") or {}).get("name") or pkg.replace("-", "_")
        cenv = self._cargo_env()
        try:
            cenv["PYO3_PYTHON"] = self._interpreter(env or self.env())
        except HarnessError as e:
            return None, str(e)
        cmd = [
            "cargo",
            "rustc",
            "-q",
            "--manifest-path",
            str(farm / "Cargo.toml"),
            "-p",
            pkg,
            "--lib",
            "--crate-type",
            "cdylib",
        ]
        if platform.system() == "Darwin":
            cmd += ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
        rc, out = ctx.run(cmd, env=cenv, timeout=900)
        if rc != 0:
            return None, out
        ext = "dylib" if platform.system() == "Darwin" else "so"
        built = Path(cenv["CARGO_TARGET_DIR"]) / "debug" / f"lib{libname}.{ext}"
        if not built.is_file():
            return None, f"cargo built no {built.name} (is crate-type cdylib?)"
        dest = self.work / "pyext"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "tinyllm_rs.so"
        if not target.is_file() or target.read_bytes() != built.read_bytes():
            # A new inode every time: macOS caches a loaded Mach-O's code
            # signature per vnode, and rewriting the file in place gets the
            # next process that loads it killed.
            tmp = dest / ".tinyllm_rs.so.tmp"
            shutil.copy2(built, tmp)
            os.replace(tmp, target)
        self._pyext = dest
        return dest, ""

    # -- Go -----------------------------------------------------------------------

    def _go_mod(self, text: str, orig_dir: Path) -> str:
        out = []
        in_block = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("replace ("):
                in_block = True
                out.append(line)
                continue
            if in_block and s == ")":
                in_block = False
                out.append(line)
                continue
            spec = (
                s[len("replace ") :]
                if s.startswith("replace ")
                else (s if in_block else None)
            )
            if spec is not None and "=>" in spec:
                left, right = (x.strip() for x in spec.split("=>", 1))
                if left.split()[0] == CONTRACTS_GO_MODULE:
                    continue  # go.work provides the contracts module
                if right.startswith(("./", "../")):
                    right = str((orig_dir / right).resolve())
                    line = ("\t" if in_block else "replace ") + f"{left} => {right}"
            out.append(line)
        return "\n".join(out) + "\n"

    def build_go(self) -> tuple[Path, Path]:
        farm = self.work / "go"
        desired: dict[str, bytes] = {}
        unit_set = set(self.units_in("go"))
        roots = [self.course / "ref" / "go"]
        if self.learner:
            roots.append(self.learner / "go")
        for root in roots:
            for rel, p in _walk(root):
                if f"go/{rel}" in unit_set:
                    continue
                data = p.read_bytes()
                if rel == "go.mod":
                    data = self._go_mod(data.decode(), root.resolve()).encode()
                desired[rel] = data
        for u in unit_set:
            desired[u.split("/", 1)[1]] = (self.resolve(u).text or "").encode()
        if "go.mod" not in desired:
            desired["go.mod"] = b"module tinyllm\n\ngo 1.22\n"
        _sync(farm, desired)
        mods = [farm]
        if (self.contracts / "go" / "go.mod").is_file():
            mods.append(self.contracts / "go")
        if (self.course / "tests" / "go" / "go.mod").is_file():
            mods.append(self.course / "tests" / "go")
        if (self.course / "testkit" / "go" / "go.mod").is_file():
            mods.append(
                self.course / "testkit" / "go"
            )  # supersource.urmzd.com/tl/testkit
        version = max((_go_version(m / "go.mod") for m in mods), default=(1, 22))
        work = self.work / "go.work"
        body = (
            f"go {go_version_str(version)}\n\nuse (\n"
            + "".join(f"\t{m}\n" for m in mods)
            + ")\n"
        )
        if not work.is_file() or work.read_text() != body:
            work.write_text(body)
        return farm, work

    def _go_env(self, work: Path) -> dict:
        env = self.env()
        env["GOWORK"] = str(work)
        env["GOTOOLCHAIN"] = "local"
        env["GOFLAGS"] = "-count=1"
        return env

    def run_go(self, mid: str, names: list[str] | None, timeout: float) -> TestRun:
        farm, work = self.build_go()
        pkg = f"./{ids.underscore(mid)}/..."
        race = ["-race"] if os.environ.get("SS_GO_RACE", "1") != "0" else []
        cwd = self.course / "tests" / "go"
        env = self._go_env(work)
        rc, out = ctx.run(
            ["go", "test", *race, "-run", "^$", pkg],
            cwd=cwd,
            env=env,
            timeout=max(timeout, 300),
        )
        if rc != 0:
            return TestRun("go", False, False, out)
        cmd = ["go", "test", *race, f"-shuffle={self.seed if self.seed else 'off'}"]
        if names:
            cmd += ["-run", "^(" + "|".join(re.escape(n) for n in names) + ")$"]
        cmd.append(pkg)
        rc, out = ctx.run(cmd, cwd=cwd, env=env, timeout=timeout)
        return TestRun("go", rc == 0, True, out)

    def compile_go(self, all_tests: bool = False) -> tuple[bool, str]:
        farm, work = self.build_go()
        env = self._go_env(work)
        rc, out = ctx.run(["go", "build", "./..."], cwd=farm, env=env, timeout=600)
        if rc != 0 or not all_tests:
            return rc == 0, out
        cwd = self.course / "tests" / "go"
        if not (cwd / "go.mod").is_file():
            return True, out
        rc, out2 = ctx.run(
            ["go", "test", "-run", "^$", "./..."], cwd=cwd, env=env, timeout=600
        )
        return rc == 0, out + out2

    # -- practice artifacts -----------------------------------------------------

    def run_practice(self, mid: str, timeout: float) -> TestRun:
        check = self.test_dir(mid) / "check"
        if not os.access(check, os.X_OK):
            return TestRun(
                "check", False, True, f"no executable artifact check at {check}"
            )
        # A practice check that runs your entry points (dep.00 trains through
        # your ctypes loader) gets the same libtinyllm as the course tests:
        # built from your units in its dependency closure.
        if self._lib is None and self.needs_c_lib(["python"]):
            lib, err = self.build_c_lib()
            if lib is None:
                return TestRun(
                    "check", False, False, "building libtinyllm failed:\n" + err
                )
        env = self.env()
        env["SS_COURSE_TREE"] = str(self.course)
        rc, out = ctx.run(
            [str(check)], cwd=self.learner or self.work, env=env, timeout=timeout
        )
        return TestRun("check", rc == 0, True, out)

    # -- one module's tests -----------------------------------------------------

    def needs_c_lib(self, langs: list[str]) -> bool:
        if not ({"python", "rust"} & set(langs)):
            return False
        for u in self.units_in("c"):
            if u.endswith(".c") and self.resolve(u).active:
                return True
        return False

    def run_tests(
        self,
        mid: str,
        names: list[str] | None = None,
        timeout: float | None = None,
        langs: list[str] | None = None,
        rust_tests: list[str] | None = None,
    ) -> list[TestRun]:
        m = self.reg.get(mid)
        timeout = timeout or m.timeout_s
        if m.kind == "practice":
            return [self.run_practice(mid, timeout)]
        langs = langs or self.test_langs(mid)
        runs: list[TestRun] = []
        if self.needs_c_lib(langs) and self._lib is None:
            lib, err = self.build_c_lib()
            if lib is None:
                return [
                    TestRun("c", False, False, "building libtinyllm failed:\n" + err)
                ]
        for lang in langs:
            if lang == "python":
                runs.append(self.run_python(mid, names, timeout))
            elif lang == "c":
                runs.append(self.run_c(mid, names, timeout))
            elif lang == "rust":
                runs.append(self.run_rust(mid, names, timeout, rust_tests))
            elif lang == "go":
                runs.append(self.run_go(mid, names, timeout))
        return runs

    # -- the learner's graded tests (DESIGN 5.6), run against this overlay -------

    def learner_test_files(self, rel: str) -> list[Path]:
        return graded_files(self.learner, rel) if self.learner else []

    def run_learner_tests(
        self, rel: str, timeout: float, extra_env: dict | None = None
    ) -> TestRun:
        """Run the graded tests at <learner>/<rel> against this overlay's units
        (reference, reference plus one mutant, or the learner's own)."""
        lang = markers.lang_of(rel) or rel.split("/", 1)[0]
        files = self.learner_test_files(rel)
        if not files:
            return TestRun(lang, False, False, f"no graded test files under {rel}")
        if self.needs_c_lib([lang]) and self._lib is None:
            lib, err = self.build_c_lib()
            if lib is None:
                return TestRun("c", False, False, "building libtinyllm failed:\n" + err)
        if lang == "python":
            try:
                prefix, env = self.py_env()
            except HarnessError as e:
                return TestRun("python", False, False, str(e))
            env.update(extra_env or {})
            base = self.learner / rel
            root = base if base.is_dir() else base.parent
            cmd = prefix + [
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "_lib.ss_hypothesis",
                f"--randomly-seed={self.seed}",
                f"--rootdir={root}",
                f"--confcutdir={root}",
                *map(str, files),
            ]
            rc, out = ctx.run(cmd, cwd=self.work, env=env, timeout=timeout)
            compiled = not re.search(
                r"(SyntaxError|IndentationError|ImportError|ModuleNotFoundError) while|ERROR collecting",
                out,
            )
            return TestRun("python", rc == 0, compiled, out)
        if lang == "c":
            objs, err = self.c_objects(SAN_FLAGS, "asan")
            if err:
                return TestRun("c", False, False, err)
            exe = self.work / "build" / "learner-tests"
            exe.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "cc",
                *SAN_FLAGS,
                "-DSS_COUNTING_ALLOC=1",
                f"-I{self._c_include()}",
                *map(str, files),
                *map(str, objs),
                "-o",
                str(exe),
            ]
            if platform.system() != "Darwin":
                cmd += ["-lm", "-lpthread"]
            rc, out = ctx.run(cmd)
            if rc != 0:
                return TestRun("c", False, False, out)
            env = self.env()
            env.update(extra_env or {})
            env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
            env["ASAN_OPTIONS"] = (
                "detect_leaks=1" if platform.system() == "Linux" else "detect_leaks=0"
            )
            rc, out = ctx.run([str(exe)], cwd=self.work, env=env, timeout=timeout)
            return TestRun("c", rc == 0, True, out)
        if lang == "rust":
            farm = self.build_rust([])
            env = self._cargo_env()
            env.update(extra_env or {})
            by_pkg: dict[str, list[str]] = {}
            for f in files:
                farm_rel = f.relative_to(self.learner / "rust")
                crate = farm_rel.parts[: farm_rel.parts.index("tests")]
                man = tomllib.loads((farm.joinpath(*crate) / "Cargo.toml").read_text())
                pkg = man.get("package", {}).get("name", crate[-1])
                by_pkg.setdefault(pkg, []).append(f.stem)
            outs, ok, compiled = [], True, True
            for pkg, stems in sorted(by_pkg.items()):
                base = [
                    "cargo",
                    "test",
                    "-q",
                    "--manifest-path",
                    str(farm / "Cargo.toml"),
                    "-p",
                    pkg,
                ]
                for st in stems:
                    base += ["--test", st]
                rc, out = ctx.run(
                    base + ["--no-run"], env=env, timeout=max(timeout, 300)
                )
                if rc != 0:
                    return TestRun("rust", False, False, out)
                rc, out = ctx.run(
                    base + ["--", "--test-threads=1"], env=env, timeout=timeout
                )
                outs.append(out)
                ok &= rc == 0
            return TestRun("rust", ok, compiled, "\n".join(outs))
        if lang == "go":
            farm, work = self.build_go()
            env = self._go_env(work)
            env.update(extra_env or {})
            pkgs = sorted(
                {
                    "./" + f.parent.relative_to(self.learner / "go").as_posix()
                    for f in files
                }
            )
            rc, out = ctx.run(
                ["go", "vet", *pkgs], cwd=farm, env=env, timeout=max(timeout, 300)
            )
            if rc != 0:
                return TestRun("go", False, False, out)
            rc, out = ctx.run(
                ["go", "test", f"-shuffle={self.seed if self.seed else 'off'}", *pkgs],
                cwd=farm,
                env=env,
                timeout=timeout,
            )
            return TestRun("go", rc == 0, True, out)
        return TestRun(
            lang, False, False, f"graded tests under {rel}: unknown language"
        )

    def run_cmd(
        self, argv: list[str], timeout: float, extra_env: dict | None = None
    ) -> tuple[int, str]:
        """A learner command (an eval or benchmark script) under this overlay's
        Python environment, from the learner repo."""
        prefix, env = self.py_env()
        if self.needs_c_lib(["python"]) and self._lib is None:
            self.build_c_lib()
            env = self.py_env()[1]
        env.update(extra_env or {})
        cmd = prefix + argv if argv and argv[0] == "python" else argv
        return ctx.run(cmd, cwd=self.learner or self.work, env=env, timeout=timeout)


def graded_files(learner: Path, rel: str) -> list[Path]:
    """The learner's graded test files at `rel` (a file or a directory)."""
    base = learner / rel
    lang = markers.lang_of(rel) or rel.split("/", 1)[0]
    pick = {
        "python": lambda p: (
            p.suffix == ".py"
            and (p.name.startswith("test_") or p.name.endswith("_test.py"))
        ),
        "c": lambda p: p.suffix == ".c",
        "rust": lambda p: p.suffix == ".rs" and "tests" in p.parts,
        "go": lambda p: p.name.endswith("_test.go"),
    }.get(lang, lambda p: False)
    if base.is_file():
        return [base] if pick(base) else []
    if not base.is_dir():
        return []
    deep = lang in ("python", "c")
    it = base.rglob("*") if deep else base.iterdir()
    return sorted(
        p for p in it if p.is_file() and "__pycache__" not in p.parts and pick(p)
    )


def _headers_hash(inc: Path) -> str:
    if not inc.is_dir():
        return ""
    return _sha(
        *(
            p.relative_to(inc).as_posix() + "\0" + p.read_text()
            for p in sorted(inc.rglob("*.h"))
        )
    )


def _go_version(gomod: Path) -> tuple[int, ...]:
    """The go line of a go.mod as a comparable tuple. A release such as
    1.25.0 sorts above the language version 1.25 (Go 1.21 rules), so a
    go.work built from the max accepts a module that says `go 1.25.0`."""
    try:
        m = re.search(r"^go\s+(\d+)\.(\d+)(?:\.(\d+))?", gomod.read_text(), re.M)
    except OSError:
        return (1, 22)
    if not m:
        return (1, 22)
    v = (int(m.group(1)), int(m.group(2)))
    return v + (int(m.group(3)),) if m.group(3) is not None else v


def go_version_str(v: tuple[int, ...]) -> str:
    return ".".join(str(x) for x in v)


def need_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise HarnessError(f"`{tool}` is not on PATH; install it (see `ss doctor`)")
