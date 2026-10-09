"""OpenAPI conformance (DESIGN 5.8): the contract, not an implementation.

A suite is named `openapi[:<version>][:<tier>][:smoke]`, for example
`openapi:v0`, `openapi:v1:gateway:smoke`. The cases live in the course tree at
`course/conformance/openapi/cases/*.toml`:

    id       = "v0.stream.framing"
    title    = "SSE framing is byte exact"
    versions = ["v0"]                 # API versions whose suite includes it
    tiers    = ["engine", "gateway"]
    smoke    = true                   # part of the `:smoke` subset
    requires = []                     # module ids; until they pass, the case is `pending`
    check    = "stream_framing"       # a named check in CHECKS below
    [params]
    prompt = "Once upon a time"

Every response body is validated against the schema the contract
(`contracts/openapi/openai-subset.<version>.yaml`) declares for that path,
method, status, and content type. SSE chunks validate against the schema in
the 200 `text/event-stream` content's `x-tl-chunk` extension.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, schema, web

VERSIONS = ("v0", "v1", "v2")
TIERS = ("engine", "gateway")


@dataclass
class Suite:
    version: str = "v0"
    tier: str | None = None
    smoke: bool = False

    @property
    def id(self) -> str:
        return "openapi:" + ":".join(
            [self.version, self.tier or "engine"] + (["smoke"] if self.smoke else [])
        )


def parse_suite(name: str, target: str | None = None) -> Suite:
    parts = name.split(":")
    if parts[0] != "openapi":
        raise HarnessError(
            f"unknown suite {name!r}: `ss conform` knows openapi[:v0|v1|v2][:engine|gateway][:smoke]"
        )
    s = Suite()
    for p in parts[1:]:
        if p in VERSIONS:
            s.version = p
        elif p in TIERS:
            s.tier = p
        elif p == "smoke":
            s.smoke = True
        else:
            raise HarnessError(f"suite {name!r}: unknown part {p!r}")
    if target:
        if target not in TIERS:
            raise HarnessError(f"--target must be engine or gateway, not {target!r}")
        if s.tier and s.tier != target:
            raise HarnessError(
                f"suite {name!r} names tier {s.tier} but --target is {target}"
            )
        s.tier = target
    s.tier = s.tier or "engine"
    return s


@dataclass
class Case:
    id: str
    title: str
    versions: list[str]
    tiers: list[str]
    check: str
    smoke: bool = False
    requires: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def load_cases(course: Path) -> list[Case]:
    d = course / "conformance" / "openapi" / "cases"
    out = []
    for p in sorted(d.glob("*.toml")) if d.is_dir() else []:
        try:
            raw = tomllib.loads(p.read_text())
        except tomllib.TOMLDecodeError as e:
            raise HarnessError(f"{p}: {e}") from None
        c = Case(
            id=raw.get("id", p.stem),
            title=raw.get("title", ""),
            versions=list(raw.get("versions", [])),
            tiers=list(raw.get("tiers", ["engine"])),
            check=raw.get("check", ""),
            smoke=bool(raw.get("smoke", False)),
            requires=list(raw.get("requires", [])),
            params=dict(raw.get("params", {})),
        )
        if c.check not in CHECKS:
            raise HarnessError(
                f"{p}: unknown check {c.check!r} (known: {', '.join(sorted(CHECKS))})"
            )
        out.append(c)
    return out


def load_spec(path: Path) -> dict:
    if not path.is_file():
        raise HarnessError(
            f"the contract {path} is missing; this suite needs it vendored (`ss contracts sync`)"
        )
    import yaml

    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise HarnessError(f"{path}: {e}") from None
    if not isinstance(doc, dict) or "paths" not in doc:
        raise HarnessError(f"{path}: not an OpenAPI document (no `paths`)")
    return doc


# ---------------------------------------------------------------------------
# a client bound to one base URL and the contract


@dataclass
class Client:
    base: str
    tier: str
    spec: dict
    model: str = "tracer"
    api_key: str | None = None
    health_base: str | None = None
    timeout: float = 30.0

    def headers(self, auth: bool = True) -> dict:
        if auth and self.tier == "gateway" and self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def call(
        self,
        method: str,
        path: str,
        body=None,
        auth: bool = True,
        base: str | None = None,
    ) -> web.Response:
        return web.request(
            method,
            (base or self.base).rstrip("/") + path,
            json_body=body,
            headers=self.headers(auth),
            timeout=self.timeout,
        )

    def completion(self, **kw) -> dict:
        body = {
            "model": self.model,
            "prompt": "Once upon a time",
            "max_tokens": 8,
            "temperature": 0,
            "seed": 0,
            "stream": False,
        }
        body.update(kw)
        return body

    # -- schema --------------------------------------------------------------

    def _content(
        self, path: str, method: str, status: int
    ) -> tuple[dict | None, str | None]:
        op = (self.spec.get("paths", {}).get(path) or {}).get(method.lower())
        if op is None:
            return None, f"{method} {path} is not in the contract"
        resps = op.get("responses", {})
        resp = (
            resps.get(str(status))
            or resps.get(status)
            or resps.get(f"{str(status)[0]}XX")
            or resps.get("default")
        )
        if resp is None:
            return None, f"{method} {path}: the contract declares no {status} response"
        if "$ref" in resp:
            resp = schema.resolve_ref(resp["$ref"], self.spec)
        return resp.get("content") or {}, None

    def validate(self, path: str, method: str, r: web.Response) -> list[str]:
        if r.status == 0:
            return [r.error]
        content, err = self._content(path, method, r.status)
        if err:
            return [err]
        if not content:
            return []
        ctype = r.content_type
        if ctype not in content:
            return [
                f"{method} {path} {r.status}: content type {ctype or '(none)'} not in the contract ({', '.join(content)})"
            ]
        media = content[ctype] or {}
        if ctype == "text/event-stream":
            payloads, errs = web.sse_events(r.body)
            chunk = media.get("x-tl-chunk")
            for i, p in enumerate(payloads):
                if p == "[DONE]":
                    continue
                try:
                    obj = json.loads(p)
                except json.JSONDecodeError:
                    errs.append(f"chunk {i}: not JSON: {p[:80]!r}")
                    continue
                if chunk:
                    errs += [
                        f"chunk {i}: {e}"
                        for e in schema.validate(obj, chunk, self.spec)
                    ]
            return errs
        if "schema" not in media:
            return []
        try:
            body = r.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [f"{method} {path} {r.status}: body is not JSON: {r.text[:120]!r}"]
        return [
            f"{method} {path} {r.status} {e}"
            for e in schema.validate(body, media["schema"], self.spec)
        ]


def _want(r: web.Response, status: int, what: str) -> list[str]:
    if r.status == 0:
        return [r.error]
    if r.status != status:
        return [f"{what}: HTTP {r.status}, want {status}: {r.text[:200]!r}"]
    return []


def _text(r: web.Response) -> str | None:
    try:
        return r.json()["choices"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _stream_text(r: web.Response) -> tuple[str, list[dict], list[str]]:
    payloads, errs = web.sse_events(r.body)
    chunks, out = [], []
    for p in payloads:
        if p == "[DONE]":
            continue
        try:
            obj = json.loads(p)
        except json.JSONDecodeError:
            errs.append(f"chunk not JSON: {p[:80]!r}")
            continue
        chunks.append(obj)
        try:
            out.append(obj["choices"][0].get("text") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
    return "".join(out), chunks, errs


# ---------------------------------------------------------------------------
# named checks: (client, params) -> errors


def check_healthz(c: Client, p: dict) -> list[str]:
    r = c.call("GET", p.get("path", "/healthz"), auth=False, base=c.health_base)
    return _want(r, 200, "GET /healthz")


def check_completion_schema(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        "/v1/completions",
        c.completion(prompt=p.get("prompt", "Once upon a time")),
    )
    errs = _want(r, 200, "POST /v1/completions") or c.validate(
        "/v1/completions", "post", r
    )
    if not errs and not isinstance(_text(r), str):
        errs.append("choices[0].text is missing or not a string")
    return errs


def check_greedy_stable(c: Client, p: dict) -> list[str]:
    body = c.completion(
        prompt=p.get("prompt", "The cat"),
        max_tokens=p.get("max_tokens", 16),
        temperature=0,
    )
    a, b = (
        c.call("POST", "/v1/completions", body),
        c.call("POST", "/v1/completions", body),
    )
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if not errs and _text(a) != _text(b):
        errs.append(f"temperature 0 is not stable: {_text(a)!r} then {_text(b)!r}")
    return errs


def check_length(c: Client, p: dict) -> list[str]:
    n = int(p.get("max_tokens", 4))
    r = c.call(
        "POST",
        "/v1/completions",
        c.completion(max_tokens=n, prompt=p.get("prompt", "Hello")),
    )
    errs = _want(r, 200, "POST /v1/completions")
    if errs:
        return errs
    body = r.json()
    ch = (body.get("choices") or [{}])[0]
    if ch.get("finish_reason") != "length":
        errs.append(
            f"max_tokens={n}: finish_reason {ch.get('finish_reason')!r}, want 'length'"
        )
    usage = body.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") != n:
        errs.append(
            f"max_tokens={n}: usage.completion_tokens {usage.get('completion_tokens')!r}, want {n}"
        )
    return errs


def check_stream_framing(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        "/v1/completions",
        c.completion(stream=True, max_tokens=p.get("max_tokens", 8)),
    )
    errs = _want(r, 200, "POST /v1/completions stream")
    if errs:
        return errs
    if r.content_type != "text/event-stream":
        errs.append(f"content-type {r.content_type!r}, want text/event-stream")
    errs += c.validate("/v1/completions", "post", r)
    _, chunks, _ = _stream_text(r)
    if not chunks:
        errs.append("no data chunks before [DONE]")
    elif (chunks[-1].get("choices") or [{}])[0].get("finish_reason") is None:
        errs.append("the last chunk carries no finish_reason")
    return errs


def check_stream_equals_nonstream(c: Client, p: dict) -> list[str]:
    kw = dict(
        prompt=p.get("prompt", "Once upon a time"),
        max_tokens=p.get("max_tokens", 12),
        temperature=0,
    )
    a = c.call("POST", "/v1/completions", c.completion(**kw))
    b = c.call("POST", "/v1/completions", c.completion(stream=True, **kw))
    errs = _want(a, 200, "non-stream") + _want(b, 200, "stream")
    if errs:
        return errs
    streamed, _, ferrs = _stream_text(b)
    if ferrs:
        return ferrs
    if streamed != _text(a):
        errs.append(f"stream {streamed!r} != non-stream {_text(a)!r} at temperature 0")
    return errs


def check_seed_stable(c: Client, p: dict) -> list[str]:
    body = c.completion(
        prompt=p.get("prompt", "Once"),
        max_tokens=p.get("max_tokens", 12),
        temperature=p.get("temperature", 1.0),
        seed=p.get("seed", 1234),
    )
    a, b = (
        c.call("POST", "/v1/completions", body),
        c.call("POST", "/v1/completions", body),
    )
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if not errs and _text(a) != _text(b):
        errs.append(f"same seed, different output: {_text(a)!r} then {_text(b)!r}")
    return errs


def _error_shape(
    r: web.Response, status: int, typ: str | None, code: str | None
) -> list[str]:
    errs = _want(r, status, f"want {status}")
    if errs:
        return errs
    try:
        e = r.json()["error"]
    except (ValueError, KeyError, TypeError):
        return [f'{status} body is not {{"error": {{...}}}}: {r.text[:200]!r}']
    for k in ("message", "type", "param", "code"):
        if not isinstance(e, dict) or k not in e:
            errs.append(f"{status} error object lacks {k!r}")
    if not errs and typ and e.get("type") != typ:
        errs.append(f"{status} error type {e.get('type')!r}, want {typ!r}")
    if not errs and code and e.get("code") != code:
        errs.append(f"{status} error code {e.get('code')!r}, want {code!r}")
    return errs


def check_error_400(c: Client, p: dict) -> list[str]:
    body = c.completion(**p.get("override", {"temperature": -1}))
    r = c.call("POST", "/v1/completions", body)
    return _error_shape(r, 400, "invalid_request_error", None) or c.validate(
        "/v1/completions", "post", r
    )


def check_auth_401(c: Client, p: dict) -> list[str]:
    r = c.call("POST", "/v1/completions", c.completion(), auth=False)
    return _error_shape(r, 401, "invalid_request_error", "invalid_api_key")


CHECKS = {
    "healthz": check_healthz,
    "completion_schema": check_completion_schema,
    "greedy_stable": check_greedy_stable,
    "length": check_length,
    "stream_framing": check_stream_framing,
    "stream_equals_nonstream": check_stream_equals_nonstream,
    "seed_stable": check_seed_stable,
    "error_400": check_error_400,
    "auth_401": check_auth_401,
}


@dataclass
class Result:
    case: str
    status: str  # pass fail pending
    detail: str = ""


def run(
    client: Client, cases: list[Case], suite: Suite, passed=lambda mid: True
) -> list[Result]:
    out: list[Result] = []
    for c in cases:
        if (
            suite.version not in c.versions
            or client.tier not in c.tiers
            or (suite.smoke and not c.smoke)
        ):
            continue
        waiting = [m for m in c.requires if not passed(m)]
        if c.check == "healthz" and not client.health_base:
            out.append(
                Result(
                    c.id,
                    "pending",
                    "no health endpoint to call: pass --health-base URL of the health "
                    "port (kind runs: set [deploy].gateway_health_url)",
                )
            )
            continue
        if waiting:
            out.append(Result(c.id, "pending", f"requires {', '.join(waiting)}"))
            continue
        try:
            errs = CHECKS[c.check](client, c.params)
        except (
            Exception
        ) as e:  # a malformed response must fail the case, not crash the suite
            errs = [f"{type(e).__name__}: {e}"]
        out.append(Result(c.id, "fail" if errs else "pass", "\n".join(errs)))
    return out
