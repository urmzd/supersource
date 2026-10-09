# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Maintainer generator for course/fixtures/small-corpora/digits.npz (MS-L0).

Source: the UCI "Optical Recognition of Handwritten Digits" dataset (E. Alpaydin
and C. Kaynak, 1998; UCI dataset 80, license CC BY 4.0). Each row is an 8x8 grid
of counts in 0..16 (on pixels per 4x4 block of a 32x32 bitmap) and a class
0..9. The official split is kept: optdigits.tra (3823 rows, 30 writers) for
training and optdigits.tes (1797 rows, 13 other writers) for testing, so no
writer appears in both.

    uv run --script course/oracle/MS-L0/digits.py [path/to/optdigits.zip]

Without a path it downloads the archive from the UCI repository. Either way
the archive must match the pinned sha256. Run from the repo root. It writes the
.npz with fixed zip timestamps (byte-identical on every run) and prints the
MANIFEST.tsv row.

The .npz holds x_train uint8 [3823, 64], y_train uint8 [3823], x_test uint8
[1797, 64], y_test uint8 [1797], features in row-major pixel order.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
ZIP_SHA256 = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
MEMBERS = {
    "optdigits.tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    "optdigits.tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
}
OUT = Path("course/fixtures/small-corpora/digits.npz")
EPOCH = (1980, 1, 1, 0, 0, 0)


def parse(text: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        list(map(int, line.split(","))) for line in text.splitlines() if line.strip()
    ]
    a = np.array(rows, dtype=np.int64)
    if a.shape[1] != 65 or a[:, :64].min() < 0 or a[:, :64].max() > 16:
        raise SystemExit("unexpected optdigits layout")
    if sorted(set(a[:, 64].tolist())) != list(range(10)):
        raise SystemExit("unexpected class labels")
    return a[:, :64].astype(np.uint8), a[:, 64].astype(np.uint8)


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(
                member, np.ascontiguousarray(arr), allow_pickle=False
            )
            info = zipfile.ZipInfo(name + ".npy", date_time=EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, member.getvalue(), compresslevel=9)
    return buf.getvalue()


def main() -> None:
    raw = (
        Path(sys.argv[1]).read_bytes()
        if len(sys.argv) > 1
        else urllib.request.urlopen(URL).read()
    )
    if hashlib.sha256(raw).hexdigest() != ZIP_SHA256:
        raise SystemExit("optdigits.zip does not match the pinned sha256")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        texts = {}
        for name, digest in MEMBERS.items():
            b = z.read(name)
            if hashlib.sha256(b).hexdigest() != digest:
                raise SystemExit(f"{name} does not match the pinned sha256")
            texts[name] = b.decode("ascii")
    xtr, ytr = parse(texts["optdigits.tra"])
    xte, yte = parse(texts["optdigits.tes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = npz_bytes({"x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte})
    OUT.write_bytes(blob)
    with np.load(OUT) as z:
        assert z["x_train"].shape == (3823, 64) and z["x_test"].shape == (1797, 64)
    print(f"train {xtr.shape[0]} rows, test {xte.shape[0]} rows, classes 0..9")
    print(
        "\t".join(
            [
                OUT.as_posix(),
                hashlib.sha256(blob).hexdigest(),
                str(len(blob)),
                "course/oracle/MS-L0/digits.py",
                "numpy==2.2.6",
                f"uci:80/optdigits.zip@sha256:{ZIP_SHA256}",
                "CC-BY-4.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
