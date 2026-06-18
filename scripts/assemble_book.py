#!/usr/bin/env python3
"""Assemble the whole supersource curriculum into one pandoc-ready markdown file.

This is the preprocessor behind ``scripts/build-book.sh``. It does three things
that a naive ``cat`` of every README cannot:

1. Orders tracks and topics into a coherent learning sequence (a track's
   README first, then its numbered topic READMEs in order).
2. Rewrites *relative* image paths (e.g. ``../diagrams/c4-context.svg``) to
   absolute paths, so links survive being concatenated into one document in a
   different directory.
3. Injects LaTeX ``\\part{...}`` dividers so the PDF has a real
   Part / Chapter hierarchy and a clean table of contents.

Mermaid fenced code blocks are left untouched -- pandoc's ``mermaid-filter``
renders them downstream (or they fall back to code blocks).

Usage:
    python scripts/assemble_book.py --out /tmp/book.md [--root .]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ordered learning sequence: (directory, Part title). Within each directory we
# include README.md, then every <subdir>/README.md sorted by name (so the
# 01-, 02-, ... topic prefixes order naturally).
TRACKS: list[tuple[str, str]] = [
    ("math", "Mathematics Foundations"),
    ("algorithms", "Algorithm Mastery"),
    ("information-theory", "Information Theory"),
    ("ml", "Machine Learning & AI"),
    ("systems", "Systems & Architecture"),
    ("data-engineering", "Data Engineering"),
    ("ai-platform-engineering", "AI Platform Engineering"),
    ("programming-languages", "Programming Languages"),
    ("software-craftsmanship", "Software Craftsmanship"),
    ("diagramming-and-documentation", "Diagramming & Documentation"),
    ("infrastructure", "Infrastructure"),
    ("competitive-programming", "Competitive Programming"),
]

# Appended after the tracks as a final Part.
APPENDICES: list[tuple[str, str]] = [
    ("STUDY-PLAN.md", "Appendix: Study Plans"),
]

# Markdown inline image:  ![alt](target "title")
IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)([^)]*\))")

# Emoji that LaTeX text fonts (DejaVu Serif) can't render. The check/cross
# code points aren't in DejaVu Serif either, so map to plain words -- legible
# and unambiguous in the PDF's comparison tables.
GLYPH_SUBS = {"✅": "Yes", "❌": "No"}


def substitute_glyphs(text: str) -> str:
    for src, dst in GLYPH_SUBS.items():
        text = text.replace(src, dst)
    return text


def single_chapter(text: str) -> str:
    """Demote every H1 after the first to H2 so each README is ONE chapter.

    House-style READMEs use two H1s (the title, then "# Concepts & Techniques")
    plus extra H1s like "# Part 1". Left alone, each becomes its own PDF chapter,
    producing a redundant "Concepts & Techniques" chapter per topic. Keeping only
    the first H1 as the chapter and demoting the rest to sections fixes the
    ordering. Code-fence aware so `#` comments inside code blocks are untouched.
    """
    out: list[str] = []
    in_fence = False
    fence = ""
    seen_h1 = False
    for line in text.split("\n"):
        marker = line.lstrip()[:3]
        if in_fence:
            if marker == fence:
                in_fence = False
            out.append(line)
            continue
        if marker in ("```", "~~~"):
            in_fence = True
            fence = marker
            out.append(line)
            continue
        if line.startswith("# "):
            if seen_h1:
                out.append("#" + line)  # H1 -> H2
            else:
                seen_h1 = True
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def is_remote_or_absolute(target: str) -> bool:
    return (
        target.startswith(("http://", "https://", "/"))
        or target.startswith("data:")
        or target.startswith("#")
    )


def rewrite_images(text: str, base_dir: Path) -> str:
    """Rewrite relative image targets to absolute paths anchored at base_dir."""

    def repl(m: re.Match[str]) -> str:
        prefix, target, suffix = m.group(1), m.group(2), m.group(3)
        if is_remote_or_absolute(target):
            return m.group(0)
        resolved = (base_dir / target).resolve()
        return f"{prefix}{resolved}{suffix}"

    return IMAGE_RE.sub(repl, text)


def latex_escape(title: str) -> str:
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        title = title.replace(char, repl)
    return title


def part_divider(title: str) -> str:
    return f"\n```{{=latex}}\n\\part{{{latex_escape(title)}}}\n```\n\n"


def collect_files(root: Path, directory: str) -> list[Path]:
    base = root / directory
    if base.is_file():  # an appendix like STUDY-PLAN.md
        return [base]
    files: list[Path] = []
    readme = base / "README.md"
    if readme.exists():
        files.append(readme)
    files.extend(sorted(base.glob("*/README.md")))
    return files


def assemble(root: Path) -> tuple[str, int]:
    chunks: list[str] = []
    chapters = 0
    for directory, title in [*TRACKS, *APPENDICES]:
        files = collect_files(root, directory)
        if not files:
            print(f"[warn] no content for '{directory}', skipping", file=sys.stderr)
            continue
        chunks.append(part_divider(title))
        for path in files:
            text = path.read_text(encoding="utf-8")
            text = single_chapter(text)
            text = substitute_glyphs(rewrite_images(text, path.parent))
            chunks.append(text)
            chunks.append("\n\n")
            chapters += 1
    return "".join(chunks), chapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, type=Path, help="combined markdown output path"
    )
    parser.add_argument(
        "--root", default=Path(__file__).resolve().parent.parent, type=Path
    )
    args = parser.parse_args()

    body, chapters = assemble(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"[info] assembled {chapters} chapter(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
