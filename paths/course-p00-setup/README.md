# Course Pass 0: Setup

Before any model, a repo that grades itself. You install the toolchain, learn the Python and numpy you need to count bytes fast and hand arrays to C, learn the shell, git, and make that every later build rests on, and set up your repo's CI so every push lints its commit message, runs your native tests, and runs `ss check --all --ci`.

**Part of**: [the course](../course/). About 1.5 weeks at 10 to 12 hours a week.

**Gate**: [MS-P0](milestone.md), your first conventional commit green in CI.

## Toolchain (stage 1)

```bash
practice/bin/ss doctor              # what pass 0 needs: python, uv, cc, git, make
practice/bin/ss doctor --pass 1     # what pass 1 adds: cargo, go, docker, kubectl, kind, helm
```

On macOS: `xcode-select --install`, then `brew install uv rustup go kubectl kind helm` and Docker Desktop (`tilt` joins in Pass 7). On Linux use your package manager and the upstream installers. Then create your repo:

```bash
practice/bin/ss course init --name <system>   # a git repo with contracts/ vendored and system.toml
```

The chapters write plain `ss`. Put supersource's `practice/bin` on your `PATH` once, so `ss` works from any directory:

```bash
export PATH="$PWD/practice/bin:$PATH"   # run in your supersource clone; add the line to ~/.zshrc or ~/.bashrc with the full path
which ss                                # prints .../practice/bin/ss
```

The stages, their chapters, and what "done" means for each are in [`path.tsv`](path.tsv).

```bash
practice/bin/ss learn course-p00-setup         # stages and progress
practice/bin/ss learn course-p00-setup next    # read the next unfinished stage
```
