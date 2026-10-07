# Role Paths

A role path is a reading order for one job, cut across the tracks. The tracks stay where they are and own their content; a path only decides **which chapters, in what order, and what "done" means** for that role. Nothing is copied, so a chapter fixed in its track is fixed in every path that uses it.

| Path | For | Stages |
|------|-----|--------|
| [fde-inference](fde-inference/) | Forward Deployed Engineer at an inference and fine-tuning cloud (Fireworks, Together AI, Baseten style) | 11 |

## Reading a path

```bash
practice/bin/ss learn                              # every path, and your progress
practice/bin/ss learn fde-inference                # stages, files, and done-when criteria
practice/bin/ss learn fde-inference next           # read the first unfinished stage
practice/bin/ss learn fde-inference 8              # read stage 8
practice/bin/ss learn fde-inference --done 8       # mark it (progress lives in .scratchpad/, never committed)
```

Stages render with [glow](https://github.com/charmbracelet/glow) when it is installed, and as plain markdown through `$PAGER` otherwise.

Each path also builds into its own PDF (`supersource-<path>.pdf`): CI attaches it to every build artifact and GitHub Release, and `./scripts/build-book.sh --path <path>` builds it locally.

## Adding a path

1. Create `paths/<name>/README.md`. Its first `# ` heading is the path's title.
2. Create `paths/<name>/path.tsv`, one row per stage, tab-separated:

   ```text
   stage <TAB> title <TAB> file[,file...] <TAB> done when
   ```

   Files are repo-relative markdown, read in the order listed. Lines starting with `#` are comments. "Done when" is a checkable outcome (you built, measured, or explained something), not "read the chapter".
3. Put role-only material (things no track should own) next to the manifest, as `field-craft.md` is for the FDE path. Anything a second role could use belongs in a track instead.
4. Run `practice/bin/ss learn --verify`. CI runs it too, so a renamed chapter fails the build instead of breaking the path.
5. Add the PDF to `sr.yaml` under `packages[].artifacts` so releases attach it.
