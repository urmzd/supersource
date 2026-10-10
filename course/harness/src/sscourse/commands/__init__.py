"""One module per `ss` verb. `sscourse.cli` imports `commands.<verb>` lazily
and calls its `main(argv) -> int`, so a verb is added by adding a file."""
