"""Fixture CLI for the `tinyllm` role (spec/cli-roles.md): generate prints the
generated ids as a final JSON line; logits prints next-token logits under
teacher forcing on --prefix-ids."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "serve"))
import bigram  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="tinyllm")
    ap.add_argument("verb", choices=["generate", "logits"])
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix-ids", default="")
    a = ap.parse_args()
    if a.verb == "generate":
        ids = bigram.generate(a.prompt, a.max_tokens, 0.0 if a.greedy else 1.0, a.seed)
        print(f"generating {a.max_tokens} tokens")
        print(json.dumps({"ids": ids, "text": bytes(ids).decode()}))
    else:
        prefix = [int(x) for x in a.prefix_ids.split(",") if x]
        print(json.dumps({"logits": bigram.logits(a.prompt, prefix)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
