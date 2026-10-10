# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for course/fixtures/data.05/labelled.jsonl.

Each line is {"text", "spans": [{"kind", "start", "end"}]}: a sentence from
a template with values put into its slots. A slot filled with PII records
its span as it is written, so the labels come from construction, never from
a detector. Every value is synthetic: example domains (RFC 2606), 555-01xx
phone numbers, documentation address blocks (RFC 5737, RFC 3849), card
numbers with random digits and a computed Luhn check digit, keys with
random characters. Lookalike lines (ISBNs, versions, dates, times, UUIDs,
commit hashes, Luhn-failing numbers, Luhn-valid numbers that no card
network issues, npm specs, slices) carry no spans: a detector that fires on
them loses precision.

    uv run --script course/oracle/data.05/make_labelled.py

Run from the repo root; prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from pathlib import Path

OUT = Path("course/fixtures/data.05/labelled.jsonl")
FIRST = [
    "ana",
    "bo",
    "chen",
    "dara",
    "eli",
    "fatima",
    "goran",
    "hiro",
    "ines",
    "jamal",
    "kai",
    "lena",
]
LAST = ["lopez", "smith", "okafor", "nguyen", "rossi", "kim", "haddad", "novak"]
DOMAINS = [
    "example.com",
    "example.org",
    "mail.example.net",
    "corp-example.co.uk",
    "example.io",
]


def luhn_digit(body: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def card(rng: random.Random) -> str:
    net = rng.choice(["visa16", "visa13", "mc", "mc2", "amex", "disc", "visa19"])
    prefix, n = {
        "visa16": ("4", 16),
        "visa13": ("4", 13),
        "visa19": ("4", 19),
        "mc": (str(rng.randint(51, 55)), 16),
        "mc2": (str(rng.randint(2221, 2720)), 16),
        "amex": (rng.choice(["34", "37"]), 15),
        "disc": ("6011", 16),
    }[net]
    body = prefix + "".join(
        rng.choice(string.digits) for _ in range(n - len(prefix) - 1)
    )
    num = body + luhn_digit(body)
    style = rng.choice(["plain", "space", "dash"])
    if style == "plain":
        return num
    sep = " " if style == "space" else "-"
    if net == "amex":
        return sep.join([num[:4], num[4:10], num[10:]])
    return sep.join(num[i : i + 4] for i in range(0, len(num), 4))


def email(rng: random.Random) -> str:
    local = rng.choice(FIRST) + rng.choice([".", "_", "", "-"]) + rng.choice(LAST)
    if rng.random() < 0.3:
        local += "+" + rng.choice(["news", "shop", "work"])
    if rng.random() < 0.2:
        local = local.capitalize()
    if rng.random() < 0.2:
        local += str(rng.randint(1, 99))
    return local + "@" + rng.choice(DOMAINS)


def phone(rng: random.Random) -> str:
    a, x = rng.randint(201, 989), f"01{rng.randint(0, 99):02d}"
    return rng.choice(
        [
            f"({a}) 555-{x}",
            f"{a}-555-{x}",
            f"{a}.555.{x}",
            f"+1 {a} 555 {x}",
            f"+44 20 7946 0{rng.randint(100, 999)}",
            f"+4420794{rng.randint(10000, 99999)}",
            f"+49 30 {rng.randint(1000000, 9999999)}",
        ]
    )


def ip(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return rng.choice(["192.0.2.", "198.51.100.", "203.0.113."]) + str(
            rng.randint(1, 254)
        )
    if rng.random() < 0.5:
        return "2001:db8::" + ":".join(
            f"{rng.randint(1, 0xFFFF):x}" for _ in range(rng.randint(1, 3))
        )
    return "2001:db8:" + ":".join(f"{rng.randint(0, 0xFFFF):x}" for _ in range(6))


def key(rng: random.Random) -> str:
    def rand(alpha: str, n: int) -> str:
        return "".join(rng.choice(alpha) for _ in range(n))

    an = string.ascii_letters + string.digits
    return rng.choice(
        [
            "sk-" + rand(an, 32),
            "sk-proj-" + rand(an + "_-", 40),
            "AKIA" + rand(string.ascii_uppercase + string.digits, 16),
            "ghp_" + rand(an, 36),
            # Letters only after a literal EXAMPLE tag: still matches the scrubber,
            # but not a real Slack token shape, so secret scanners do not flag it.
            "xoxb-EXAMPLE-" + rand(string.ascii_letters, 24),
            "AIza" + rand(an + "_-", 35),
            "tl_"
            + rand(string.ascii_lowercase + string.digits, 8)
            + "_"
            + rand(an, 32),
        ]
    )


TEMPLATES = [
    "Please write to {email} before Friday.",
    "Call me at {phone} or email {email}.",
    "Payment card {card} was declined.",
    "The server at {ip} stopped answering at noon.",
    "Rotate the key {key} today.",
    "Contact: {email}; backup line {phone}.",
    "Charge {card} and send the receipt to {email}.",
    "Login from {ip} used key {key}.",
    "{email} reported that {ip} is down.",
    "My number changed to {phone}.",
]

LOOKALIKES = [
    "ISBN 978-0-306-40615-7 is on the shelf.",
    "ISBN 0-306-40615-2 has a new edition.",
    "Upgrade to version 1.2.3 or 10.4.11 first.",
    "Release v1.2.3.4 shipped, then 1.2.3.4.5 by mistake.",
    "The meeting is on 2024-05-20 at 12:30:45.",
    "Request id 123e4567-e89b-12d3-a456-426614174000 failed.",
    "Commit 9fceb02d0ae598e95dc970b74767f19372d61af8 fixed it.",
    "Total: $1,234.56 for 3 items.",
    "Order 4111111111111112 is late.",
    "Timestamp 1696867200123 in milliseconds.",
    "Tracking 7992739871300 is valid but not a card.",
    "Run npm install react@18.3.1 now.",
    "In Python a[1::2] takes every other item; ::1 is loopback.",
    "Use std::vector and Foo::Bar in C++.",
    "Mail root@localhost on the box.",
    "Dial 555-0123 from the lobby.",
    "The ratio was 3:2:1 overall.",
    "Coordinates 51.5074, -0.1278 are central.",
    "Symbols tl_matmul_f32 and tl_kv_pool_create are in the header.",
    "The sklearn module and sk-learn alias differ.",
    "Call extension 212 or 555 directly.",
    "We sold 4,111 1111 units.",
]


def main() -> None:
    rng = random.Random(5)
    makers = {"email": email, "phone": phone, "card": card, "ip": ip, "key": key}
    rows = []
    for i in range(330):
        tpl = TEMPLATES[i % len(TEMPLATES)]
        text, spans, pos = "", [], 0
        for part in string.Formatter().parse(tpl):
            lit, field = part[0], part[1]
            text += lit
            if field:
                val = makers[field](rng)
                spans.append(
                    {"kind": field, "start": len(text), "end": len(text) + len(val)}
                )
                text += val
        rows.append({"text": text, "spans": spans})
    for lk in LOOKALIKES:
        for _ in range(3):
            rows.append({"text": lk, "spans": []})
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    b = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/data.05/make_labelled.py\t-\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
