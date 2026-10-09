"""The 300 test strings shared by the tokenizer goldens (L1.2, L1.3, L1.4, L1.6).

60 hand-picked strings hit one rule each (contractions, white-space runs,
Unicode digits, CJK, emoji sequences, combining marks, control characters,
special-token texts, an over-long word); 240 more are seeded compositions
of fragments. `strings()` is deterministic: random.Random(20261009).
"""

from __future__ import annotations

import random

HAND = [
    "",
    " ",
    "Hello world",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "don't stop: it's what we'll do, they're sure, I've seen, I'm in, he'd go",
    "DON'T SHOUT: IT'S LOUD, WE'LL SEE",
    "she said ’twas fine and it’s ok",
    "'s 't 're 've 'm 'll 'd",
    "rock 'n' roll",
    "  leading spaces",
    "trailing spaces   ",
    "two  spaces   three    four",
    "tab\tseparated\tvalues",
    "line one\nline two\n\nline four\r\nline five",
    " \n \n ",
    "\n\n\n",
    "non breaking space and ideographic　space",
    "line separator and paragraph separator",
    "zero​width space",
    "12345 67890",
    "3.14159 2.71828 1,000,000 1e-9 0x1F",
    "2024-03-15 06:30:00",
    "fractions ½ ¾ and superscripts x² y³",
    "Arabic-Indic digits ٣٤٥ and roman Ⅻ",
    "$19.99 + 5% = ?",
    "café naïve résumé façade",
    "café decomposed",
    "Straße über Größe",
    "İstanbul İIıi",
    "ΣΑΣ σας",
    "日本語のテキスト",
    "中文文本测试",
    "한국어 텍스트",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "สวัสดีครับ",
    "emoji \U0001f642\U0001f680\U0001f30a",
    "family \U0001f468‍\U0001f469‍\U0001f467 and skin \U0001f44d\U0001f3fd",
    "flags \U0001f1e8\U0001f1e6\U0001f1ef\U0001f1f5 keycap 1️⃣",
    "def f(x):\n    return x ** 2  # square\n",
    "if (a != b && c <= d) { return a->b[0]; }",
    "https://example.com/path?q=1&r=two#frag",
    "user.name+tag@example.org",
    "!!! ??? ... --- *** ### @@@",
    "...and then?!",
    "\"quoted\" and (parenthesized) [bracketed] {braced}",
    "<|endoftext|>",
    "before<|endoftext|>after",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|> partial <|im",
    "a [MASK] b [CLS] c [SEP] d [mask]",
    "the <unk> token and <s> </s>",
    "control \x00 \x04 \x1c \x1f \x7f \x85 chars",
    "replacement � char",
    "Pneumonoultramicroscopicsilicovolcanoconiosis" * 3,
    "\U0001d54f\U0001d556\U0001d55d\U0001d55d\U0001d560 math letters",
    "ABCÅÅÅ",
    "Unicode ⅠⅡⅢ ①② 〇",
]

FRAGMENTS = [
    "the", " the", "The", " and", " of", " to", " a", " in", " is", " was",
    "count", "counted", "counting", " lighthouse", " robot", " boats", " Pip", " Mia",
    "'s", "'t", "'ll", "'re", "'ve", "'d", "'m", "’s", "'S",
    " ", "  ", "   ", "\t", "\n", "\n\n", " \n", "\r\n", " ", "　",
    "0", "1", "42", " 2024", "3.14", "1,000", "½", "²", "٣", "Ⅻ",
    ".", ",", "!", "?", "...", "!!", "-", "--", "(", ")", "\"", "'", ":", ";", "#", "@", "$", "%",
    "café", "é", "über", "naïve", "İ", "Σ",
    "日本", "中文", "한국", "مر", "नमस्ते",
    "\U0001f642", "\U0001f468‍\U0001f469", "\U0001f44d\U0001f3fd", "\U0001f1e8\U0001f1e6",
    "<|endoftext|>", "<|im_start|>", "[MASK]", "<unk>",
    "x", "y", "z", "ab", "abc", "Hello", " world", "WORLD", "snake_case", "camelCase",
]


def strings() -> list[str]:
    rng = random.Random(20261009)
    out = list(HAND)
    seen = set(out)
    while len(out) < 300:
        s = "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 8)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
