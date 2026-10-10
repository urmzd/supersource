from tinyllm.post.sft import assistant_loss_mask, pack


class Tokenizer:
    def render_chat(self, messages):
        return [1, 2, 3, 4]

    def assistant_spans(self, messages):
        return [(2, 4)]


def test_hand_rendered_chat():
    # WHY: user tokens provide context but only the assistant answer belongs in the SFT loss.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L12.1 section 3, Worked example by hand
    ids, mask = assistant_loss_mask(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        Tokenizer(),
    )
    assert ids == [1, 2, 3, 4]
    assert mask == [False, False, True, True]


def test_assistant_only_mask():
    # WHY: the mask length and true count must match exactly the assistant token span.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: L12.1 section 4, The interface
    _, mask = assistant_loss_mask([], Tokenizer())
    assert sum(mask) == 2


def test_packed_documents_do_not_attend_across_boundaries():
    # WHY: each packed token needs its own document id so attention can stop at boundaries.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: L12.1 section 3, Worked example by hand
    ids, mask, docs = pack([([11, 12], [False, True]), ([21], [True])], 4)
    assert ids.tolist() == [11, 12, 21, 0]
    assert mask.tolist() == [False, True, True, False]
    assert docs.tolist() == [0, 0, 1, -1]
