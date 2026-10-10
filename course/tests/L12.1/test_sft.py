import numpy as np
from tinyllm.post.sft import assistant_loss_mask, pack


class Tokenizer:
    def render_chat(self, messages):
        return [1, 2, 3, 4]
    def assistant_spans(self, messages):
        return [(2, 4)]


def test_hand_rendered_chat():
    ids, mask = assistant_loss_mask([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}], Tokenizer())
    assert ids == [1, 2, 3, 4]
    assert mask == [False, False, True, True]


def test_assistant_only_mask():
    _, mask = assistant_loss_mask([], Tokenizer())
    assert sum(mask) == 2


def test_packed_documents_do_not_attend_across_boundaries():
    ids, mask, docs = pack([([11, 12], [False, True]), ([21], [True])], 4)
    assert ids.tolist() == [11, 12, 21, 0]
    assert mask.tolist() == [False, True, True, False]
    assert docs.tolist() == [0, 0, 1, -1]
