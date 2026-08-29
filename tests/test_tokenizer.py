from __future__ import annotations

from pathlib import Path

from apex.tokenizer import ByteBPETokenizer


def test_byte_bpe_round_trip_and_serialization(tmp_path: Path) -> None:
    tokenizer = ByteBPETokenizer()
    corpus = ["APEX builds from first principles.", "नमस्ते APEX", "code: value = 42"]
    tokenizer.train(corpus, vocab_size=280)
    text = "नमस्ते APEX — value = 42"
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == 1 and encoded[-1] == 2
    assert tokenizer.decode(encoded) == text
    assert tokenizer.vocab_size == 280

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = ByteBPETokenizer.load(path)
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_chat_template_validates_roles() -> None:
    tokenizer = ByteBPETokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Hello"}]
    )
    assert "<system>" in rendered and rendered.endswith("<assistant>\n")
