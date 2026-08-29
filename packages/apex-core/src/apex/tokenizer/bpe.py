from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class ByteBPETokenizer:
    """Small, dependency-free byte BPE suitable for pipeline validation."""

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, merges: list[tuple[bytes, bytes]] | None = None) -> None:
        self.merges = merges or []
        self.token_to_id: dict[bytes, int] = {bytes([value]): value + 4 for value in range(256)}
        for left, right in self.merges:
            merged = left + right
            if merged not in self.token_to_id:
                self.token_to_id[merged] = len(self.token_to_id) + 4
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id) + len(self.SPECIAL_TOKENS)

    def train(self, texts: list[str], vocab_size: int) -> None:
        if vocab_size < 260:
            raise ValueError("byte BPE requires vocab_size >= 260")
        sequences = [[bytes([value]) for value in text.encode("utf-8")] for text in texts]
        self.merges = []
        while self.vocab_size < vocab_size:
            counts: Counter[tuple[bytes, bytes]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:], strict=False))
            if not counts:
                break
            pair, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
            merged = pair[0] + pair[1]
            if merged not in self.token_to_id:
                self.token_to_id[merged] = len(self.token_to_id) + 4
                self.id_to_token[self.token_to_id[merged]] = merged
                self.merges.append(pair)
            sequences = [self._merge_pair(sequence, pair) for sequence in sequences]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        sequence = [bytes([value]) for value in text.encode("utf-8")]
        for pair in self.merges:
            sequence = self._merge_pair(sequence, pair)
        ids = [self.token_to_id[token] for token in sequence]
        if add_bos:
            ids.insert(0, 1)
        if add_eos:
            ids.append(2)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        payload = b"".join(
            self.id_to_token[token_id]
            for token_id in token_ids
            if token_id >= len(self.SPECIAL_TOKENS) and token_id in self.id_to_token
        )
        return payload.decode("utf-8", errors="replace")

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        allowed_roles = {"system", "developer", "user", "assistant", "tool"}
        rendered: list[str] = []
        for message in messages:
            role = message["role"]
            if role not in allowed_roles:
                raise ValueError(f"unsupported role: {role}")
            rendered.append(f"<{role}>\n{message['content']}\n</{role}>")
        if add_generation_prompt:
            rendered.append("<assistant>\n")
        return "\n".join(rendered)

    def save(self, path: str | Path) -> None:
        data = {"version": 1, "merges": [[left.hex(), right.hex()] for left, right in self.merges]}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("unsupported tokenizer version")
        return cls([(bytes.fromhex(left), bytes.fromhex(right)) for left, right in data["merges"]])

    @staticmethod
    def _merge_pair(sequence: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        merged: list[bytes] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(pair[0] + pair[1])
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged
