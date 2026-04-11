"""Tiny subset of whisper.tokenizer needed for CosyVoice imports."""
from dataclasses import dataclass


@dataclass
class Tokenizer:
    encoding: object
    num_languages: int | None = None
    language: str | None = None
    task: str | None = None

    def encode(self, text: str, **kwargs):
        allowed_special = kwargs.get("allowed_special", "all")
        return self.encoding.encode(text, allowed_special=allowed_special)

    def decode(self, tokens):
        return self.encoding.decode(tokens)