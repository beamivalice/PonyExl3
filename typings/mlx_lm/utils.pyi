from pathlib import Path
from typing import Any

class TokenizerWrapper:
    eos_token_id: int | None
    eos_token_ids: list[int] | None
    chat_template: str | None
    detokenizer: Any
    def encode(self, text: str) -> list[int]: ...
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]: ...

def load_tokenizer(
    model_path: str | Path,
    tokenizer_config_extra: dict[str, Any] | None = None,
    eos_token_ids: list[int] | None = None,
) -> TokenizerWrapper: ...
def load(path: str | Path) -> tuple[Any, dict[str, Any]]: ...
