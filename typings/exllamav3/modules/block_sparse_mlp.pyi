from collections.abc import Callable
from typing import Any

class BlockSparseMLP:
    key: str
    routing_fn: Callable[..., Any] | None
    def forward(self, x: Any, params: dict[str, Any] | None = None) -> Any: ...
