from typing import Any

class LinearEXL3:
    key: str
    in_features: int
    out_features: int
    trellis: Any
    def forward(self, x: Any, params: dict[str, Any] | None = None) -> Any: ...
