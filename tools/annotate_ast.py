#!/usr/bin/env python3
"""Add missing parameter type annotations using name heuristics."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ponyexl3"

PARAM_TYPES: dict[str, str] = {
    "model": "MlxLmModel",
    "tokenizer": "Tokenizer",
    "mtp": "DraftModule",
    "drafter": "DraftModule",
    "eagle3": "DraftModule",
    "dflash": "DraftModule",
    "layer": "EXL3Layer",
    "x": "mx.array",
    "xh": "mx.array",
    "w": "mx.array",
    "y": "mx.array",
    "z": "mx.array",
    "cache": "KvCache",
    "weights": "dict[str, mx.array]",
    "mod": "nn.Module",
    "module": "nn.Module",
    "lm_head": "Any",
    "input_ids": "np.ndarray",
    "input_ids_t": "Any",
    "params": "dict[str, Any]",
    "bucket": "dict[str, Any]",
    "hooks": "dict[str, Any]",
    "a": "mx.array",
    "k": "int",
    "cb": "CodebookMode",
    "blk": "Any",
    "i": "int",
    "fin": "int",
    "back": "np.ndarray",
    "s": "int",
    "catch_t": "mx.array",
    "catch_h": "mx.array",
    "catch_g": "mx.array",
    "mask": "mx.array | None",
    "self": "Any",
}

TYPE_IMPORTS = {
    "MlxLmModel": "from ponyexl3.types import MlxLmModel",
    "ExLlamaModel": "from ponyexl3.types import ExLlamaModel",
    "DraftModule": "from ponyexl3.types import DraftModule",
    "Tokenizer": "from ponyexl3.types import Tokenizer",
    "KvCache": "from ponyexl3.types import KvCache",
    "JsonDict": "from ponyexl3.types import JsonDict",
    "EXL3Layer": "from ponyexl3.ref.layer import EXL3Layer",
    "CodebookMode": "from ponyexl3.ref.codebook import CodebookMode",
    "mx.array": "import mlx.core as mx",
    "nn.Module": "import mlx.nn as nn",
    "np.ndarray": "import numpy as np",
    "Any": "from typing import Any",
}


class Annotator(ast.NodeTransformer):
    def __init__(self) -> None:
        self.needed: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is not None:
                continue
            ann = PARAM_TYPES.get(arg.arg)
            if ann is None:
                ann = "Any"
                self.needed.add("Any")
            arg.annotation = ast.parse(ann, mode="eval").body
            self.needed.add(ann)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def _ensure_imports(source: str, needed: set[str]) -> str:
    lines = source.splitlines(keepends=True)
    insert = 0
    for i, line in enumerate(lines):
        if line.startswith("from __future__"):
            insert = i + 1
    add: list[str] = []
    text = source
    type_syms: list[str] = []
    for ann in sorted(needed):
        if ann in ("mx.array", "nn.Module", "np.ndarray", "Any"):
            imp = TYPE_IMPORTS[ann]
            if imp not in text:
                add.append(imp + "\n")
        elif ann.startswith("dict["):
            if "from typing import Any" not in text and "import Any" not in text:
                add.append("from typing import Any\n")
        elif ann.endswith("| None") or "|" in ann:
            base = ann.split("|")[0].strip()
            if base in TYPE_IMPORTS and TYPE_IMPORTS[base] not in text:
                type_syms.append(base)
        elif ann in TYPE_IMPORTS:
            type_syms.append(ann)
    pony_types = [s for s in sorted(set(type_syms)) if s in ("MlxLmModel", "ExLlamaModel", "DraftModule", "Tokenizer", "KvCache", "JsonDict")]
    if pony_types:
        imp = f"from ponyexl3.types import {', '.join(pony_types)}\n"
        if imp not in text:
            add.append(imp)
    for s in sorted(set(type_syms) - set(pony_types)):
        imp = TYPE_IMPORTS.get(s)
        if imp and imp not in text:
            add.append(imp + "\n")
    if not add:
        return source
    return "".join(lines[:insert] + ["\n"] + add + lines[insert:])


def process(path: Path) -> bool:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    ann = Annotator()
    new_tree = ann.visit(tree)
    if not ann.needed:
        return False
    ast.fix_missing_locations(new_tree)
    new_src = ast.unparse(new_tree)
    new_src = _ensure_imports(new_src, ann.needed)
    if new_src != src:
        path.write_text(new_src)
        return True
    return False


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else list(PKG.rglob("*.py"))
    for path in sorted(paths):
        if process(path):
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
