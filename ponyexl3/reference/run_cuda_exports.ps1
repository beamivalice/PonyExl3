# Generate the standard CUDA reference bundle for MLX parity work (Windows).
#
# Usage (from repo root):
#   .\ponyexl3\reference\run_cuda_exports.ps1 `
#     "C:\path\to\Qwen3.6-35B-A3B-exl3-4.00bpw" `
#     ".\ponyexl3\reference" `
#     "qwen3.6-35B-A3B-exl3-4.00bpw_win"
#
# Requires: Python 3.10+, exllamav3 (set EXLLAMAV3_ROOT), torch+cuda, numpy

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ModelDir,

    [Parameter(Position = 1)]
    [string]$OutDir = ".\ponyexl3\reference",

    [Parameter(Position = 2)]
    [string]$Stem = "reference"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$Exl3Root = if ($env:EXLLAMAV3_ROOT) { $env:EXLLAMAV3_ROOT } else { "C:\path\to\exllamav3" }
$env:PYTHONPATH = "$Exl3Root;$RepoRoot"
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Logits = Join-Path $OutDir "$Stem.npz"
$Trace = Join-Path $OutDir "${Stem}_trace.npz"
$Moe = Join-Path $OutDir "${Stem}_moe.npz"
$LinearIo = Join-Path $OutDir "${Stem}_linear_io.npz"

Write-Host "==> exllamav3: $Exl3Root"
Write-Host "==> pony repo: $RepoRoot"
Write-Host "==> model:     $ModelDir"
Write-Host ""

Write-Host "==> module inventory"
& $Py (Join-Path $RepoRoot "exl3\reference\list_modules.py") $ModelDir --exl3-only

Write-Host "==> logits (end-to-end)"
& $Py (Join-Path $RepoRoot "exl3\reference\export_reference.py") `
    -m $ModelDir -o $Logits -s 512 -r 1 --seed 0

Write-Host "==> forward trace (last token, all blocks)"
& $Py (Join-Path $RepoRoot "exl3\reference\export_trace.py") `
    -m $ModelDir -o $Trace --from-npz $Logits --row-slice last:1

Write-Host "==> MoE routing (last token)"
& $Py (Join-Path $RepoRoot "exl3\reference\export_moe_routing.py") `
    -m $ModelDir -o $Moe --from-npz $Logits --row-slice last:1

Write-Host "==> sample EXL3 linear I/O"
$Modules = @(
    "model.language_model.layers.0.mlp.shared_expert.gate_proj"
    "model.language_model.layers.0.mlp.experts.0.gate_proj"
    "model.language_model.layers.3.self_attn.q_proj"
    "model.language_model.layers.3.linear_attn.in_proj_qkv"
    "lm_head"
)
& $Py (Join-Path $RepoRoot "exl3\reference\export_linear_io.py") `
    -m $ModelDir -o $LinearIo `
    --from-npz $Logits --row-slice last:1 --reconstruct `
    @Modules

Write-Host ""
Write-Host "Done. Copy to Mac:"
Write-Host "  $Logits"
Write-Host "  $Trace"
Write-Host "  $Moe"
Write-Host "  $LinearIo"
