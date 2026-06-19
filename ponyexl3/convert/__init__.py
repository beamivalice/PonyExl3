"""PonyExl3: HF → EXL3 conversion on Apple Silicon (see DESIGN.md).

Status: checkpoint-backed tile fixture, CPU reference search, complete M2
Metal tile search for K=2..8, M3 direct no-LDL layer emit/load pilots,
M4 selected-module/layer-set Hessian/LDLQ emit with oracle proxy comparison,
post-M4 computed scales/calibration activation inputs, M5a allocation
scaffolding, bounded M5b LDLQ candidate measurement, and measured-budget
optimization.
The Metal path ports exllamav3's quantize_tiles_kernel shape and reuses the
inference side's decode_3inst snippet for bit parity by construction.
"""
