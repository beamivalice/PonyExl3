"""PonyExl3: HF → EXL3 conversion on Apple Silicon (see DESIGN.md).

Status: checkpoint-backed tile fixture, CPU reference search, and complete M2
Metal tile search for K=2..8. The Metal path ports exllamav3's
quantize_tiles_kernel shape and reuses the inference side's decode_3inst
snippet for bit parity by construction.
"""
