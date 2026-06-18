"""PonyExl3: HF → EXL3 conversion on Apple Silicon (see DESIGN.md).

Status: checkpoint-backed tile fixture, CPU reference search, and Metal tile
search pilot. The Metal path ports exllamav3's quantize_tiles_kernel shape and
reuses the inference side's decode_3inst snippet for bit parity by construction.
"""
