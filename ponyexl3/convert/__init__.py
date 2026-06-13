"""PonyExl3: HF → EXL3 conversion on Apple Silicon (see DESIGN.md).

Status: design + reference stubs. The trellis search kernel ports
exllamav3's quantize_tiles_kernel to Metal, reusing the inference side's
decode_3inst snippet for bit parity by construction.
"""
