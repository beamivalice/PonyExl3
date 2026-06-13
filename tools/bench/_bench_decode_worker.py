
import json, os, sys, time

import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from ponyexl3.mlx.generate import stream_generate, speculative_stream_generate
from ponyexl3.mlx.layer_state import clear_layer_caches
from ponyexl3.mlx.model import load_model
from ponyexl3.mlx.mtp import load_mtp

model_dir = sys.argv[1]
steps = int(sys.argv[2])
warmup = int(sys.argv[3])
mtp_dir = sys.argv[4]

for k, v in json.loads(sys.argv[5]).items():
    os.environ[k] = v

model, config = load_model(model_dir, engine="exl3", warm=True, verbose=False)
tok = load_tokenizer(model_dir)
prompt = "Write a short Python function to compute fibonacci."
if getattr(tok, "chat_template", None):
    prompt_ids = list(tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True))
else:
    prompt_ids = list(tok.encode(prompt))

use_mtp = os.environ.get("USE_MTP") == "1"
mtp = None
num_draft = int(os.environ.get("NUM_DRAFT", "3"))
if use_mtp:
    mtp = load_mtp(model_dir, config, mtp_dir)

def run_gen():
    if mtp is not None:
        return speculative_stream_generate(
            model, mtp, prompt_ids, max_tokens=steps, num_draft=num_draft)
    return stream_generate(model, prompt_ids, max_tokens=steps)

for _ in range(warmup):
    list(run_gen())
    clear_layer_caches()

t0 = time.perf_counter()
n = 0
for _ in run_gen():
    n += 1
mx.synchronize()
dt = time.perf_counter() - t0
print(json.dumps({"tok_s": n / dt if dt else 0.0, "steps": n, "seconds": dt}))
