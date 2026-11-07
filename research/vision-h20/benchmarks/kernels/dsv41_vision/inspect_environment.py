"""Record environment and safetensors metadata without loading the LLM."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

root = Path("/model")
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
selected = {
    k: v
    for k, v in index.items()
    if k.startswith(("vision.", "aligner."))
    or k in ("image_start", "image_end", "image_newline")
}
info = {}
for shard in sorted(set(selected.values())):
    with safe_open(str(root / shard), framework="pt", device="cpu") as f:
        for k in selected:
            if selected[k] == shard:
                s = f.get_slice(k)
                info[k] = dict(shard=shard, shape=s.get_shape(), dtype=s.get_dtype())
versions = {}
for package in [
    "torch",
    "triton",
    "vllm",
    "safetensors",
    "Pillow",
    "flash-attn",
    "flashinfer-python",
]:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
print(
    json.dumps(
        dict(
            platform=platform.platform(),
            versions=versions,
            cuda=torch.version.cuda,
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
                    "--format=csv",
                ],
                text=True,
            ),
            topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
            config=json.loads((root / "config.json").read_text()),
            config_sha256=hashlib.sha256(
                (root / "config.json").read_bytes()
            ).hexdigest(),
            weight_map_sha256=hashlib.sha256(
                (root / "model.safetensors.index.json").read_bytes()
            ).hexdigest(),
            selected_weight_count=len(selected),
            selected_weights=info,
            full_model_tensors_loaded=False,
        ),
        indent=2,
    )
)
