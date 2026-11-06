"""Build the canonical capture manifest for our offline-extracted HS data.

Uses capture.py's own build_capture_manifest + write_capture_manifest so the
contract is enforced at write time (one pass, no hand-reverse-engineering).

Our data: the offline extraction (vllm-ascend v0.28 extract_hidden_states,
eagle_aux_hidden_state_layer_ids=[78]) captured the verifier layer-77 RAW
output = PRE-final-norm, bf16. The training (fastmtp + the pre_norm stage)
applies the frozen verifier_norm in-forward.
"""

import glob
import hashlib
import sys

import torch

sys.path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src")
)
from speculators.models.mtp.capture import (
    build_capture_manifest,
    write_capture_manifest,
)

DATA = "/mnt/share_space/l00840580/dspark-glm53/data_prep_full"
MODEL = "/mnt/weight2/l00840580-glm53-w4a8-biasfix"
OUT = "/mnt/share_space/l00840580/dspark-glm53/data_prep_full/capture_manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = sorted(glob.glob(DATA + "/hidden_states/hs_*.safetensors"))
    assert files, "no HS files"

    # the model.norm.weight from the verifier checkpoint
    from safetensors import safe_open

    norm_w = None
    for f in sorted(glob.glob(MODEL + "/quant_model_weights-*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            if "model.norm.weight" in sf.keys():
                norm_w = sf.get_tensor("model.norm.weight").to(torch.float32)
                break
    assert norm_w is not None

    # the input bytes: the tokenized dataset
    input_sha = sha256_file(DATA + "/../data/tok_full.jsonl")
    pos_sha = hashlib.sha256(
        torch.arange(2048, dtype=torch.int64).numpy().tobytes()
    ).hexdigest()

    manifest = build_capture_manifest(
        hidden_state_stage="pre_norm",
        verifier_source_layer_ordinal=78,
        vllm_target_layer_id=78,
        native_mtp_layer_index=78,
        native_mtp_key_prefix="model.layers.78.",
        capture_files=files,
        model_norm_weight=norm_w,
        # GLM-5.3 config.indexer_types is per-layer; the capture source layer
        # (0-based 77 = ordinal 78) is 'shared'.
        verifier_source_indexer_type="shared",
        source_norm="model.norm",
        hidden_size=6144,
        dtype="bfloat16",
        input_bytes_sha256=input_sha,
        position_ids_sha256=pos_sha,
        extra={
            "extraction": "vllm-ascend v0.28 extract_hidden_states connector",
            "note": "offline capture; the rot transform lives in the serving "
            "stack (rot.weight sidecar), not in the training data",
        },
    )
    write_capture_manifest(OUT, manifest, verify_files=True)
    print(f"manifest written+validated -> {OUT} ({len(files)} capture files)")


if __name__ == "__main__":
    main()
