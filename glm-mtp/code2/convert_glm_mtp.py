"""Convert the verified native extraction to a loadable Speculators checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.models.mtp.glm_mtp import native_key_to_training
from speculators.proposals.greedy import GreedyTokenProposalConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    src, dst = Path(args.source), Path(args.output)
    if dst.exists():
        raise FileExistsError(dst)
    index_path = src / 'model.safetensors.index.json'
    if not index_path.exists():
        # GLM-5.3 quantized verifier uses a quant-specific index filename.
        index_path = src / 'quant_model_weights.safetensors.index.json'
    if not index_path.exists():
        raise FileNotFoundError(
            f'No safetensors index under {src}; expected model.safetensors.index.json '
            'or quant_model_weights.safetensors.index.json'
        )
    index = json.loads(index_path.read_text())['weight_map']
    config = MTPSpeculatorConfig(
        transformer_layer_config=AutoConfig.from_pretrained(src),
        mtp_shared_weights_path=str(src.resolve()),
        speculators_config=SpeculatorsConfig(
            algorithm='mtp', default_proposal_method='greedy',
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            verifier=VerifierConfig(name_or_path=args.teacher, architectures=['GlmMoeDsaForCausalLM']),
        ),
    )
    with torch.device('meta'):
        model = MTPDraftModel(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()
                if k not in ('verifier_lm_head.weight',)}
    layer_index = config.transformer_layer_config.num_hidden_layers
    mapping = {
        key: native_key_to_training(key, layer_index)
        for key in index
        if key.startswith(f'model.layers.{layer_index}.')
    }
    if not mapping:
        raise ValueError(f'No native MTP keys found at model.layers.{layer_index}.*')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Conversion mapping is not bijective')
    if set(mapping.values()) != set(expected):
        raise ValueError({'missing': sorted(set(expected) - set(mapping.values())),
                          'unexpected': sorted(set(mapping.values()) - set(expected))})
    dst.mkdir(parents=True)
    config.save_pretrained(dst)
    output_index, records = {}, {}
    for number, shard in enumerate(sorted(set(index.values())), 1):
        tensors = {}
        with safe_open(str(src / shard), framework='pt') as f:
            for key in sorted(
                k for k, value in index.items()
                if value == shard and k in mapping
            ):
                tensor = f.get_tensor(key)
                newkey = mapping[key]
                if tuple(tensor.shape) != expected[newkey]:
                    raise ValueError(f'Shape mismatch: {key}: {tuple(tensor.shape)} != {expected[newkey]}')
                digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                records[newkey] = dict(source_key=key, shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest)
                if newkey not in ('embed_tokens.weight', 'lm_head.weight'):
                    tensors[newkey] = tensor.contiguous()
        if not tensors:
            continue
        filename = f'model-{number:05d}.safetensors'
        save_file(tensors, str(dst / filename))
        with safe_open(str(dst / filename), framework='pt') as f:
            for key in tensors:
                actual = hashlib.sha256(f.get_tensor(key).view(torch.uint8).numpy().tobytes()).hexdigest()
                if actual != records[key]['sha256']:
                    raise RuntimeError(f'Round-trip mismatch: {key}')
                output_index[key] = filename
        print(f'verified {filename}: {len(tensors)} tensors', flush=True)
    (dst / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': output_index}, indent=2))
    (dst / 'conversion_report.json').write_text(json.dumps(dict(
        status='passed', source=str(src), teacher=args.teacher, tensor_count=len(records),
        saved_tensor_count=len(output_index), shared_weights_path=str(src),
        mapping=records, semantics_parity='NOT_YET_VALIDATED'), indent=2))
    print(f'CONVERSION_PASSED {len(records)} tensors, independent shared head retained', flush=True)

if __name__ == '__main__':
    main()
