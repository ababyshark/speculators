"""Opt-in capture of actual native Ascend MTP inputs/outputs for parity."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from vllm.distributed import get_tensor_model_parallel_rank
from vllm_ascend.models.deepseek_mtp import AscendDeepSeekMTP
from vllm_ascend.ops.layernorm import AscendRMSNorm
from glm_teacher_sorted_sfa_extension import GLMTeacherSortedSFA

root = Path('/data1/mtp_training/parity')
original_forward = AscendDeepSeekMTP.forward
original_load_weights = AscendDeepSeekMTP.load_weights

def load_weights_with_optional_norm_bias(self, weights):
    loaded = original_load_weights(self, weights)
    removed = []
    for name, module in self.named_modules():
        if isinstance(module, AscendRMSNorm) and module.bias is not None and not module.bias_loaded:
            if bool(torch.count_nonzero(module.bias)):
                raise RuntimeError(f'Unloaded norm bias is not zero: {name}')
            # Main-model quantization allocates optional biases globally.
            # MTP norms without a source bias must remain bias-free.
            module.register_parameter('bias', None)
            removed.append(name + '.bias')
    if removed and get_tensor_model_parallel_rank() == 0:
        print('MTP_OPTIONAL_ZERO_BIASES_REMOVED ' + json.dumps(removed), flush=True)
    return loaded

AscendDeepSeekMTP.load_weights = load_weights_with_optional_norm_bias

def capture_forward(self, input_ids, positions, hidden_states,
                    intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0):
    request = None
    arm = root / 'native_capture_arm.json'
    if arm.exists():
        request = json.loads(arm.read_text())
        if spec_step_idx != 0 or int(positions.min()) != 0 or int(positions.max()) + 1 != request['length']:
            request = None
    # Clone before native kernels can alter the buffers.
    if request is not None:
        capture = dict(input_ids=input_ids.detach().clone(), positions=positions.detach().clone(),
                       teacher_hidden=hidden_states.detach().clone())
    result = original_forward(self, input_ids, positions, hidden_states,
                              intermediate_tensors, inputs_embeds, spec_step_idx)
    if request is not None:
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError('Expected native pre-norm and post-norm MTP outputs')
        pre, post = result
        # All ranks execute the head collective; only rank 0 writes files.
        length = request['length']
        logits = self.compute_logits(pre[length-1:length], spec_step_idx)
        if get_tensor_model_parallel_rank() == 0:
            capture.update(native_pre=pre[:length].detach(), native_post=post[:length].detach(),
                           native_last_logits=logits.detach())
            layer = self.model.layers[str(self.model.mtp_start_layer_idx)]
            mla = layer.mtp_block.self_attn.mla_attn
            if hasattr(mla, 'topk_indices_buffer'):
                capture['native_topk'] = mla.topk_indices_buffer[:length].detach().clone()
            save_file({k: v.cpu().contiguous() for k,v in capture.items()},
                      str(root / ('native_' + request['name'] + '.safetensors')))
            (root / ('native_' + request['name'] + '.json')).write_text(json.dumps(dict(
                name=request['name'], length=length, own_lm_head=bool(self.has_own_lm_head),
                keys=list(capture), layer_idx=self.model.mtp_start_layer_idx,
            ), indent=2))
            arm.unlink()
    return result

AscendDeepSeekMTP.forward = capture_forward

class GLMMTPCapture(GLMTeacherSortedSFA):
    pass
