"""GLM MTP teacher: sorted SFA plus actual post-final-norm capture."""
from glm_teacher_sorted_sfa_extension import GLMTeacherSortedSFA
from vllm_ascend.worker.worker import NPUWorker

original_load_model = NPUWorker.load_model

def final_hidden_hook(module, args, output):
    if not isinstance(output, tuple) or len(output) != 2 or len(output[1]) != 1:
        raise RuntimeError("MTP teacher expects exactly one final-layer capture")
    final_hidden, auxiliary = output
    return final_hidden, [final_hidden]

def load_model_with_final_capture(self):
    result = original_load_model(self)
    config = self.vllm_config
    if not config.speculative_config.uses_extract_hidden_states():
        raise RuntimeError("This extension is only for MTP feature extraction")
    if config.parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("MTP teacher capture currently requires PP=1")
    model = self.model_runner.model
    model.register_forward_hook(final_hidden_hook)
    print(f"MTP_FINAL_CAPTURE_HOOK_INSTALLED {type(model).__name__}", flush=True)
    return result

# Install after load_model: Ascend replaces the backbone forward during setup,
# so an earlier class-level forward replacement is not a reliable capture point.
NPUWorker.load_model = load_model_with_final_capture

class GLMMTPTeacher(GLMTeacherSortedSFA):
    pass
