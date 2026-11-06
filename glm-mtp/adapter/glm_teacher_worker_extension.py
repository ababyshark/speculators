"""Opt-in deterministic operators for the GLM feature-extraction workers."""
import torch
from vllm.logger import init_logger

torch.use_deterministic_algorithms(True)
init_logger(__name__).info("GLM teacher deterministic operators enabled: %s", torch.are_deterministic_algorithms_enabled())


class GLMTeacherDeterminism:
    """Module import enables deterministic operators in each spawned worker."""
