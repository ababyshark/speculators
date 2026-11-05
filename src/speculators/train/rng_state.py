"""Serializable per-rank RNG state for restartable online training."""
import random
from contextlib import contextmanager

import numpy as np
import torch


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
    }
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is not None:
        state["device_type"] = accelerator.type
        state["device"] = torch.get_device_module(accelerator.type).get_rng_state()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    name, values, *tail = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), *tail))
    torch.set_rng_state(state["torch"].cpu())
    if "device" in state:
        torch.get_device_module(state["device_type"]).set_rng_state(state["device"].cpu())


@contextmanager
def validation_rng(seed=0):
    state = capture_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is not None:
            torch.get_device_module(accelerator.type).manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)
