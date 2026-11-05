import functools
import gc
import json
import logging
import shutil
from abc import abstractmethod
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from transformers.modeling_utils import PreTrainedModel

from speculators.train.distributed import get_rank, is_distributed
from speculators.utils.util import get_current_device


def _fill_missing_named_optimizer_states(model, full_state_dict) -> int:
    """Insert zero AdamW moments (FQN-keyed, full tensors) for params missing
    from a DCP optimizer checkpoint — params whose grads never flowed (e.g.
    DSA indexer top-k, non-differentiable) so resume can proceed."""
    states = full_state_dict.setdefault("state", {})
    sample = next(
        (s for s in states.values() if isinstance(s, dict) and "exp_avg" in s), None
    )
    if sample is None:
        return 0  # not AdamW-style state; let the error propagate
    step = sample.get("step")
    filled = 0
    for name, p in model.named_parameters():
        if name in states or not p.requires_grad:
            continue
        states[name] = {
            "step": step.clone()
            if torch.is_tensor(step)
            else torch.tensor(step or 0.0),
            "exp_avg": torch.zeros(p.shape, dtype=sample["exp_avg"].dtype),
            "exp_avg_sq": torch.zeros(p.shape, dtype=sample["exp_avg_sq"].dtype),
        }
        filled += 1
    return filled


def _fill_missing_optimizer_states(opt, state_dict) -> int:
    """Insert zero AdamW moments for params missing from a checkpoint.

    torch assigns state labels by enumerating param_groups' params in order;
    params whose grads never flowed (e.g. DSA indexer, non-differentiable
    top-k) have no saved state and make strict resume fail.
    """
    states = state_dict.setdefault("state", {})
    steps = [s["step"] for s in states.values() if isinstance(s, dict) and "step" in s]
    if not any("exp_avg" in s for s in states.values() if isinstance(s, dict)):
        return 0  # not AdamW-style state; let the error propagate
    max_step = None
    for s in steps:
        if torch.is_tensor(s):
            max_step = s if max_step is None or float(s) > float(max_step) else max_step
        else:
            max_step = s if max_step is None or s > max_step else max_step
    label = 0
    filled = 0
    for group in opt.param_groups:
        for p in group["params"]:
            if label not in states:
                states[label] = {
                    "step": max_step.clone()
                    if torch.is_tensor(max_step)
                    else torch.tensor(max_step or 0.0),
                    "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ),
                }
                filled += 1
            label += 1
    return filled


logger = logging.getLogger("speculators")

# Optimizers/schedulers may be a single object (legacy) or a list (e.g. Muon + AdamW).
OptimizerOrList = torch.optim.Optimizer | list[torch.optim.Optimizer]
SchedulerOrList = (
    torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler]
)


def _as_list(value):
    """Normalize a single object or a list/tuple of objects into a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _rank0_only(fn):
    """Execute *fn* only on rank 0, then barrier (no-op when not distributed)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = None
        if get_rank() == 0:
            result = fn(*args, **kwargs)
        if is_distributed():
            dist.barrier()
        return result

    return wrapper


class BaseCheckpointer:
    """Helper class to save and load checkpoints.

    Checkpoint file structure:
    ../path/
        0/ # epoch number
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
        1/
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
        ...
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.previous_epoch = self._get_previous_epoch()

        if self.previous_epoch != -1:
            self.prev_path: Path | None = self.path / str(self.previous_epoch)
        else:
            self.prev_path = None

    @abstractmethod
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        raise NotImplementedError

    def load_scheduler_state_dict(self, scheduler: SchedulerOrList):
        scheduler_path = self.scheduler_path(self.previous_epoch)
        if not scheduler_path.exists():
            return
        loaded = torch.load(scheduler_path, weights_only=True)
        schedulers = _as_list(scheduler)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        for sched, state_dict in zip(schedulers, loaded_list, strict=True):
            sched.load_state_dict(state_dict)
            for param_group, lr in zip(
                sched.optimizer.param_groups, sched.get_last_lr(), strict=True
            ):
                param_group["lr"] = lr

    @_rank0_only
    def save_scheduler_state_dict(self, scheduler: SchedulerOrList, epoch: int | str):
        schedulers = _as_list(scheduler)
        state_dicts = [sched.state_dict() for sched in schedulers]
        # Preserve the legacy single-scheduler format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.scheduler_path(epoch))

    @abstractmethod
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError

    def _get_previous_epoch(self) -> int:
        if not self.path.exists():
            return -1
        last_checkpoint_num = -1
        for d in self.path.iterdir():
            if d.is_symlink():
                continue  # skip descriptive symlinks like epoch0_step16626
            if d.is_dir():
                if d.name == "interrupted":
                    logger.warning(
                        f"Found interrupted checkpoint at {d}. "
                        "To resume from it, rename it to an epoch number "
                        "(e.g., 'mv interrupted 5' to resume as epoch 5)."
                    )
                    continue
                try:
                    last_checkpoint_num = max(last_checkpoint_num, int(d.name))
                except ValueError:
                    continue
        return last_checkpoint_num

    def model_path(self, epoch: int | str):
        model_fname = "model.safetensors"
        return self.path / str(epoch) / model_fname

    def optimizer_path(self, epoch: int | str):
        optimizer_fname = "optimizer_state_dict.pt"
        return self.path / str(epoch) / optimizer_fname

    def scheduler_path(self, epoch: int | str):
        scheduler_fname = "scheduler_state_dict.pt"
        return self.path / str(epoch) / scheduler_fname

    def best_path(self) -> Path:
        return self.path / "checkpoint_best"

    def val_metrics_path(self, epoch: int) -> Path:
        return self.path / str(epoch) / "val_metrics.json"

    TRAIN_COMMAND_FILENAME = "train_command.txt"

    def _copy_train_command(self, epoch: int | str) -> None:
        src = self.path / self.TRAIN_COMMAND_FILENAME
        if src.exists():
            shutil.copy2(src, self.path / str(epoch) / self.TRAIN_COMMAND_FILENAME)

    @_rank0_only
    def save_val_metrics(self, epoch: int, val_metrics: dict[str, float]):
        path = self.val_metrics_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(val_metrics))

    def load_best_val_loss(self) -> float | None:
        best_epoch = self.read_best_epoch()
        if best_epoch is None:
            return None
        p = self.val_metrics_path(best_epoch)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return float(data["loss_epoch"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def read_best_epoch(self) -> int | None:
        """Return the epoch that `checkpoint_best` points to."""
        best_path = self.best_path()
        if not best_path.exists() or not best_path.is_symlink():
            return None
        try:
            target = best_path.readlink()
        except OSError:
            return None
        try:
            return int(Path(target).name)
        except ValueError:
            return None

    def load_model_state_dict_for_epoch(
        self, model: PreTrainedModel, epoch: int, float_dtype: torch.dtype | None = None
    ):
        """Temporarily load weights for a specific epoch."""
        old_epoch = self.previous_epoch
        try:
            self.previous_epoch = epoch
            self.load_model_state_dict(model, float_dtype=float_dtype)
        finally:
            self.previous_epoch = old_epoch

    @_rank0_only
    def update_best_symlink(self, epoch: int):
        best_path = self.best_path()
        target = Path(str(epoch))  # relative symlink inside checkpoint root

        if best_path.is_symlink() or best_path.exists():
            if best_path.is_dir() and not best_path.is_symlink():
                shutil.rmtree(best_path)
            else:
                best_path.unlink()

        best_path.symlink_to(target, target_is_directory=True)

    @_rank0_only
    def cleanup_keep_only_best(self, best_epoch: int) -> None:
        """
        Delete all epoch dir. except best_epoch, and keep best_checkpoint symlink.
        """
        keep_dir = self.path / str(best_epoch)
        best_link = self.best_path()

        # Safety checks
        if not keep_dir.exists() or not keep_dir.is_dir():
            raise FileNotFoundError(f"Best epoch dir does not exist: {keep_dir}")

        train_cmd_file = self.path / self.TRAIN_COMMAND_FILENAME

        for child in self.path.iterdir():
            # Keep the symlink itself
            if child == best_link:
                continue

            # Keep the best epoch directory
            if child == keep_dir:
                continue

            if child == train_cmd_file:
                continue

            # Delete numbered epoch directories and any other stray dirs/files
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise RuntimeError(f"Failed to delete {child}") from exc


# Suffixes of tensors that must KEEP their training-time dtype when the model
# state dict is cast for checkpointing. e_score_correction_bias is the
# DeepSeek aux-loss-free router balancing bias: a BUFFER (never trained), it
# accumulates to ~15.8 magnitude while its informative spread is only ~0.08.
# bf16 rounding (~0.016 mean at that magnitude = 11.5% of the spread) flips
# the top-8 expert selection for 75% of tokens (measured on real hidden
# states), which collapses serving acceptance. Keep it fp32 end to end.
PRECISION_PRESERVE_SUFFIXES = ("e_score_correction_bias",)


def cast_model_state_dict(sd: dict, dtype: torch.dtype | None) -> dict:
    """Named-tensor cast for checkpointing that preserves precision-sensitive
    tensors (see PRECISION_PRESERVE_SUFFIXES)."""
    out = {}
    for key, value in sd.items():
        if (
            dtype is not None
            and isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and not key.endswith(PRECISION_PRESERVE_SUFFIXES)
        ):
            out[key] = value.to(dtype)
        else:
            out[key] = value
    return out


def convert_float_dtype(sd: pytree.PyTree, dtype: torch.dtype) -> pytree.PyTree:
    def convert_fn(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(dtype)
        return x

    return pytree.tree_map(convert_fn, sd)


def load_safetensors_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    full_state_dict = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():  # noqa: SIM118
            full_state_dict[key] = f.get_tensor(key)
    return full_state_dict


def patch_config_dtype(config_path: Path, float_dtype: torch.dtype) -> None:
    """Patch config.json to match the actual on-disk tensor dtype.

    When models are kept in FP32 but saved as BF16, save_pretrained writes
    the in-memory dtype to config.json. This patches it to match the saved dtype.
    """
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    # Convert torch.bfloat16 -> "bfloat16"
    dtype_str = str(float_dtype).split(".")[-1]
    # Support both dtype (transformers 5.x) and torch_dtype (older versions)
    if "dtype" in config:
        config["dtype"] = dtype_str
    if "torch_dtype" in config:
        config["torch_dtype"] = dtype_str
    config_path.write_text(json.dumps(config, indent=2) + "\n")


class SingleGPUCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        device = get_current_device()
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch),
            device,
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        # Note: `strict=False` because we don't load the verifier weights
        model.load_state_dict(full_state_dict, strict=False)

    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        device = get_current_device()
        loaded = torch.load(
            self.optimizer_path(self.previous_epoch),
            weights_only=True,
            map_location=device,
        )
        optimizers = _as_list(optimizer)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        dtype = float_dtype or raw_model.dtype
        for opt, state_dict in zip(optimizers, loaded_list, strict=True):
            try:
                opt.load_state_dict(convert_float_dtype(state_dict, dtype))
            except (RuntimeError, ValueError) as e:
                # Params whose gradients never flowed (e.g. DSA indexer top-k
                # selection is non-differentiable) have no optimizer state in
                # the checkpoint. Fill zero moments for them (identical to
                # their epoch-1 behavior) so resume can proceed.
                if "no saved optimizer state" not in str(e):
                    raise
                filled = _fill_missing_optimizer_states(opt, state_dict)
                logging.getLogger(__name__).warning(
                    "Optimizer checkpoint missing state for %d zero-grad "
                    "params (e.g. DSA indexer); filled with zero moments.",
                    filled,
                )
                opt.load_state_dict(convert_float_dtype(state_dict, dtype))

    @_rank0_only
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raw_model: PreTrainedModel = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )  # type: ignore[assignment]
        model_state_dict = convert_float_dtype(raw_model.state_dict(), float_dtype)
        raw_model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
        patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)

        optimizers = _as_list(optimizer)
        state_dicts = [
            convert_float_dtype(opt.state_dict(), float_dtype) for opt in optimizers
        ]
        # Preserve the legacy single-optimizer format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.optimizer_path(epoch))
        self._copy_train_command(epoch)


class DistributedCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch), "cpu"
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        # Note: `strict=False` because we don't load the verifier weights
        set_model_state_dict(
            model,
            full_state_dict,  # type: ignore[arg-type]
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
        dist.barrier()

    def load_optimizer_state_dict(
        self,
        model,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        optimizers = _as_list(optimizer)
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        filled = _fill_missing_named_optimizer_states(model, full_state_dict)
        if filled:
            logging.getLogger(__name__).warning(
                "Optimizer checkpoint missing state for %d zero-grad params "
                "(e.g. DSA indexer top-k, non-differentiable); filled with "
                "zero moments.",
                filled,
            )

        set_optimizer_state_dict(
            model,
            optimizers,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

        # Cast step counters back to float32
        for opt in optimizers:
            for state in opt.state.values():
                if "step" in state and isinstance(state["step"], torch.Tensor):
                    state["step"] = state["step"].float()

        dist.barrier()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        # Free allocator caches before the DCP full-gather: the gather
        # re-shards each DTensor to FULL on device before cpu_offload, and the
        # training working set sits at the ~65 GB card limit — the largest
        # tensor (grouped gate_up, ~26 GB bf16) plus leftover activation cache
        # blocks can tip the allocation (observed: 5.9 MB short, HcclAllGather
        # OOM at the epoch-1 save).
        gc.collect()
        for _m in ("npu", "cuda"):
            _mod = getattr(torch, _m, None)
            if _mod is None or not hasattr(_mod, "empty_cache"):
                continue
            try:
                if _m == "cuda" and not _mod.is_available():
                    continue
                _mod.empty_cache()
            except Exception:  # noqa: BLE001 -- best-effort cache release
                logger.debug("empty_cache(%s) failed during save", _m, exc_info=True)

        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        model_state_dict = cast_model_state_dict(model_state_dict, float_dtype)

        gc.collect()
        for _m in ("npu", "cuda"):
            _mod = getattr(torch, _m, None)
            if _mod is None or not hasattr(_mod, "empty_cache"):
                continue
            try:
                if _m == "cuda" and not _mod.is_available():
                    continue
                _mod.empty_cache()
            except Exception:  # noqa: BLE001 -- best-effort cache release
                logger.debug("empty_cache(%s) failed during save", _m, exc_info=True)

        optimizer_state_dict = get_optimizer_state_dict(
            model,
            _as_list(optimizer),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # Keep optimizer moments in their native fp32: casting moments to bf16
        # loses update precision and compounds across resumes.
        optimizer_state_dict = convert_float_dtype(optimizer_state_dict, None)

        if get_rank() == 0:
            # Only rank 0 saves the checkpoint
            model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
            patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)
            torch.save(optimizer_state_dict, self.optimizer_path(epoch))
            self._copy_train_command(epoch)

        dist.barrier()
