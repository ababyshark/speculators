import logging
import time
import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.endpoint_router import EndpointRouter
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    InvalidResponseError,
    generate_hidden_states,
)
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    GenerationRecoveryGuard,
    RecoveryMetadata,
    SampleUnavailable,
)

BatchType = dict[str, Any]
logger = logging.getLogger("speculators")

DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1"

# Server-side failure modes worth failing over to another endpoint for.
# Connection errors and timeouts are covered by openai's exception hierarchy;
# these status codes are additionally treated as endpoint-scoped faults.
FAILOVER_STATUS_CODES = frozenset({408, 429, 502, 503, 504})

# Backoff ceiling for retries that reuse the same endpoint (dead-end fallback
# when every endpoint is in cooldown). Mirrors vllm_client.RETRY_BACKOFF_BASE.
SAME_ENDPOINT_BACKOFF_CAP = 8


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    out_dict: dict = {"input_ids": dataset_item["input_ids"].tolist()}

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]

    return cast("ClientItem", out_dict)


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index: int) -> BatchType | SampleUnavailable:
        raise NotImplementedError

    def __getitem__(self, index) -> BatchType | SampleUnavailable:
        data = self._get_raw_data(index)

        if isinstance(data, SampleUnavailable):
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        return data


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = DEFAULT_VLLM_ENDPOINT,
        vllm_endpoints: Sequence[str] | None = None,
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        train_ratio: float = 1.0,
        split: Literal["train", "val"] = "train",
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        generation_validation_retries: int = 2,
        max_consecutive_generation_failures: int = 20,
        router: EndpointRouter | None = None,
    ):
        self.data = load_from_disk(datapath)
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0.0, 1.0], got {train_ratio}")
        if split == "val" and train_ratio == 1.0:
            raise ValueError("train_ratio=1.0 leaves no validation split")

        # Both splits derive their boundary from this one expression,
        # so they are exactly complementary.
        split_idx = int(len(self.data) * train_ratio)
        start, stop = (
            (0, split_idx) if split == "train" else (split_idx, len(self.data))
        )
        if start >= stop:
            raise ValueError(
                f"{split} split is empty (dataset has {len(self.data)} rows, "
                f"train_ratio={train_ratio} gives split_idx={split_idx})"
            )
        self.start_file_idx = start
        self.data = self.data.select(range(start, stop))

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.on_missing = on_missing
        self._requested_model = model
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.generation_recovery = GenerationRecoveryGuard(
            retries=generation_validation_retries,
            max_consecutive_failures=max_consecutive_generation_failures,
        )

        if router is not None:
            self.router = router
        else:
            endpoints = (
                list(vllm_endpoints)
                if vllm_endpoints
                else [vllm_endpoint or DEFAULT_VLLM_ENDPOINT]
            )
            self.router = EndpointRouter(endpoints)
        self._endpoints_ready = False
        self._validated_endpoints: set[str] = set()

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, hidden_states_dtype)

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def _setup_endpoints(self) -> None:
        """First-use initialization: transfer setup + model discovery per endpoint.

        Every endpoint gets a model-id probe so a misrouted endpoint (serving a
        different model) is evicted before it poisons a sample. Discovery or
        transfer failures leave ``_endpoints_ready`` False so the outer
        full-round-trip retry (``GenerationRecoveryGuard``) redoes
        initialization.
        """
        self.transfer.setup()

        discovery_errors: list[str] = []
        for endpoint in self.router.endpoints:
            client = self.router.client_for(endpoint)
            try:
                model_id = client.models.list().data[0].id
            except Exception as error:  # noqa: BLE001 - endpoint-scoped fault
                self.router.report_failure(endpoint)
                discovery_errors.append(f"{endpoint}: {type(error).__name__}: {error}")
                logger.warning(
                    "Model discovery failed for vLLM endpoint %s: %s: %s",
                    endpoint,
                    type(error).__name__,
                    error,
                )
                continue

            if self.model is None:
                self.model = model_id
            if model_id != self.model:
                self.router.evict(endpoint)
                discovery_errors.append(
                    f"{endpoint}: serves model '{model_id}', expected '{self.model}'"
                )
                logger.error(
                    "vLLM endpoint %s serves model '%s' but expected '%s'; "
                    "evicted from routing",
                    endpoint,
                    model_id,
                    self.model,
                )
                continue
            self._validated_endpoints.add(endpoint)

        if not self._validated_endpoints:
            detail = "; ".join(discovery_errors) or "no endpoints configured"
            raise RuntimeError(
                "No usable vLLM endpoint for hidden-state generation. "
                f"Endpoint probe results: {detail}"
            )
        # Do not retain half-initialized state if any later step fails; the
        # outer full-round-trip retry should redo initialization.
        self._endpoints_ready = True

    def _generate_with_failover(
        self,
        client_item: ClientItem,
    ) -> tuple[str, dict[str, Any]]:
        """Attempt one generation, failing over between healthy endpoints.

        Retry budget is ``max_retries + 1`` attempts, redistributed across
        endpoints: connection-level failures rotate to the next healthy
        endpoint without backoff, while a repeat pick of the same endpoint
        (dead-end fallback when everything is in cooldown) backs off like the
        single-endpoint path did. Model/transfer-level failures are not
        endpoint faults and propagate immediately.
        """
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        previous_endpoint: str | None = None
        for attempt in range(attempts):
            client, endpoint = self.router.pick()
            if previous_endpoint == endpoint and attempt > 0:
                time.sleep(min(2**attempt, SAME_ENDPOINT_BACKOFF_CAP))
            previous_endpoint = endpoint
            try:
                handle, hs_transfer_params = generate_hidden_states(
                    client,
                    self.model,  # type:ignore[arg-type]
                    client_item,
                    timeout=self.request_timeout,
                    max_retries=0,  # retries are handled here, across endpoints
                )
            except InvalidResponseError:
                # The endpoint answered; content validation is not its fault.
                self.router.report_success(endpoint)
                raise
            except openai.APIStatusError as error:
                if error.status_code in FAILOVER_STATUS_CODES:
                    # Endpoint-scoped server fault: evict path, fail over.
                    self.router.report_failure(endpoint)
                    last_error = error
                    logger.warning(
                        "vLLM endpoint %s returned status %d; failing over",
                        endpoint,
                        error.status_code,
                    )
                else:
                    # Semantic error (bad request, auth, ...): the same
                    # request would fail on any endpoint, so don't poison
                    # endpoint health -- fail fast.
                    self.router.report_success(endpoint)
                    raise
            except openai.APIConnectionError as error:
                # Includes APITimeoutError.
                self.router.report_failure(endpoint)
                last_error = error
                logger.warning(
                    "vLLM endpoint %s connection failure; failing over: %s",
                    endpoint,
                    error,
                )
            else:
                self.router.report_success(endpoint)
                return handle, hs_transfer_params
        if last_error is None:  # pragma: no cover - loop guarantees one record
            raise RuntimeError("Endpoint failover exhausted without an error")
        raise last_error

    def _generate_hidden_states_once(
        self,
        dataset_item: dict,
        client_item: ClientItem,
    ) -> dict[str, torch.Tensor]:
        handle: str | None = None
        try:
            if not self._endpoints_ready:
                self._setup_endpoints()
            handle, hs_transfer_params = self._generate_with_failover(client_item)

            loaded_hs = self.transfer.get_generated(hs_transfer_params)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest validation first.
            check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())

            try:
                self.transfer.delete(handle)
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning(
                    "Loaded a valid hidden-state sample but failed to delete "
                    "handle %s: %s",
                    handle,
                    cleanup_error,
                )
            return loaded_hs
        except Exception:
            if handle is not None:
                try:
                    self.transfer.delete(handle)
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Failed to clean generated hidden-state handle %s: %s",
                        handle,
                        cleanup_error,
                    )
            raise

    def _get_raw_data(self, index: int) -> BatchType | SampleUnavailable:
        file_idx = self._map_to_file_idx(index)
        cached_hs = self.transfer.get_cached(file_idx)
        if cached_hs is None:
            if self.on_missing == "generate":
                dataset_item = self.data[index]
                client_item = build_client_item(dataset_item)
                loaded_hs = self.generation_recovery.run(
                    lambda: self._generate_hidden_states_once(
                        dataset_item,
                        client_item,
                    ),
                    description=(
                        f"Hidden-state round trip failed for dataset index {index}, "
                        f"file index {file_idx}"
                    ),
                )
            elif self.on_missing == "skip":
                return SampleUnavailable()
            elif self.on_missing == "warn":
                warnings.warn(
                    f"Failed to load hidden states for sample {index}. Skipping...",
                    stacklevel=1,
                )
                return SampleUnavailable(
                    reason=f"Hidden states unavailable for sample {index}"
                )
            else:
                raise RuntimeError(f"Failed to load hidden states for sample {index}.")
        else:
            loaded_hs = cached_hs

        if isinstance(loaded_hs, SampleUnavailable):
            return loaded_hs

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        if not torch.equal(loaded_hs["token_ids"], self.data[index]["input_ids"]):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {self.data[index]['input_ids']}",
                stacklevel=1,
            )
            return SampleUnavailable(
                reason=f"Cached token ids do not match sample {index}"
            )

        return {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": self.data[index]["loss_mask"],  # [seq_len]
        }


class CollateFn:
    """Picklable collate function for use with ``multiprocessing_context='spawn'``."""

    def __init__(
        self,
        max_len: int,
        hidden_size: int,
        num_target_layers: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        preprocess: Callable[[BatchType], BatchType] | None = None,
    ):
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_target_layers = num_target_layers
        self.dtype = dtype
        self.preprocess = preprocess

    def _clean_batch(
        self, batch: Sequence[BatchType | SampleUnavailable | None]
    ) -> tuple[list[BatchType], list[SampleUnavailable], int]:
        """Preprocess valid samples and collect unavailable and dropped samples."""
        preprocess = self.preprocess
        unavailable = []
        num_dropped = 0
        new_batch = []
        for item in batch:
            if item is None:
                num_dropped += 1
                continue
            if isinstance(item, SampleUnavailable):
                unavailable.append(item)
                num_dropped += 1
                continue

            new_batch.append(preprocess(item) if preprocess else item)

        return new_batch, unavailable, num_dropped

    def __call__(
        self, batch: Sequence[BatchType | SampleUnavailable | None]
    ) -> BatchType:
        max_len = self.max_len
        dtype = self.dtype

        batch, unavailable, num_dropped = self._clean_batch(batch)

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(
                self.hidden_size, self.num_target_layers, dtype=dtype
            )
            if self.preprocess:
                empty = self.preprocess(empty)
            batch = [empty]
            locally_empty = True
        else:
            locally_empty = False

        collated_data: BatchType = {}
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "lengths":
                collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
                continue
            # one copy per sample: preallocated buffer, hidden states cast during write
            first = batch[0][key]  # type: ignore[index]
            buffer_dtype = dtype if "hidden_states" in key else first.dtype
            out = torch.zeros(
                (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
            )
            offset = 0
            for b in batch:
                tensor = b[key]  # type: ignore[index]
                num_rows = min(tensor.shape[0], max_len - offset)
                out[offset : offset + num_rows] = tensor[:num_rows]
                offset += num_rows
                if offset == max_len:
                    break
            collated_data[key] = out.unsqueeze(0)
            # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids

        collated_data["error_records"] = num_dropped
        metadata = RecoveryMetadata.from_unavailable(
            unavailable,
            locally_empty=locally_empty,
        )
        if metadata.failure_count or metadata.locally_empty:
            collated_data[RECOVERY_METADATA_KEY] = metadata

        return collated_data
