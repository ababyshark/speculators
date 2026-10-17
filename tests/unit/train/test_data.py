"""Unit tests for data processing in speculators.train.data."""

import logging
from pathlib import Path

import httpx
import openai
import torch
from datasets import Dataset

import speculators.train.data as data_module
from speculators.data_generation.endpoint_router import EndpointRouter
from speculators.models.eagle3.data import shift_batch
from speculators.train.data import (
    ArrowDataset,
    CollateFn,
)
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    RecoveryMetadata,
    SampleUnavailable,
)


class _EndpointClient:
    """Stand-in for an openai client, tagged with its endpoint URL."""

    def __init__(self, name: str):
        self.name = name


def _connection_error(endpoint: str) -> openai.APIConnectionError:
    request = httpx.Request("POST", endpoint)
    return openai.APIConnectionError(request=request)


def _status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://fake:8000/v1")
    response = httpx.Response(code, request=request)
    return openai.APIStatusError(f"status {code}", response=response, body=None)


def test_shift_batch():
    """Test shift_batch function."""
    batch = {
        "input_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "hidden_states": torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ]
        ),
        "verifier_last_hidden_states": torch.tensor(
            [[10.0], [11.0], [12.0], [13.0], [14.0]]
        ),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "lengths": torch.tensor([5], dtype=torch.long),
        "position_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
    }

    expected_output = {
        "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "hidden_states": torch.tensor(
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        ),
        "verifier_last_hidden_states": torch.tensor([[11.0], [12.0], [13.0], [14.0]]),
        "loss_mask": torch.tensor([0, 1, 1, 1], dtype=torch.long),
        "lengths": torch.tensor([4], dtype=torch.long),
        "position_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
    }

    shifted = shift_batch(batch)

    for key, value in shifted.items():
        assert torch.allclose(value, expected_output[key])


def test_collate_fn_basic():
    """Test basic collation functionality."""
    max_len = 10
    hidden_size = 1
    num_target_layers = 3
    collate_fn = CollateFn(
        max_len, hidden_size, num_target_layers=num_target_layers, dtype=torch.float32
    )

    batch = [
        {
            "input_ids": torch.tensor([0, 1], dtype=torch.long),
            "hidden_states": torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]),
            "verifier_last_hidden_states": torch.tensor([[2.0], [3.0]]),
            "loss_mask": torch.tensor([0, 1], dtype=torch.long),
            "lengths": torch.tensor([2], dtype=torch.long),
            "position_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long),
            "hidden_states": torch.tensor(
                [
                    [4.0, 4.1, 4.2],
                    [5.0, 5.1, 5.2],
                    [6.0, 6.1, 6.2],
                    [7.0, 7.1, 7.2],
                    [8.0, 8.1, 8.2],
                    [9.0, 9.1, 9.2],
                ]
            ),
            "verifier_last_hidden_states": torch.tensor(
                [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]
            ),
            "loss_mask": torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long),
            "lengths": torch.tensor([6], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        },
    ]

    expected_output = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, -1, -1]], dtype=torch.long),
        "hidden_states": torch.tensor(
            [
                [
                    [0.0, 0.1, 0.2],
                    [1.0, 1.1, 1.2],
                    [4.0, 4.1, 4.2],
                    [5.0, 5.1, 5.2],
                    [6.0, 6.1, 6.2],
                    [7.0, 7.1, 7.2],
                    [8.0, 8.1, 8.2],
                    [9.0, 9.1, 9.2],
                    [-1, -1, -1],
                    [-1, -1, -1],
                ]
            ]
        ),
        "verifier_last_hidden_states": torch.tensor(
            [[[2.0], [3.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [-1], [-1]]]
        ),
        "loss_mask": torch.tensor([[0, 1, 0, 0, 1, 0, 1, 1, -1, -1]], dtype=torch.long),
        "document_ids": torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, -1, -1]], dtype=torch.long
        ),
        "position_ids": torch.tensor(
            [[0, 1, 0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.long
        ),
        "error_records": 0,
    }

    collated = collate_fn(batch)

    for key, value in collated.items():
        if isinstance(value, torch.Tensor):
            assert isinstance(expected_output[key], torch.Tensor)
            assert value.shape == expected_output[key].shape  # type: ignore[attr-defined]

            is_masking = expected_output[key] == -1
            assert torch.all(
                torch.isclose(value[~is_masking], expected_output[key][~is_masking])  # type: ignore[index]
            )

        else:
            assert value == expected_output[key]


def test_collate_fn_casts_hidden_states_dtype():
    """Test that hidden-states keys are cast to the target dtype during collation."""
    collate_fn = CollateFn(4, 1, dtype=torch.bfloat16)
    batch = [
        {
            "input_ids": torch.tensor([0], dtype=torch.long),
            "hidden_states": torch.ones(1, 3, dtype=torch.float32),
            "verifier_last_hidden_states": torch.ones(1, 1, dtype=torch.float32),
            "loss_mask": torch.ones(1, dtype=torch.long),
            "lengths": torch.tensor([1], dtype=torch.long),
            "position_ids": torch.tensor([0], dtype=torch.long),
        }
    ]

    collated = collate_fn(batch)

    assert collated["hidden_states"].dtype == torch.bfloat16
    assert collated["verifier_last_hidden_states"].dtype == torch.bfloat16
    assert collated["input_ids"].dtype == torch.long


def test_collate_fn_length_truncation():
    """Test that lengths are truncated when they exceed max_len."""
    max_len = 11
    hidden_size = 8
    num_target_layers = 3
    collate_fn = CollateFn(
        max_len, hidden_size, num_target_layers=num_target_layers, dtype=torch.float32
    )

    batch = [
        {
            "input_ids": torch.arange(5, dtype=torch.long),
            "hidden_states": torch.randn(5, num_target_layers * hidden_size),
            "verifier_last_hidden_states": torch.randn(5, hidden_size),
            "loss_mask": torch.ones(5, dtype=torch.long),
            "lengths": torch.tensor([5], dtype=torch.long),
            "position_ids": torch.arange(5, dtype=torch.long),
        },
        {
            "input_ids": torch.arange(7, dtype=torch.long),
            "hidden_states": torch.randn(7, num_target_layers * hidden_size),
            "verifier_last_hidden_states": torch.randn(7, hidden_size),
            "loss_mask": torch.ones(7, dtype=torch.long),
            "lengths": torch.tensor([7], dtype=torch.long),
            "position_ids": torch.arange(7, dtype=torch.long),
        },
    ]

    collated = collate_fn(batch)

    # document_ids: doc 0 has length 5, doc 1 truncated to length 6, rest is padding
    expected_document_ids = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long
    )
    assert torch.equal(collated["document_ids"], expected_document_ids)
    assert "lengths" not in collated

    for key in [
        "input_ids",
        "hidden_states",
        "verifier_last_hidden_states",
        "loss_mask",
        "position_ids",
    ]:
        assert collated[key].shape[0] == 1
        assert collated[key].shape[1] == max_len


def test_arrow_dataset_default_train_ratio_does_not_crash(tmp_path: Path):
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[1, 1, 1]],
            "seq_len": [3],
        }
    )
    ds.save_to_disk(str(tmp_path / "data"))
    (tmp_path / "data" / "hidden_states").mkdir()

    arrow_ds = ArrowDataset(
        max_len=128,
        datapath=str(tmp_path / "data"),
        on_missing="skip",
    )

    # Should not raise AttributeError
    assert arrow_ds._map_to_file_idx(0) == 0
    assert arrow_ds._map_to_file_idx(5) == 5


class _SequenceTransfer:
    """Minimal transfer fake which returns or raises queued generated results."""

    def __init__(self, generated_results):
        self.generated_results = list(generated_results)
        self.deleted: list[str] = []

    def setup(self):
        return None

    def get_cached(self, _file_idx):
        return None

    def get_generated(self, _hs_transfer_params):
        result = self.generated_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def delete(self, handle):
        self.deleted.append(handle)


def _make_generation_dataset(
    tmp_path: Path,
    transfer: _SequenceTransfer,
    **kwargs,
) -> ArrowDataset:
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[1, 1, 1]],
            "seq_len": [3],
        }
    )
    ds.save_to_disk(str(tmp_path / "data"))
    arrow_ds = ArrowDataset(
        max_len=8,
        datapath=str(tmp_path / "data"),
        transfer=transfer,  # type: ignore[arg-type]
        **kwargs,
    )
    arrow_ds.data.set_format(type="torch")
    # Bypass endpoint discovery/transfer setup: the fake transfer needs none
    # and the patched generate_hidden_states never touches the client.
    arrow_ds.router = EndpointRouter(
        ["http://fake:8000/v1"], client_factory=lambda _ep: object()
    )
    arrow_ds._endpoints_ready = True
    arrow_ds.model = "model"
    return arrow_ds


def _valid_generated_sample() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.ones(3, 2, 4, dtype=torch.bfloat16),
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
    }


def _next_generation_result(handles):
    handle = next(handles)
    return handle, {"handle": handle}


def test_arrow_dataset_retries_nonfinite_read_then_recovers(
    tmp_path, monkeypatch, caplog
):
    corrupt = _valid_generated_sample()
    corrupt["hidden_states"][1, 0, 0] = float("nan")
    transfer = _SequenceTransfer([corrupt, _valid_generated_sample()])
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        generation_validation_retries=1,
    )
    handles = iter(["bad-handle", "good-handle"])
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    with caplog.at_level(logging.WARNING, logger="speculators"):
        item = arrow_ds[0]

    assert isinstance(item, dict)
    assert "non-finite" in caplog.text
    assert item["hidden_states"].shape == (3, 4)
    assert torch.isfinite(item["hidden_states"]).all()
    assert transfer.deleted == ["bad-handle", "good-handle"]
    assert arrow_ds.generation_recovery.consecutive_failures == 0


def test_exhausted_generation_produces_locally_empty_zero_loss_batch(
    tmp_path, monkeypatch, caplog
):
    transfer = _SequenceTransfer(
        [ValueError("checksum mismatch"), ValueError("checksum mismatch")]
    )
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        generation_validation_retries=1,
        max_consecutive_generation_failures=10,
    )
    handles = iter(["bad-1", "bad-2"])
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    with caplog.at_level(logging.WARNING, logger="speculators"):
        failure = arrow_ds[0]

    assert "checksum mismatch" in caplog.text
    assert isinstance(failure, SampleUnavailable)
    assert not failure.fatal
    collated = CollateFn(
        max_len=8,
        hidden_size=4,
        num_target_layers=1,
        dtype=torch.bfloat16,
    )([failure])
    assert collated["error_records"] == 1
    metadata = collated[RECOVERY_METADATA_KEY]
    assert isinstance(metadata, RecoveryMetadata)
    assert metadata.locally_empty
    assert metadata.failure_count == 1
    assert not metadata.fatal
    assert not collated["loss_mask"].bool().any()
    assert torch.equal(collated["document_ids"], torch.full((1, 8), -1))
    assert collated["hidden_states"].shape == (1, 8, 4)
    assert collated["hidden_states"].dtype == torch.bfloat16


def test_consecutive_generation_failures_trip_circuit_breaker(
    tmp_path, monkeypatch, caplog
):
    transfer = _SequenceTransfer(
        [
            ValueError("bad read 1"),
            ValueError("bad read 2"),
            ValueError("bad read 3"),
            ValueError("bad read 4"),
        ]
    )
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        generation_validation_retries=1,
        max_consecutive_generation_failures=2,
    )
    handles = iter(["bad-1", "bad-2", "bad-3", "bad-4"])
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    with caplog.at_level(logging.WARNING, logger="speculators"):
        first_failure = arrow_ds[0]
        failure = arrow_ds[0]

    assert "consecutive failures=2/2" in caplog.text
    assert isinstance(first_failure, SampleUnavailable)
    assert not first_failure.fatal
    assert first_failure.consecutive_failures == 1
    assert isinstance(failure, SampleUnavailable)
    assert failure.fatal
    assert failure.consecutive_failures == 2
    collated = CollateFn(8, 4, num_target_layers=1)([failure])
    metadata = collated[RECOVERY_METADATA_KEY]
    assert isinstance(metadata, RecoveryMetadata)
    assert metadata.fatal
    assert "bad read 4" in metadata.error


def test_collator_keeps_valid_samples_when_one_generation_fails():
    valid = {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "hidden_states": torch.ones(2, 4, dtype=torch.bfloat16),
        "verifier_last_hidden_states": torch.ones(2, 4, dtype=torch.bfloat16),
        "loss_mask": torch.ones(2, dtype=torch.long),
        "lengths": torch.tensor([2], dtype=torch.long),
        "position_ids": torch.arange(2, dtype=torch.long),
    }
    failure = SampleUnavailable(
        "transient read",
        counts_as_failure=True,
        consecutive_failures=1,
    )

    collated = CollateFn(8, 4, num_target_layers=1)([None, failure, valid])

    assert collated["error_records"] == 2
    metadata = collated[RECOVERY_METADATA_KEY]
    assert isinstance(metadata, RecoveryMetadata)
    assert not metadata.locally_empty
    assert metadata.failure_count == 1
    assert collated["loss_mask"].sum() == 2
    assert torch.equal(collated["input_ids"][0, :2], torch.tensor([1, 2]))


def test_failover_rotates_to_healthy_endpoint(tmp_path, monkeypatch):
    transfer = _SequenceTransfer([_valid_generated_sample()])
    arrow_ds = _make_generation_dataset(tmp_path, transfer, max_retries=2)
    arrow_ds.router = EndpointRouter(
        ["http://bad:8000/v1", "http://good:8000/v1"],
        client_factory=_EndpointClient,
    )
    calls = []

    def fake_generate(client, _model, _item, *, timeout=None, max_retries=0):
        calls.append(client.name)
        if client.name == "http://bad:8000/v1":
            raise _connection_error(client.name)
        return _next_generation_result(iter(["good-handle"]))

    monkeypatch.setattr(data_module, "generate_hidden_states", fake_generate)

    item = arrow_ds[0]

    assert isinstance(item, dict)
    assert item["hidden_states"].shape == (3, 4)
    assert calls == ["http://bad:8000/v1", "http://good:8000/v1"]
    assert arrow_ds.router.stats("http://bad:8000/v1")["failures"] == 1
    assert arrow_ds.router.stats("http://good:8000/v1")["failures"] == 0
    # One connection failure is below the eviction threshold (2): the endpoint
    # stays in rotation and the sample never counts as a recovery failure.
    assert arrow_ds.router.is_healthy("http://bad:8000/v1")
    assert arrow_ds.generation_recovery.consecutive_failures == 0


def test_failover_evicts_endpoint_after_threshold_and_recovers(tmp_path, monkeypatch):
    transfer = _SequenceTransfer([_valid_generated_sample(), _valid_generated_sample()])
    arrow_ds = _make_generation_dataset(tmp_path, transfer, max_retries=2)
    arrow_ds.router = EndpointRouter(
        ["http://bad:8000/v1", "http://good:8000/v1"],
        client_factory=_EndpointClient,
        failure_threshold=2,
    )
    bad_calls = 0

    def fake_generate(client, _model, _item, *, timeout=None, max_retries=0):
        nonlocal bad_calls
        if client.name == "http://bad:8000/v1":
            bad_calls += 1
            raise _connection_error(client.name)
        return _next_generation_result(iter(["h1", "h2"]))

    monkeypatch.setattr(data_module, "generate_hidden_states", fake_generate)

    assert isinstance(arrow_ds[0], dict)
    assert bad_calls == 1  # RR alternates: one failure per sample on 'bad'
    assert arrow_ds.router.is_healthy("http://bad:8000/v1")

    assert isinstance(arrow_ds[0], dict)
    assert bad_calls == 2  # second failure -> threshold reached mid-sample
    assert not arrow_ds.router.is_healthy("http://bad:8000/v1")
    assert bad_calls == 2  # evicted endpoint gets no more traffic
    # third sample: served by the good endpoint alone
    assert arrow_ds.transfer.generated_results == []


def test_semantic_error_fails_fast_without_endpoint_eviction(tmp_path, monkeypatch):
    transfer = _SequenceTransfer([])
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        max_retries=2,
        generation_validation_retries=0,
    )
    arrow_ds.router = EndpointRouter(
        ["http://a:8000/v1", "http://b:8000/v1"],
        client_factory=_EndpointClient,
    )
    calls = []

    def fake_generate(client, _model, _item, *, timeout=None, max_retries=0):
        calls.append(client.name)
        raise _status_error(400)

    monkeypatch.setattr(data_module, "generate_hidden_states", fake_generate)

    result = arrow_ds[0]

    assert isinstance(result, SampleUnavailable)
    assert calls == ["http://a:8000/v1"]  # no cross-endpoint retry
    assert arrow_ds.router.is_healthy("http://a:8000/v1")
    assert arrow_ds.router.is_healthy("http://b:8000/v1")


def test_all_endpoints_failing_yields_sample_unavailable(tmp_path, monkeypatch):
    transfer = _SequenceTransfer([])
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        max_retries=0,
        generation_validation_retries=0,
    )
    arrow_ds.router = EndpointRouter(
        ["http://e1:8000/v1", "http://e2:8000/v1"],
        client_factory=_EndpointClient,
    )

    def fake_generate(client, _model, _item, *, timeout=None, max_retries=0):
        raise _connection_error(client.name)

    monkeypatch.setattr(data_module, "generate_hidden_states", fake_generate)

    result = arrow_ds[0]

    assert isinstance(result, SampleUnavailable)
    assert result.counts_as_failure
    assert arrow_ds.router.stats("http://e1:8000/v1")["failures"] == 1
    assert arrow_ds.router.stats("http://e2:8000/v1")["failures"] == 0
