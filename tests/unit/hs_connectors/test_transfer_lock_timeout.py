"""R5: hidden-state lock timeout defaults must be 1800s, not 10s.

The backend-args auto-registration mechanism makes every
``getattr(args, "hidden_states_lock_timeout", <fallback>)`` fallback dead —
the attribute always exists with the CLI-registered default — so the
registration default is the only effective knob. A 10s default caused 771
lock timeouts in a 256-concurrent cold-start storm (4 workers x 4 prefetch
x 16 ranks); 1800s reduced them to zero.
"""

import argparse

from hs_connectors.transfer import FileBackend, FileTransfer, FP8Backend, FP8Transfer


def test_file_transfer_default_lock_timeout_is_1800(tmp_path):
    assert FileTransfer(tmp_path).lock_timeout == 1800.0


def test_fp8_transfer_default_lock_timeout_is_1800(tmp_path):
    assert FP8Transfer(tmp_path).lock_timeout == 1800.0


def test_cli_registration_default_is_1800():
    parser = argparse.ArgumentParser()
    FileBackend.add_train_args(parser)
    args = parser.parse_args([])
    assert args.hidden_states_lock_timeout == 1800.0


def test_fp8_cli_registration_default_is_1800():
    parser = argparse.ArgumentParser()
    FP8Backend.add_train_args(parser)
    args = parser.parse_args([])
    assert args.hidden_states_lock_timeout == 1800.0


def test_from_train_args_resolves_1800_without_explicit_flag(tmp_path):
    parser = argparse.ArgumentParser()
    FileBackend.add_train_args(parser)
    args = parser.parse_args([])
    transfer = FileBackend.from_train_args(args, str(tmp_path))
    assert transfer.lock_timeout == 1800.0


def test_explicit_flag_still_honored(tmp_path):
    parser = argparse.ArgumentParser()
    FileBackend.add_train_args(parser)
    args = parser.parse_args(["--hidden-states-lock-timeout", "3600"])
    transfer = FileBackend.from_train_args(args, str(tmp_path))
    assert transfer.lock_timeout == 3600.0
