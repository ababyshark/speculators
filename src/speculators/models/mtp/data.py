from speculators.train.data import BatchType

__all__ = ["shift_batch_mtp"]


def shift_batch_mtp(batch: BatchType) -> BatchType:
    """Rename verifier_last_hidden_states to hidden_states for MTP.

    No token-level shifting — the MTP forward pass handles alignment
    internally via per-step offset slicing of input_ids.
    """
    return {
        **({"document_ids": batch["document_ids"]} if "document_ids" in batch else {}),
        "input_ids": batch["input_ids"],
        "hidden_states": batch["verifier_last_hidden_states"],
        "loss_mask": batch["loss_mask"],
        **({"lengths": batch["lengths"]} if "lengths" in batch else {}),
        "position_ids": batch["position_ids"],
    }
