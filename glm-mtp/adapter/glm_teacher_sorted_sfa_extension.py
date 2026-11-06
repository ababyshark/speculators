"""Opt-in logical ordering for GLM teacher sparse attention (eager mode)."""

from glm_sfa_determinism import sort_sparse_indices
from glm_teacher_worker_extension import GLMTeacherDeterminism
from vllm.logger import init_logger
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def install_sorted_sparse_attention():
    original = AscendSFAImpl._execute_sparse_flash_attention_process

    def ordered(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        return original(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            sort_sparse_indices(topk_indices),
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
        )

    AscendSFAImpl._execute_sparse_flash_attention_process = ordered
    init_logger(__name__).info(
        "GLM teacher sparse indices sorted by logical token position"
    )


install_sorted_sparse_attention()


class GLMTeacherSortedSFA(GLMTeacherDeterminism):
    pass
