from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    enable_sp: bool = False
    num_tokens: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_enforce_eager: bool = True
    real_bs: int = -1
    block_size: int = 256


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None, is_enforce_eager=None, real_bs=None, block_size=None,
                enable_sp=False, num_tokens=0):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        enable_sp=enable_sp,
        num_tokens=num_tokens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_enforce_eager=is_enforce_eager,
        real_bs=real_bs,
        block_size=block_size,
    )
    # print(f"_CONTEXT is set to {_CONTEXT}")


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
