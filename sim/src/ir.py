"""Operator IR for the NPU-for-VLA performance simulator (W1).

One Op record per operator instance, fed from the workload builder. The
roofline model reads these fields; nothing here is hardware-specific.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# op_type taxonomy
GEMM = "gemm"            # M>1 dense matmul (ViT / prefill / action head)
GEMV = "gemv"            # M=1 matmul (decode projections), weight stream-once
ATTN_SCORE = "attn_score"   # Q·K^T, activation x activation
ATTN_AV = "attn_av"         # scores·V
SOFTMAX = "softmax"
NORM = "norm"            # RMSNorm / LayerNorm
ACT = "act"              # GELU / SiLU / elementwise modulation
CONV = "conv"            # patch embed (lowered to GEMM)

SFU_TYPES = {SOFTMAX, NORM, ACT}

# reuse classes (drive the SRAM-byte accounting in the energy model)
WEIGHT_STATIONARY = "weight_stationary"
OUTPUT_STATIONARY = "output_stationary"
STREAM_ONCE = "stream_once"      # decode: each weight byte read once per token


@dataclass
class Op:
    op_id: str
    stage: str                  # vit | connector | prefill | decode | action
    op_type: str
    M: int = 1
    N: int = 1
    K: int = 1
    heads: int = 1              # for attention ops (per-head MxN over K)
    count: int = 1              # instances per ONE execution of the stage
    reuse_class: str = OUTPUT_STATIONARY
    resident: bool = False      # weights pinned on-chip across calls
    has_weights: bool = True    # attention score/av ops are act x act
    kv_read_bytes: int = 0      # decode attention KV-cache reads (per instance)
    dtype_bytes: int = 1        # INT8

    # ---- derived quantities (per single instance) ----
    @property
    def macs(self) -> int:
        if self.op_type in (ATTN_SCORE, ATTN_AV):
            return self.M * self.N * self.K * self.heads
        if self.op_type in SFU_TYPES:
            return 0
        return self.M * self.N * self.K

    @property
    def weight_bytes(self) -> int:
        if not self.has_weights or self.op_type in SFU_TYPES:
            return 0
        return self.K * self.N * self.dtype_bytes

    @property
    def act_in_bytes(self) -> int:
        return self.M * self.K * self.dtype_bytes * max(1, self.heads if
            self.op_type in (ATTN_SCORE, ATTN_AV) else 1)

    @property
    def act_out_bytes(self) -> int:
        return self.M * self.N * self.dtype_bytes * max(1, self.heads if
            self.op_type in (ATTN_SCORE, ATTN_AV) else 1)

    @property
    def is_sfu(self) -> bool:
        return self.op_type in SFU_TYPES

    @property
    def is_tensor(self) -> bool:
        # large-M matmul runs on the tensor core; attention scores too (GEMM-ish)
        return self.op_type in (GEMM, CONV, ATTN_SCORE, ATTN_AV)
