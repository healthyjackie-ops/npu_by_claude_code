"""Representative ~3B VLA workload builder (W1).

Parametric so the sim can sweep model/runtime dims too. Returns a list of Op
records grouped by stage. Stage execution multipliers (how many times each
stage runs per inference / control step) are returned separately so decode
(per-token) and the action expert (per-step) scale correctly:

  vit, connector, prefill : x1 per control step
  decode                  : x n_decode_tokens
  action                  : x flow_expert_steps

Dims follow docs/arch_spec.md (verified: decode weight traffic ~2.6 GB/token).
"""
from __future__ import annotations
from dataclasses import dataclass
from ir import (Op, GEMM, GEMV, ATTN_SCORE, ATTN_AV, SOFTMAX, NORM, ACT, CONV,
                OUTPUT_STATIONARY, STREAM_ONCE)


@dataclass
class VLADims:
    # ViT vision encoder
    vit_layers: int = 24
    vit_hidden: int = 1152
    vit_patches: int = 576
    vit_heads: int = 16
    vit_mlp: int = 4304
    patch_k: int = 588              # 3*14*14
    # LLM backbone
    llm_layers: int = 28
    llm_hidden: int = 2560
    q_heads: int = 20
    kv_heads: int = 4
    head_dim: int = 128
    llm_mlp: int = 8192
    vocab: int = 152064
    # connector
    conn_out: int = 2560
    # action (flow-matching DiT expert)
    act_tokens: int = 64
    act_hidden: int = 1024
    act_blocks: int = 18
    act_dof: int = 32
    # runtime
    prompt_len: int = 256
    seq_len: int = 832              # image+text context for decode KV reads
    n_decode_tokens: int = 8        # action tokens generated autoregressively


def build_vla(d: VLADims, kv_quant_bits: int = 8, flow_steps: int = 8):
    """Return (ops, stage_mult). ops grouped by stage; multipliers per stage."""
    ops = []
    H = d.vit_hidden
    hd = H // d.vit_heads

    # ---- ViT encoder (per layer, x vit_layers) ----
    ops += [
        Op("vit.patch_embed", "vit", CONV, M=d.vit_patches, K=d.patch_k,
           N=H, count=1),
        Op("vit.qkv", "vit", GEMM, M=d.vit_patches, K=H, N=3 * H,
           count=d.vit_layers),
        Op("vit.qk", "vit", ATTN_SCORE, M=d.vit_patches, K=hd,
           N=d.vit_patches, heads=d.vit_heads, count=d.vit_layers,
           has_weights=False),
        Op("vit.softmax", "vit", SOFTMAX, M=d.vit_patches, N=d.vit_patches,
           heads=d.vit_heads, count=d.vit_layers, has_weights=False),
        Op("vit.av", "vit", ATTN_AV, M=d.vit_patches, K=d.vit_patches,
           N=hd, heads=d.vit_heads, count=d.vit_layers, has_weights=False),
        Op("vit.o", "vit", GEMM, M=d.vit_patches, K=H, N=H, count=d.vit_layers),
        Op("vit.fc1", "vit", GEMM, M=d.vit_patches, K=H, N=d.vit_mlp,
           count=d.vit_layers),
        Op("vit.gelu", "vit", ACT, M=d.vit_patches, N=d.vit_mlp,
           count=d.vit_layers, has_weights=False),
        Op("vit.fc2", "vit", GEMM, M=d.vit_patches, K=d.vit_mlp, N=H,
           count=d.vit_layers),
        Op("vit.norm", "vit", NORM, M=d.vit_patches, N=H,
           count=2 * d.vit_layers, has_weights=False),
    ]

    # ---- connector (x1) ----
    ops += [Op("conn.proj", "connector", GEMM, M=d.vit_patches, K=H,
               N=d.conn_out, count=2)]

    # ---- LLM prefill (per layer, x llm_layers) ----
    Hl = d.llm_hidden
    qn = d.q_heads * d.head_dim
    kvn = d.kv_heads * d.head_dim
    P = d.prompt_len
    ops += [
        Op("pf.q", "prefill", GEMM, M=P, K=Hl, N=qn, count=d.llm_layers),
        Op("pf.kv", "prefill", GEMM, M=P, K=Hl, N=2 * kvn, count=d.llm_layers),
        Op("pf.qk", "prefill", ATTN_SCORE, M=P, K=d.head_dim, N=P,
           heads=d.q_heads, count=d.llm_layers, has_weights=False),
        Op("pf.softmax", "prefill", SOFTMAX, M=P, N=P, heads=d.q_heads,
           count=d.llm_layers, has_weights=False),
        Op("pf.av", "prefill", ATTN_AV, M=P, K=P, N=d.head_dim,
           heads=d.q_heads, count=d.llm_layers, has_weights=False),
        Op("pf.o", "prefill", GEMM, M=P, K=qn, N=Hl, count=d.llm_layers),
        Op("pf.gate", "prefill", GEMM, M=P, K=Hl, N=d.llm_mlp,
           count=d.llm_layers),
        Op("pf.up", "prefill", GEMM, M=P, K=Hl, N=d.llm_mlp,
           count=d.llm_layers),
        Op("pf.silu", "prefill", ACT, M=P, N=d.llm_mlp, count=d.llm_layers,
           has_weights=False),
        Op("pf.down", "prefill", GEMM, M=P, K=d.llm_mlp, N=Hl,
           count=d.llm_layers),
        Op("pf.norm", "prefill", NORM, M=P, N=Hl, count=2 * d.llm_layers,
           has_weights=False),
    ]

    # ---- LLM decode (per layer, x llm_layers = ONE token; stage x n_decode) ----
    kvb = (kv_quant_bits + 7) // 8
    kv_read = 2 * d.seq_len * d.kv_heads * d.head_dim * kvb  # K+V cache, per layer per head-group
    ops += [
        Op("dec.q", "decode", GEMV, M=1, K=Hl, N=qn, count=d.llm_layers,
           reuse_class=STREAM_ONCE),
        Op("dec.kv", "decode", GEMV, M=1, K=Hl, N=2 * kvn, count=d.llm_layers,
           reuse_class=STREAM_ONCE),
        Op("dec.qk", "decode", ATTN_SCORE, M=1, K=d.head_dim, N=d.seq_len,
           heads=d.q_heads, count=d.llm_layers, has_weights=False,
           kv_read_bytes=kv_read // 2, reuse_class=STREAM_ONCE),
        Op("dec.softmax", "decode", SOFTMAX, M=1, N=d.seq_len, heads=d.q_heads,
           count=d.llm_layers, has_weights=False),
        Op("dec.av", "decode", ATTN_AV, M=1, K=d.seq_len, N=d.head_dim,
           heads=d.q_heads, count=d.llm_layers, has_weights=False,
           kv_read_bytes=kv_read // 2, reuse_class=STREAM_ONCE),
        Op("dec.o", "decode", GEMV, M=1, K=qn, N=Hl, count=d.llm_layers,
           reuse_class=STREAM_ONCE),
        Op("dec.gate", "decode", GEMV, M=1, K=Hl, N=d.llm_mlp,
           count=d.llm_layers, reuse_class=STREAM_ONCE),
        Op("dec.up", "decode", GEMV, M=1, K=Hl, N=d.llm_mlp,
           count=d.llm_layers, reuse_class=STREAM_ONCE),
        Op("dec.silu", "decode", ACT, M=1, N=d.llm_mlp, count=d.llm_layers,
           has_weights=False),
        Op("dec.down", "decode", GEMV, M=1, K=d.llm_mlp, N=Hl,
           count=d.llm_layers, reuse_class=STREAM_ONCE),
        Op("dec.norm", "decode", NORM, M=1, N=Hl, count=2 * d.llm_layers,
           has_weights=False),
        Op("dec.lm_head", "decode", GEMV, M=1, K=Hl, N=d.vocab, count=1,
           reuse_class=STREAM_ONCE),
    ]

    # ---- action expert (flow-matching DiT, per step; stage x flow_steps) ----
    A = d.act_hidden
    Na = d.act_tokens
    ah = A // 8  # 8 heads
    ops += [
        Op("act.in", "action", GEMM, M=Na, K=d.conn_out, N=A, count=1,
           resident=True),
        Op("act.qkv", "action", GEMM, M=Na, K=A, N=3 * A, count=d.act_blocks,
           resident=True),
        Op("act.qk", "action", ATTN_SCORE, M=Na, K=ah, N=Na, heads=8,
           count=d.act_blocks, has_weights=False),
        Op("act.av", "action", ATTN_AV, M=Na, K=Na, N=ah, heads=8,
           count=d.act_blocks, has_weights=False),
        Op("act.o", "action", GEMM, M=Na, K=A, N=A, count=d.act_blocks,
           resident=True),
        Op("act.mlp1", "action", GEMM, M=Na, K=A, N=4 * A, count=d.act_blocks,
           resident=True),
        Op("act.mlp2", "action", GEMM, M=Na, K=4 * A, N=A, count=d.act_blocks,
           resident=True),
        Op("act.timestep", "action", ACT, M=Na, N=A, count=d.act_blocks,
           has_weights=False),
        Op("act.out", "action", GEMM, M=Na, K=A, N=d.act_dof, count=1,
           resident=True),
    ]

    stage_mult = {"vit": 1, "connector": 1, "prefill": 1,
                  "decode": d.n_decode_tokens, "action": flow_steps}
    return ops, stage_mult
