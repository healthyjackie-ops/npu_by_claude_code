# NPU-for-VLA architecture spec (W0, frozen)

Decided by a 3-way microarchitecture design panel (systolic / vector / hybrid)
scored on a VLA-fit rubric. **HV-1+ hybrid won decisively (41 vs 33 vs 32).**

## The decisive insight

Roofline math on the representative 3B VLA is **resource-separating, not
resource-competing**:

- **Decode** streams ~2.6 GB of INT8 weights + KV per token → at ~136 GB/s
  LPDDR5X the LLM tops out at ~40–53 tok/s and needs only **~136 GMAC/s** of
  compute to saturate the bus. Bandwidth-bound.
- **Prefill** (256 tok) ≈ 564 GMAC and **ViT** (576 patches) ≈ 210 GMAC, both
  compute-bound and latency-critical for the 10–50 Hz loop.

The compute needed to keep decode fed (~136 GMAC/s) is ~0.2 % of a 256×256
array's 65 TMAC/s. So the two regimes do **not** fight over silicon: **size the
tensor core to the prefill roofline, spend the power/pin budget on DRAM
bandwidth for decode, bridge with near-free vector lanes.** A 256×256 systolic
array is dark silicon at batch-1 decode (1/256 row utilization) and its leakage
hurts the 10–30 W envelope — rejected.

## HV-1+ architecture

- **Tensor core**: moderate INT8 systolic array (default 128×128, sweep 64–192),
  output-stationary, INT8×INT8→INT32. Serves ViT + prefill + action-head GEMM.
  Foldable into groups (grafted from the systolic candidate) so it can join
  decode when `decode_batch > 1`.
- **Vector path**: many INT8 MAC lanes (default 512, dot-len 4) over the
  scratchpad — serves decode GEMV and the SFU (softmax/RMSNorm/GELU/SiLU/RoPE).
  ~16 % of tensor-core area at the 128² baseline (512×4 lanes), and over-serves the decode compute roof by orders of magnitude.
- **Unified banked scratchpad** (default 4 MB, 24 banks): prefill weight/act
  tiles + ViT activations resident; KV-cache window + hot resident weights for
  decode. `sram_partition_decode_frac` splits it.
- **Programmable DMA + KV engine** (grafted from the vector candidate): double-
  buffered weight/KV streaming, `stream_tile_kb` granularity, `dma_outstanding`
  to hide LPDDR latency.
- **LPDDR5X** shared SoC DRAM (default ~136 GB/s): the decode bottleneck;
  first-class swept knob.

## Sweepable hyperparameters (22) — see `sim/configs/baseline.json`

tensor_array_dim, tensor_clock_ghz, vector_lanes, vector_dot_len,
vector_clock_ghz, tensor_fold_groups, scratchpad_kb, scratchpad_banks,
sram_partition_decode_frac, dram_bw_gbps, dram_bw_util_decode, stream_tile_kb,
tile_buf, dma_outstanding, decode_batch, kv_quant_bits, kv_groups_gqa,
flow_expert_steps, util_gemm, sfu_ratio, accum_width_bits, resident_weight_frac.

## Reconciled 7nm / 0.7 V coefficients

| coeff | value | note |
|---|---|---|
| MAC energy | 0.18 pJ/INT8-MAC | |
| SRAM read | 0.35 pJ/byte | on-chip |
| SRAM write | 0.42 pJ/byte | |
| DRAM | 8 pJ/byte | LPDDR5X |
| per-MAC area | 430 µm² | incl. PE reg/skew/control overhead |
| SRAM area | 410 µm²/KB | |
| clock | 1 GHz typical | |

## Representative ~3B VLA workload — see `sim/src/workload_vla.py`

ViT (24 L, hidden 1152, 576 patches, 16 heads) → connector 1152→2560 → LLM
(28 L, hidden 2560, 20 q-heads / 4 kv-heads GQA, MLP 8192, vocab 152064)
prefill (256 prompt tok) + decode (per token) → flow-matching action expert
(64 action tok, hidden 1024, ~18 DiT blocks, 8 steps). Decode weight traffic
≈ 2.6 GB/token (28×78.6 M + 389 M lm_head) — verified.

## Efficiency metrics the sweep optimizes

prefill+ViT sustained TOPS/W · tensor MAC utilization · decode DRAM-BW
utilization · decode tok/s · energy/token (mJ, DRAM/SRAM/MAC breakdown) ·
end-to-end control-step latency vs 20–100 ms · control Hz · area efficiency
(TOPS/mm², tok/s/mm²) · avg+peak power vs 10–30 W · binding-roof histogram.

## Methodology

roofline sim (this) → cycle-approximate → cycle-accurate → RTL (sv2v + yosys +
ASAP7, reusing the JPEG/H.264 flow). Validate the roofline against a known
datapoint before trusting sweep conclusions.
