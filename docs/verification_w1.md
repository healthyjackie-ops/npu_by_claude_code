# W1.1 — adversarial verification of the roofline simulator

A 4-dimension review panel (roofline / workload / energy-area / CLI) with each
finding independently re-verified against the code produced **22 confirmed**
issues. Outcome:

## Fixed (model + JS port, re-validated, Python==JS exact)

- **Prefill M omitted the image tokens** (#6): prefill now ingests
  `vit_patches + prompt_len` (832, matching the decode KV window). Headline
  effect: the recommended point is 7.9 Hz, not the earlier optimistic 11.2 Hz.
- **Peak power never reported** (#10): added `peak_power_w` (worst sequential
  stage) — the 30 W envelope check should use peak, not the BW-idle-diluted avg.
- **On-chip BW roof ignored weight traffic** (#2): staged/reused tensor-GEMM
  weights now count toward the scratchpad-bank roof (streamed decode weights
  ride the DMA path, not the banks — deliberately excluded from the bank roof
  but charged staging energy).
- **decode_batch was modeled wrong** (#5): it divided weight DRAM by B (free
  lunch). Now B raises decode M (compute + KV scale), weights are fetched once,
  per-token metrics divide by B — correct amortization that eventually becomes
  compute-bound.
- **SRAM-staging energy** (#3,#11,#15): reads = act_in + staged weights;
  writes = act_out + DRAM→SRAM weight fill. Removed the read-of-own-output;
  streamed decode weights now incur staging energy (+~9% energy/token).
- **Leakage** (#9): folded into e2e_energy so `avg_power == e2e_energy/e2e_time`.
- **DRAM-BW utilisation split** (#1,#13): `dram_bw_util_compute` vs `_decode`,
  selected per reuse class.
- **SFU binding tag** (#4): SFU ops keep their tag in the roof histogram.
- **selftest hardened** (#16,#17,#18,#22): honors `--config/-D`; analytic decode
  roof summed from the actual op list (×count) not a magic constant; added a
  decode-not-on-systolic-core invariant and a peak≥avg invariant; tolerance 5%.
- **CLI robustness** (#19,#20,#21): `-D` validates finite/positive/int-typed;
  clean errors on missing/bad config; explicit "NO FEASIBLE POINTS" banner.

## Accepted as first-order simplifications (documented, revisit in W2)

- **Action-expert "resident" weights** (#8,#12): marked free though ~113 MB > the
  4 MB scratchpad; they would actually stream. Action stage is ~7% of the step,
  so secondary — W2 cycle-approx will model the on-chip residency limit.
- **Connector count=2** (#7): a 2-layer projector approximated with one set of
  dims; minor.
- **DMA double-buffer sufficiency** (tile_buf/dma_outstanding) not yet derating
  effective BW — that's exactly what W2's event-driven model adds.

The roofline stays a first-order analytic bound by design; W2 (cycle-approximate)
validates its corners.
