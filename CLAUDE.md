# NPU for end-side VLA — project instructions

A from-scratch NPU (neural processing unit) + toolchain (compiler / quantizer)
to run **VLA (vision-language-action)** large models on-device for real-time
robot control. Same methodology as the companion `h264_by_claude_code` /
`jpeg_by_claude_code` ASIC projects: **software model first, verify, iterate,
then RTL.**

## Frozen design point (W0)

- **Workload**: model-agnostic — the NPU + simulator consume an operator graph.
  Representative graph is a ~3B-class VLA: ViT vision encoder + transformer LLM
  backbone (prefill + autoregressive decode) + action head.
- **Deployment**: robot main-control SoC, **~10–30 W**, ~7nm-class, 0.7 V.
- **Quantization**: **W8A8 (full INT8)**. PE = INT8×INT8 → INT32 accumulate.
- **Central tension**: one NPU must serve compute-bound INT8 GEMM (ViT/prefill)
  AND bandwidth-bound INT8 GEMV (decode + KV-cache). Resolving this is the job.

## Methodology (build order)

1. **Simulator first**, layered for iteration speed:
   - **roofline** (operator-level, analytical compute/BW roofs + on-chip reuse +
     energy model) → second-scale hyperparameter sweeps. **Build this first.**
   - **cycle-approximate** (event-driven PE/SRAM/DMA/NoC) → validate roofline's
     optimum and catch the corners it misses.
   - **cycle-accurate** → final confirmation just before RTL.
2. Sweep the hardware hyperparameters on the simulator until efficiency is high
   enough (tokens/s/W, MAC & decode-BW utilization, energy/token, control-step
   latency), pick a design point.
3. Toolchain: quantizer (W8A8 PTQ) + compiler (graph → tiled NPU schedule) that
   targets the same operator IR the simulator consumes — so sim and silicon run
   the same schedule.
4. **Then** RTL (sv2v + yosys + ASAP7, reusing the JPEG/H.264 flow), differential
   against the cycle-accurate model.

## Conventions

- Simulator in Python under `sim/` (operator IR, hardware config, models, sweep).
- Every architectural knob is a numeric hyperparameter in a config the simulator
  sweeps — no hard-coded magic numbers in the model.
- Per-phase commits with detailed messages; keep `docs/roadmap.md` current.
- Validate the roofline model against at least one known datapoint before trusting
  its sweep conclusions (no silent "the model says so").
