# npusim — NPU-for-VLA roofline simulator

Executable performance simulator for the HV-1+ NPU (see `../docs/arch_spec.md`).
Operator-level roofline model: per op `time = max(compute, DRAM-BW, on-chip-BW)`,
records the binding roof, plus an energy model and per-stage / end-to-end
aggregation. Config-driven — every hardware knob is in `configs/baseline.json`.

## Run

    python3 sim/npusim.py run                 # baseline
    python3 sim/npusim.py run -D tensor_array_dim=192 -D dram_bw_gbps=546 --ndec 8
    python3 sim/npusim.py run --json          # machine-readable
    python3 sim/npusim.py sweep               # constrained Pareto
    python3 sim/npusim.py report              # baseline + sensitivities
    python3 sim/npusim.py selftest            # validate model (exit!=0 on fail)

or `make run | sweep | report | test`.

## Knobs

- HW: `-D key=value` for any key in `configs/baseline.json` (array dim, clock,
  vector lanes, scratchpad, DRAM BW, decode_batch, kv_quant_bits, coefficients…).
- Workload: `--ndec` (decode tokens), `--seq` (KV seq len), `--prompt`, `--flow`.

## Layout

    src/ir.py            operator IR (Op records, derived MAC/byte quantities)
    src/workload_vla.py  parametric ~3B VLA op-graph builder
    src/roofline.py      roofline + energy model + aggregation
    npusim.py            CLI (run / sweep / report / selftest)
    configs/baseline.json  HV-1+ baseline hyperparameters + 7nm coefficients

`selftest` checks the model against the analytic decode DRAM roof (≈20 ms/token)
and monotonicity/area invariants — run it before trusting any sweep.
