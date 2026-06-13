#!/usr/bin/env python3
"""Hyperparameter sweep → constrained Pareto for the NPU-for-VLA design point.

Objective: maximize control loop Hz subject to power <= 30 W, area <= budget,
end-to-end step <= 100 ms (>= 10 Hz). Reports the binding constraint/roof so
the engineer sees WHY a point is limited before committing to RTL.
"""
import sys, os, itertools, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import roofline
from workload_vla import VLADims, build_vla

BASE = roofline.load_config(os.path.join(os.path.dirname(__file__),
                                         "configs", "baseline.json"))
AREA_BUDGET_MM2 = 60.0
POWER_BUDGET_W = 30.0
LATENCY_BUDGET_MS = 100.0   # >= 10 Hz

AXES = {
    "tensor_array_dim":  [128, 160, 192, 256],
    "tensor_clock_ghz":  [1.0, 1.2],
    "dram_bw_gbps":      [136, 204, 273, 410, 546],
    "scratchpad_kb":     [4096, 8192],
}


def run_point(overrides):
    c = copy.deepcopy(BASE)
    c.update(overrides)
    d = VLADims()
    ops, mult = build_vla(d, kv_quant_bits=c["kv_quant_bits"],
                          flow_steps=int(c["flow_expert_steps"]))
    r = roofline.simulate(ops, mult, c)
    return c, r


def main():
    rows = []
    keys = list(AXES)
    for combo in itertools.product(*[AXES[k] for k in keys]):
        ov = dict(zip(keys, combo))
        c, r = run_point(ov)
        feasible = (r["area_mm2"] <= AREA_BUDGET_MM2 and
                    r["avg_power_w"] <= POWER_BUDGET_W and
                    r["e2e_latency_ms"] <= LATENCY_BUDGET_MS)
        rows.append({**ov, "hz": r["control_hz"], "e2e_ms": r["e2e_latency_ms"],
                     "tok_s": r["decode_tok_s"], "area": r["area_mm2"],
                     "power": r["avg_power_w"], "feasible": feasible,
                     "energy_tok": r["energy_per_token_mj"]})

    feas = [x for x in rows if x["feasible"]]
    feas.sort(key=lambda x: -x["hz"])
    print(f"swept {len(rows)} points; {len(feas)} feasible "
          f"(area<={AREA_BUDGET_MM2}mm2, power<={POWER_BUDGET_W}W, "
          f">=10Hz)\n")
    hdr = (f"{'arr':>4} {'clk':>4} {'BW':>4} {'SRAM':>5} | {'Hz':>5} "
           f"{'e2e_ms':>7} {'tok/s':>6} {'mm2':>5} {'W':>5}")
    print(hdr); print("-" * len(hdr))
    show = feas[:12] if feas else sorted(rows, key=lambda x: x["e2e_ms"])[:12]
    for x in show:
        print(f"{x['tensor_array_dim']:>4} {x['tensor_clock_ghz']:>4} "
              f"{x['dram_bw_gbps']:>4} {x['scratchpad_kb']//1024:>4}M | "
              f"{x['hz']:>5.1f} {x['e2e_ms']:>7.1f} {x['tok_s']:>6.1f} "
              f"{x['area']:>5.1f} {x['power']:>5.1f}")
    if not feas:
        print("\n(no point meets >=10Hz; showing fastest — decode/prefill "
              "still bind. Next levers: fewer decode tokens / flow-only "
              "action head, decode_batch>1 sampling, higher BW.)")
    else:
        b = feas[0]
        print(f"\nbest: {b['tensor_array_dim']}x{b['tensor_array_dim']}@"
              f"{b['tensor_clock_ghz']}GHz, {b['dram_bw_gbps']}GB/s, "
              f"{b['scratchpad_kb']//1024}MB -> {b['hz']:.1f}Hz, "
              f"{b['area']:.1f}mm2, {b['power']:.1f}W")


if __name__ == "__main__":
    main()
