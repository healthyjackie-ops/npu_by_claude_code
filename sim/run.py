#!/usr/bin/env python3
"""Run the roofline sim on a config + the representative VLA workload."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import roofline
from workload_vla import VLADims, build_vla


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "configs", "baseline.json")
    c = roofline.load_config(cfg_path)
    d = VLADims()
    ops, mult = build_vla(d, kv_quant_bits=c["kv_quant_bits"],
                          flow_steps=int(c["flow_expert_steps"]))
    r = roofline.simulate(ops, mult, c)

    print(f"=== NPU-for-VLA roofline @ {os.path.basename(cfg_path)} ===")
    print(f"config: array {c['tensor_array_dim']}x{c['tensor_array_dim']} "
          f"@{c['tensor_clock_ghz']}GHz, vec {c['vector_lanes']}x{c['vector_dot_len']}, "
          f"SRAM {c['scratchpad_kb']/1024:.0f}MB, DRAM {c['dram_bw_gbps']:.0f}GB/s, "
          f"decode_batch {c['decode_batch']}, KV{c['kv_quant_bits']}b")
    print(f"\nArea  : {r['area_mm2']:.2f} mm2  "
          f"(tensor {r['area_breakdown_mm2']['tensor']:.2f} / "
          f"vector {r['area_breakdown_mm2']['vector']:.2f} / "
          f"sram {r['area_breakdown_mm2']['sram']:.2f})")
    print(f"Peak  : {r['peak_int8_tops']:.1f} INT8 TOPS")
    print(f"\n--- per-stage latency (one execution) ---")
    for s, t in r['stage_time_ms'].items():
        print(f"  {s:10} {t:8.3f} ms  x{mult.get(s,1)}")
    print(f"\nPrefill+ViT path : {r['stage_time_ms'].get('vit',0)+r['stage_time_ms'].get('connector',0)+r['stage_time_ms'].get('prefill',0):.2f} ms (ingest a frame+prompt)")
    print(f"Decode           : {r['decode_ms_per_token']:.2f} ms/token  "
          f"= {r['decode_tok_s']:.1f} tok/s")
    print(f"End-to-end step  : {r['e2e_latency_ms']:.2f} ms  "
          f"= {r['control_hz']:.1f} Hz control loop")
    print(f"Energy/token     : {r['energy_per_token_mj']:.2f} mJ  "
          f"(DRAM {r['decode_energy_breakdown_mj']['dram']:.2f} / "
          f"SRAM {r['decode_energy_breakdown_mj']['sram']:.2f} / "
          f"MAC {r['decode_energy_breakdown_mj']['mac']:.2f})")
    print(f"Avg power        : {r['avg_power_w']:.2f} W  (envelope 10-30 W)")
    tot = sum(r['bound_hist_ms'].values())
    print(f"\n--- where time goes (binding roof, % of summed op-time) ---")
    for k, v in sorted(r['bound_hist_ms'].items(), key=lambda x: -x[1]):
        print(f"  {k:8} {100*v/tot:5.1f}%")


if __name__ == "__main__":
    main()
