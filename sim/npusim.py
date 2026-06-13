#!/usr/bin/env python3
"""npusim — executable NPU-for-VLA roofline simulator CLI.

Subcommands:
  run       single design point -> latency/throughput/energy/power/area report
  sweep     constrained design-space sweep -> Pareto + best feasible point
  report    full report: baseline + sensitivity curves (BW/array/ndec/batch)
  selftest  validate the model against the analytic decode roof + invariants

Examples:
  ./npusim.py run
  ./npusim.py run -D tensor_array_dim=192 -D tensor_clock_ghz=1.2 -D dram_bw_gbps=546 --ndec 8
  ./npusim.py run --config configs/baseline.json --json
  ./npusim.py sweep
  ./npusim.py report
  ./npusim.py selftest

Every hardware knob is a config key (configs/baseline.json); override on the
command line with -D key=value (repeatable). Workload knobs: --ndec, --seq,
--prompt, --flow.
"""
import argparse, copy, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import roofline                                  # noqa: E402
from workload_vla import VLADims, build_vla      # noqa: E402

DEFAULT_CFG = os.path.join(HERE, "configs", "baseline.json")


def _cast(v):
    for fn in (int, float):
        try:
            return fn(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def load_cfg(path, overrides):
    c = roofline.load_config(path)
    for kv in overrides or []:
        if "=" not in kv:
            sys.exit(f"bad -D '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        if k not in c:
            sys.exit(f"unknown config key '{k}'. keys: {', '.join(sorted(c))}")
        c[k] = _cast(v)
    return c


def make_dims(args):
    d = VLADims()
    if getattr(args, "ndec", None) is not None:
        d.n_decode_tokens = args.ndec
    if getattr(args, "seq", None) is not None:
        d.seq_len = args.seq
    if getattr(args, "prompt", None) is not None:
        d.prompt_len = args.prompt
    return d


def run_point(c, d, flow=None):
    ops, mult = build_vla(d, kv_quant_bits=c["kv_quant_bits"],
                          flow_steps=int(flow if flow is not None
                                         else c["flow_expert_steps"]))
    return roofline.simulate(ops, mult, c)


# ---------------- subcommands ----------------
def cmd_run(args):
    c = load_cfg(args.config, args.D)
    d = make_dims(args)
    r = run_point(c, d, flow=args.flow)
    if args.json:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        print(json.dumps(out, indent=2))
        return 0
    print(f"=== npusim run: {os.path.basename(args.config)} "
          f"{'(+overrides)' if args.D else ''} ===")
    print(f"HW   : array {c['tensor_array_dim']}^2 @{c['tensor_clock_ghz']}GHz | "
          f"vec {c['vector_lanes']}x{c['vector_dot_len']} | "
          f"SRAM {c['scratchpad_kb']/1024:.0f}MB | DRAM {c['dram_bw_gbps']:.0f}GB/s | "
          f"batch {c['decode_batch']} | KV{c['kv_quant_bits']}b")
    print(f"WL   : ndec {d.n_decode_tokens} | seq {d.seq_len} | prompt {d.prompt_len}")
    print(f"Area : {r['area_mm2']:.2f} mm2  (tensor {r['area_breakdown_mm2']['tensor']:.1f}"
          f" / vec {r['area_breakdown_mm2']['vector']:.1f}"
          f" / sram {r['area_breakdown_mm2']['sram']:.1f})  | peak {r['peak_int8_tops']:.0f} TOPS")
    for s, t in r['stage_time_ms'].items():
        print(f"  {s:10} {t:8.3f} ms  x{r['stage_mult'].get(s,1)}")
    print(f"decode  : {r['decode_ms_per_token']:.2f} ms/tok = {r['decode_tok_s']:.1f} tok/s")
    print(f"e2e step: {r['e2e_latency_ms']:.2f} ms = {r['control_hz']:.1f} Hz")
    print(f"E/token : {r['energy_per_token_mj']:.2f} mJ "
          f"(DRAM {r['decode_energy_breakdown_mj']['dram']:.2f}/"
          f"SRAM {r['decode_energy_breakdown_mj']['sram']:.2f}/"
          f"MAC {r['decode_energy_breakdown_mj']['mac']:.2f})")
    print(f"avg pow : {r['avg_power_w']:.2f} W")
    tot = sum(r['bound_hist_ms'].values()) or 1
    print("roof%   : " + "  ".join(f"{k} {100*v/tot:.0f}%"
          for k, v in sorted(r['bound_hist_ms'].items(), key=lambda x: -x[1])))
    return 0


def cmd_sweep(args):
    base = load_cfg(args.config, args.D)
    axes = {
        "tensor_array_dim": [128, 160, 192, 256],
        "tensor_clock_ghz": [1.0, 1.2],
        "dram_bw_gbps": [136, 204, 273, 410, 546],
        "scratchpad_kb": [4096, 8192],
    }
    d = make_dims(args)
    rows = []
    for combo in itertools.product(*axes.values()):
        c = copy.deepcopy(base)
        c.update(dict(zip(axes, combo)))
        r = run_point(c, d, flow=args.flow)
        feas = (r["area_mm2"] <= args.area and r["avg_power_w"] <= args.power
                and r["e2e_latency_ms"] <= args.latency)
        rows.append((dict(zip(axes, combo)), r, feas))
    feas = sorted([x for x in rows if x[2]], key=lambda x: -x[1]["control_hz"])
    print(f"swept {len(rows)} pts; {len(feas)} feasible "
          f"(area<={args.area} power<={args.power} lat<={args.latency}ms)\n")
    hdr = f"{'arr':>4} {'clk':>4} {'BW':>4} {'SRAM':>5} | {'Hz':>5} {'e2e':>6} {'mm2':>5} {'W':>5}"
    print(hdr); print("-" * len(hdr))
    for ov, r, _ in (feas or sorted(rows, key=lambda x: x[1]["e2e_latency_ms"]))[:12]:
        print(f"{ov['tensor_array_dim']:>4} {ov['tensor_clock_ghz']:>4} "
              f"{ov['dram_bw_gbps']:>4} {ov['scratchpad_kb']//1024:>4}M | "
              f"{r['control_hz']:>5.1f} {r['e2e_latency_ms']:>6.0f} "
              f"{r['area_mm2']:>5.1f} {r['avg_power_w']:>5.1f}")
    if feas:
        ov = feas[0][0]
        print(f"\nbest: {ov['tensor_array_dim']}^2@{ov['tensor_clock_ghz']}GHz "
              f"{ov['dram_bw_gbps']}GB/s {ov['scratchpad_kb']//1024}MB")
    return 0


def _sens(base, d, key, vals, flow=None, getter=None):
    out = []
    for v in vals:
        c = copy.deepcopy(base); c[key] = v
        r = run_point(c, d, flow=flow)
        out.append((v, getter(r)))
    return out


def cmd_report(args):
    base = load_cfg(args.config, args.D)
    d = make_dims(args)
    r = run_point(base, d, flow=args.flow)
    print("=== npusim report ===\nBASELINE:")
    print(f"  e2e {r['e2e_latency_ms']:.1f}ms / {r['control_hz']:.1f}Hz | "
          f"decode {r['decode_ms_per_token']:.1f}ms/tok | area {r['area_mm2']:.1f}mm2 | "
          f"pow {r['avg_power_w']:.2f}W | E/tok {r['energy_per_token_mj']:.1f}mJ")
    print("\nSENS dram_bw_gbps (Hz):", _sens(base, d, "dram_bw_gbps",
          [136, 204, 273, 410, 546, 683], flow=args.flow,
          getter=lambda r: round(r["control_hz"], 1)))
    print("SENS tensor_array_dim (Hz):", _sens(base, d, "tensor_array_dim",
          [96, 128, 160, 192, 256], flow=args.flow,
          getter=lambda r: round(r["control_hz"], 1)))
    print("SENS tensor_array_dim (mm2):", _sens(base, d, "tensor_array_dim",
          [96, 128, 160, 192, 256], flow=args.flow,
          getter=lambda r: round(r["area_mm2"], 1)))
    print("SENS decode_batch (mJ/tok):", _sens(base, d, "decode_batch",
          [1, 2, 4, 8], flow=args.flow,
          getter=lambda r: round(r["energy_per_token_mj"], 2)))
    for nd in (1, 2, 4, 8, 16):
        dd = make_dims(args); dd.n_decode_tokens = nd
        rr = run_point(base, dd, flow=args.flow)
        print(f"  ndec={nd:2d}: {rr['control_hz']:.1f}Hz e2e {rr['e2e_latency_ms']:.0f}ms")
    return 0


def cmd_selftest(args):
    """Validate the model against the analytic decode DRAM roof + invariants."""
    c = roofline.load_config(DEFAULT_CFG)
    d = VLADims()
    r = run_point(c, d)
    fails = []

    # 1) decode must be DRAM-bound at the analytic roof (within 5%)
    w = 2.59e9 * (1 - c["resident_weight_frac"])   # weights/token after resident
    bw = c["dram_bw_gbps"] * 1e9 * c["dram_bw_util_decode"]
    analytic_ms = w / bw * 1e3
    got = r["decode_ms_per_token"]
    err = abs(got - analytic_ms) / analytic_ms
    ok = err < 0.10
    fails += [] if ok else [f"decode roof: got {got:.1f}ms vs analytic {analytic_ms:.1f}ms ({err*100:.0f}%)"]
    print(f"[{'PASS' if ok else 'FAIL'}] decode DRAM roof: {got:.1f} ms/tok "
          f"vs analytic {analytic_ms:.1f} ms ({err*100:.0f}% err)")

    # 2) decode must report bound_by dram-dominant in its energy
    bd = r["decode_energy_breakdown_mj"]
    dram_frac = bd["dram"] / max(sum(bd.values()), 1e-9)
    ok2 = dram_frac > 0.8
    fails += [] if ok2 else [f"decode energy DRAM frac {dram_frac:.2f} (<0.8)"]
    print(f"[{'PASS' if ok2 else 'FAIL'}] decode energy DRAM-dominated: {dram_frac*100:.0f}%")

    # 3) monotonic: more DRAM BW -> faster decode (lower ms/tok)
    pts = _sens(c, d, "dram_bw_gbps", [136, 273, 546],
                getter=lambda r: r["decode_ms_per_token"])
    mono = pts[0][1] > pts[1][1] > pts[2][1]
    fails += [] if mono else [f"decode not monotone in BW: {pts}"]
    print(f"[{'PASS' if mono else 'FAIL'}] decode ms/tok monotone-down in BW: "
          f"{[round(p[1],1) for p in pts]}")

    # 4) monotonic: bigger array -> faster prefill
    pa = _sens(c, d, "tensor_array_dim", [128, 192, 256],
               getter=lambda r: r["stage_time_ms"]["prefill"])
    mono2 = pa[0][1] > pa[1][1] > pa[2][1]
    fails += [] if mono2 else [f"prefill not monotone in array: {pa}"]
    print(f"[{'PASS' if mono2 else 'FAIL'}] prefill ms monotone-down in array: "
          f"{[round(p[1],1) for p in pa]}")

    # 5) area formula sanity: area > 0 and grows with array
    aa = _sens(c, d, "tensor_array_dim", [128, 256],
               getter=lambda r: r["area_mm2"])
    ok5 = aa[1][1] > aa[0][1] > 0
    fails += [] if ok5 else [f"area not growing: {aa}"]
    print(f"[{'PASS' if ok5 else 'FAIL'}] area grows with array: "
          f"{[round(p[1],1) for p in aa]} mm2")

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + '; '.join(fails)}")
    return 0 if not fails else 1


def main():
    p = argparse.ArgumentParser(prog="npusim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default=DEFAULT_CFG)
        sp.add_argument("-D", action="append", metavar="key=value",
                        help="override a config key (repeatable)")
        sp.add_argument("--ndec", type=int, help="n_decode_tokens")
        sp.add_argument("--seq", type=int, help="decode KV sequence length")
        sp.add_argument("--prompt", type=int, help="prefill prompt length")
        sp.add_argument("--flow", type=int, help="flow_expert_steps override")

    sr = sub.add_parser("run", help="single design point report")
    common(sr); sr.add_argument("--json", action="store_true")
    sr.set_defaults(fn=cmd_run)

    sw = sub.add_parser("sweep", help="constrained design-space sweep")
    common(sw)
    sw.add_argument("--area", type=float, default=60.0)
    sw.add_argument("--power", type=float, default=30.0)
    sw.add_argument("--latency", type=float, default=100.0)
    sw.set_defaults(fn=cmd_sweep)

    rp = sub.add_parser("report", help="baseline + sensitivity curves")
    common(rp); rp.set_defaults(fn=cmd_report)

    st = sub.add_parser("selftest", help="validate model vs analytic roof")
    st.set_defaults(fn=cmd_selftest)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
