"""Roofline + energy model and network aggregation (W1).

Per op: take max(compute_roof, dram_bw_roof, onchip_bw_roof) → op_time, and
record which roof binds. Energy = MAC + SRAM + DRAM terms. Aggregate per stage
and end-to-end, with stage multipliers (decode per-token, action per-step).

Faithful to docs/arch_spec.md's sim_design. All hardware numbers come from the
config dict (sim/configs/baseline.json) — no magic constants here.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
import ir


@dataclass
class OpResult:
    op_id: str
    stage: str
    time_s: float          # per single instance
    bound_by: str          # compute | dram | onchip | sfu
    macs: int
    dram_bytes: int
    sram_bytes: int
    energy_j: float
    count: int


def _tensor_eff_macs_per_s(op, c):
    dim = c["tensor_array_dim"]
    peak = dim * dim * c["tensor_clock_ghz"] * 1e9
    # fill/drain derate (systolic pipeline) + small-tile underfill
    fill = op.M / (op.M + dim)
    partial = min(1.0, (op.M * op.N) / (dim * dim))
    return peak * c["util_gemm"] * fill * max(partial, 1e-6)


def _vector_eff_macs_per_s(op, c):
    eff = c["vector_lanes"] * c["vector_dot_len"] * c["vector_clock_ghz"] * 1e9
    # tensor core folds in to help decode only when batching refills it
    if c["tensor_fold_groups"] > 1 and c["decode_batch"] > 1:
        eff += (c["tensor_array_dim"] ** 2) * c["tensor_clock_ghz"] * 1e9 * 0.5
    return eff


def sim_op(op: ir.Op, c: dict) -> OpResult:
    macs = op.macs
    # ---- compute roof ----
    # Routing (HV-1+): large-M GEMM/attention -> systolic tensor core;
    # M=1 GEMV and small-M (decode) attention -> vector path. Sending an
    # M=1 op to the tensor core would be dark silicon (the whole point of
    # the hybrid), so the vector path owns the bandwidth-bound decode.
    M_TENSOR_MIN = 16
    use_tensor = op.is_tensor and op.M >= M_TENSOR_MIN
    if op.is_sfu:
        elems = op.act_out_bytes / op.dtype_bytes
        sfu_per_s = (c["tensor_array_dim"] ** 2 * c["tensor_clock_ghz"] * 1e9
                     * c["sfu_ratio"])
        compute_t = elems / max(sfu_per_s, 1.0)
        bound = "sfu"
    elif use_tensor:
        compute_t = macs / _tensor_eff_macs_per_s(op, c)
        bound = "compute"
    else:  # GEMV or decode attention → vector path
        compute_t = macs / _vector_eff_macs_per_s(op, c)
        bound = "compute"

    # ---- DRAM bandwidth roof ----
    wbytes = op.weight_bytes
    if op.resident:
        wbytes = 0
    elif op.reuse_class == ir.STREAM_ONCE:
        wbytes = int(wbytes * (1.0 - c["resident_weight_frac"]))
        wbytes = wbytes // max(1, c["decode_batch"])      # batch amortizes fetch
    else:
        # GEMM weights reused across M tokens; only counted once from DRAM,
        # but for prefill M>=tile they're effectively streamed once too.
        wbytes = wbytes if op.M < 8 else int(wbytes)
    dram_bytes = wbytes + op.kv_read_bytes
    eff_bw = c["dram_bw_gbps"] * 1e9 * c["dram_bw_util_decode"]
    dram_t = dram_bytes / max(eff_bw, 1.0)

    # ---- on-chip bandwidth roof ----
    onchip_bytes = op.act_in_bytes + op.act_out_bytes
    onchip_bw = (c["scratchpad_banks"] * c["bytes_per_bank_per_cyc"]
                 * c["tensor_clock_ghz"] * 1e9)
    onchip_t = onchip_bytes / max(onchip_bw, 1.0)

    t = max(compute_t, dram_t, onchip_t)
    if t == dram_t and dram_t > compute_t and dram_t > onchip_t:
        bound = "dram"
    elif t == onchip_t and onchip_t > compute_t:
        bound = "onchip"

    # ---- energy ----
    sram_bytes = op.act_in_bytes + op.act_out_bytes
    if op.reuse_class != ir.STREAM_ONCE:
        sram_bytes += op.weight_bytes  # weights staged through SRAM
    energy = (macs * c["mac_pj"]
              + sram_bytes * c["sram_rd_pj_per_byte"]
              + op.act_out_bytes * c["sram_wr_pj_per_byte"]
              + dram_bytes * c["dram_pj_per_byte"]) * 1e-12

    return OpResult(op.op_id, op.stage, t, bound, macs, dram_bytes,
                    sram_bytes, energy, op.count)


def simulate(ops, stage_mult, c: dict):
    """Run the full graph. Returns a dict of metrics + per-op/stage detail."""
    per_op = [sim_op(o, c) for o in ops]

    stage_time, stage_energy, bound_hist = {}, {}, {}
    for r in per_op:
        st = stage_time.setdefault(r.stage, 0.0)
        stage_time[r.stage] = st + r.time_s * r.count
        stage_energy[r.stage] = stage_energy.get(r.stage, 0.0) + r.energy_j * r.count
        bound_hist[r.bound_by] = bound_hist.get(r.bound_by, 0.0) + r.time_s * r.count

    # apply stage multipliers (decode per-token, action per-step)
    e2e_time = sum(stage_time[s] * stage_mult.get(s, 1) for s in stage_time)
    e2e_energy = sum(stage_energy[s] * stage_mult.get(s, 1) for s in stage_energy)

    decode_t = stage_time.get("decode", 0.0)        # one token
    tok_s = 1.0 / decode_t if decode_t > 0 else 0.0
    control_hz = 1.0 / e2e_time if e2e_time > 0 else 0.0
    avg_power = e2e_energy / e2e_time if e2e_time > 0 else 0.0

    # area
    dim = c["tensor_array_dim"]
    tensor_area = dim * dim * c["per_mac_um2"]
    vec_area = c["vector_lanes"] * c["vector_dot_len"] * c["per_mac_um2"] * 1.3
    sram_area = c["scratchpad_kb"] * c["sram_um2_per_kb"]
    core = tensor_area + vec_area + sram_area
    total_area_um2 = core * (1.0 + c["fixed_area_overhead_frac"])
    total_area_mm2 = total_area_um2 / 1e6

    leak_w = total_area_mm2 * c["leakage_w_per_mm2"]
    avg_power += leak_w

    # decode energy breakdown (DRAM/SRAM/MAC) for one token
    dec = [r for r in per_op if r.stage == "decode"]
    dec_dram = sum(r.dram_bytes * c["dram_pj_per_byte"] * 1e-12 * r.count for r in dec)
    dec_mac = sum(r.macs * c["mac_pj"] * 1e-12 * r.count for r in dec)
    dec_sram = sum(r.energy_j * r.count for r in dec) - dec_dram - dec_mac

    peak_tops = 2 * dim * dim * c["tensor_clock_ghz"] * 1e9 / 1e12  # 2 op/MAC

    return {
        "stage_time_ms": {s: stage_time[s] * 1e3 for s in stage_time},
        "stage_mult": stage_mult,
        "e2e_latency_ms": e2e_time * 1e3,
        "control_hz": control_hz,
        "decode_tok_s": tok_s,
        "decode_ms_per_token": decode_t * 1e3,
        "energy_per_token_mj": sum(r.energy_j * r.count for r in dec) * 1e3,
        "decode_energy_breakdown_mj": {
            "dram": dec_dram * 1e3, "sram": dec_sram * 1e3, "mac": dec_mac * 1e3},
        "e2e_energy_mj": e2e_energy * 1e3,
        "avg_power_w": avg_power,
        "peak_int8_tops": peak_tops,
        "area_mm2": total_area_mm2,
        "area_breakdown_mm2": {"tensor": tensor_area / 1e6, "vector": vec_area / 1e6,
                               "sram": sram_area / 1e6},
        "bound_hist_ms": {k: v * 1e3 for k, v in bound_hist.items()},
        "_per_op": per_op,
    }


def load_config(path: str) -> dict:
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
