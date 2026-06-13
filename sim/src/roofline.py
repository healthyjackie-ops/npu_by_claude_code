"""Roofline + energy model and network aggregation (W1, hardened W1.1).

Per op: take max(compute_roof, dram_bw_roof, onchip_bw_roof) -> op_time, record
which roof binds. Energy = MAC + SRAM(rd+wr) + DRAM terms. Aggregate per stage
and end-to-end, with stage multipliers (decode per-token, action per-step).

W1.1 fixes from adversarial review (all mirrored in web/npusim_model.js):
  - on-chip roof now includes staged weight bytes for tensor GEMMs (it could
    never bind a weight-heavy GEMM before; inconsistent with the energy model).
  - SRAM energy: reads = act_in + staged weights; writes = act_out + DRAM->SRAM
    weight fill. (Was charging act_out as a read of its own output, and never
    charging streamed/decode weights any on-chip staging.)
  - separate DRAM-BW utilisation for compute-stage streaming vs decode.
  - SFU ops keep their 'sfu' binding tag in the histogram.
  - leakage folded into e2e_energy so avg_power == e2e_energy/e2e_time exactly.
  - peak_power_w reported (worst sequential-stage instantaneous power).
  - decode_batch handled in the workload (M=batch, shared weight fetch); here we
    just divide the per-step decode metrics by the batch for per-token figures.

All hardware numbers come from the config dict — no magic constants here.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
import ir


@dataclass
class OpResult:
    op_id: str
    stage: str
    time_s: float
    bound_by: str
    macs: int
    dram_bytes: int
    sram_bytes: int
    energy_j: float
    count: int


def _tensor_eff_macs_per_s(op, c):
    dim = c["tensor_array_dim"]
    peak = dim * dim * c["tensor_clock_ghz"] * 1e9
    fill = op.M / (op.M + dim)
    partial = min(1.0, (op.M * op.N) / (dim * dim))
    return peak * c["util_gemm"] * fill * max(partial, 1e-6)


def _vector_eff_macs_per_s(op, c):
    eff = c["vector_lanes"] * c["vector_dot_len"] * c["vector_clock_ghz"] * 1e9
    if c["tensor_fold_groups"] > 1 and c["decode_batch"] > 1:
        eff += (c["tensor_array_dim"] ** 2) * c["tensor_clock_ghz"] * 1e9 * 0.5
    return eff


def sim_op(op: ir.Op, c: dict) -> OpResult:
    macs = op.macs
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
    else:
        compute_t = macs / _vector_eff_macs_per_s(op, c)
        bound = "compute"

    # ---- DRAM bandwidth roof ----
    stream = (op.reuse_class == ir.STREAM_ONCE)
    wbytes = 0 if op.resident else op.weight_bytes
    if stream:
        wbytes = int(wbytes * (1.0 - c["resident_weight_frac"]))
        # weights are fetched ONCE and shared across the decode_batch rows
        # (the batch raises M/compute in the workload, not the weight traffic)
    dram_bytes = wbytes + op.kv_read_bytes
    util = c["dram_bw_util_decode"] if stream else c.get("dram_bw_util_compute",
                                                          c["dram_bw_util_decode"])
    eff_bw = c["dram_bw_gbps"] * 1e9 * util
    dram_t = dram_bytes / max(eff_bw, 1.0)

    # ---- on-chip bandwidth roof ----
    # acts always; staged+reused tensor-GEMM weights traverse the scratchpad
    # banks (finding #2). STREAM_ONCE decode weights ride the DMA double-buffer
    # path (BW provisioned ~= DRAM), NOT the narrow activation banks, so they
    # are not gated here (their staging energy is still charged below).
    onchip_bytes = op.act_in_bytes + op.act_out_bytes
    if use_tensor and not op.resident and not stream:
        onchip_bytes += op.weight_bytes
    onchip_bw = (c["scratchpad_banks"] * c["bytes_per_bank_per_cyc"]
                 * c["tensor_clock_ghz"] * 1e9)
    onchip_t = onchip_bytes / max(onchip_bw, 1.0)

    t = max(compute_t, dram_t, onchip_t)
    if not op.is_sfu:                           # SFU keeps its identity tag
        if t == dram_t and dram_t >= compute_t and dram_t >= onchip_t:
            bound = "dram"
        elif t == onchip_t and onchip_t > compute_t:
            bound = "onchip"

    # ---- energy: DRAM->SRAM fill (write) + SRAM->PE (read) + MAC ----
    staged_w = 0 if op.resident else op.weight_bytes
    sram_rd = op.act_in_bytes + staged_w
    sram_wr = op.act_out_bytes + staged_w       # weight fill DRAM->SRAM
    energy = (macs * c["mac_pj"]
              + sram_rd * c["sram_rd_pj_per_byte"]
              + sram_wr * c["sram_wr_pj_per_byte"]
              + dram_bytes * c["dram_pj_per_byte"]) * 1e-12

    return OpResult(op.op_id, op.stage, t, bound, macs, dram_bytes,
                    sram_rd + sram_wr, energy, op.count)


def simulate(ops, stage_mult, c: dict):
    per_op = [sim_op(o, c) for o in ops]
    batch = max(1, int(c.get("decode_batch", 1)))

    stage_time, stage_energy, bound_hist = {}, {}, {}
    for r in per_op:
        stage_time[r.stage] = stage_time.get(r.stage, 0.0) + r.time_s * r.count
        stage_energy[r.stage] = stage_energy.get(r.stage, 0.0) + r.energy_j * r.count
        bound_hist[r.bound_by] = bound_hist.get(r.bound_by, 0.0) + r.time_s * r.count

    e2e_time = sum(stage_time[s] * stage_mult.get(s, 1) for s in stage_time)
    e2e_energy = sum(stage_energy[s] * stage_mult.get(s, 1) for s in stage_energy)

    # area
    dim = c["tensor_array_dim"]
    tensor_area = dim * dim * c["per_mac_um2"]
    vec_area = c["vector_lanes"] * c["vector_dot_len"] * c["per_mac_um2"] * 1.3
    sram_area = c["scratchpad_kb"] * c["sram_um2_per_kb"]
    total_area_mm2 = ((tensor_area + vec_area + sram_area)
                      * (1.0 + c["fixed_area_overhead_frac"])) / 1e6

    # leakage as an energy term too, so power & energy share one base
    leak_w = total_area_mm2 * c["leakage_w_per_mm2"]
    e2e_energy += leak_w * e2e_time
    avg_power = e2e_energy / e2e_time if e2e_time > 0 else 0.0

    # peak (worst sequential-stage instantaneous power) — stages run serially
    peak_power = 0.0
    for s in stage_time:
        if stage_time[s] > 0:
            peak_power = max(peak_power, stage_energy[s] / stage_time[s] + leak_w)

    # decode per-token figures: one decode step yields `batch` parallel samples
    decode_step = stage_time.get("decode", 0.0)
    decode_step_e = stage_energy.get("decode", 0.0)
    dec_ms_per_token = (decode_step / batch) * 1e3
    tok_s = (batch / decode_step) if decode_step > 0 else 0.0

    dec = [r for r in per_op if r.stage == "decode"]
    dec_dram = sum(r.dram_bytes * c["dram_pj_per_byte"] * 1e-12 * r.count for r in dec)
    dec_mac = sum(r.macs * c["mac_pj"] * 1e-12 * r.count for r in dec)
    dec_e = decode_step_e
    peak_tops = 2 * dim * dim * c["tensor_clock_ghz"] * 1e9 / 1e12

    return {
        "stage_time_ms": {s: stage_time[s] * 1e3 for s in stage_time},
        "stage_mult": stage_mult,
        "e2e_latency_ms": e2e_time * 1e3,
        "control_hz": 1.0 / e2e_time if e2e_time > 0 else 0.0,
        "decode_tok_s": tok_s,
        "decode_ms_per_token": dec_ms_per_token,
        "energy_per_token_mj": (dec_e / batch) * 1e3,
        "decode_energy_breakdown_mj": {
            "dram": dec_dram / batch * 1e3, "sram": (dec_e - dec_dram - dec_mac) / batch * 1e3,
            "mac": dec_mac / batch * 1e3},
        "e2e_energy_mj": e2e_energy * 1e3,
        "avg_power_w": avg_power,
        "peak_power_w": peak_power,
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
