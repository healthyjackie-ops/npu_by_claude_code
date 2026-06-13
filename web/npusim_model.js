// npusim_model.js — JavaScript port of the roofline simulator (sim/src/*.py).
// Faithful to ir.py + workload_vla.py + roofline.py. Runs in-browser (no deps)
// and in node (for the cross-check against the Python reference numbers).
(function (root) {
  "use strict";

  // ---- baseline HW config (mirror sim/configs/baseline.json) ----
  const BASELINE = {
    tensor_array_dim: 128, tensor_clock_ghz: 1.0,
    vector_lanes: 512, vector_dot_len: 4, vector_clock_ghz: 1.0,
    tensor_fold_groups: 1,
    scratchpad_kb: 4096, scratchpad_banks: 24, sram_partition_decode_frac: 0.5,
    bytes_per_bank_per_cyc: 16,
    dram_bw_gbps: 136.0, dram_bw_util_decode: 0.85, dram_bw_util_compute: 0.90, stream_tile_kb: 256,
    tile_buf: 2, dma_outstanding: 16,
    decode_batch: 1, kv_quant_bits: 8, kv_groups_gqa: 5, flow_expert_steps: 8,
    util_gemm: 0.85, sfu_ratio: 0.125, accum_width_bits: 32,
    resident_weight_frac: 0.1,
    mac_pj: 0.18, sram_rd_pj_per_byte: 0.35, sram_wr_pj_per_byte: 0.42,
    dram_pj_per_byte: 8.0, per_mac_um2: 430.0, sram_um2_per_kb: 410.0,
    fixed_area_overhead_frac: 0.20, leakage_w_per_mm2: 0.05,
  };

  // ---- model presets (VLADims). 3B is the default representative graph ----
  const MODELS = {
    "vla_3b": {
      label: "~3B VLA (ViT24 + LLM28 + flow head)",
      vit_layers: 24, vit_hidden: 1152, vit_patches: 576, vit_heads: 16, vit_mlp: 4304, patch_k: 588,
      llm_layers: 28, llm_hidden: 2560, q_heads: 20, kv_heads: 4, head_dim: 128, llm_mlp: 8192, vocab: 152064,
      conn_out: 2560, act_tokens: 64, act_hidden: 1024, act_blocks: 18, act_dof: 32,
      prompt_len: 256, seq_len: 832, n_decode_tokens: 8,
    },
    "vla_7b": {
      label: "~7B VLA (OpenVLA-class, Llama2-7B + SigLIP)",
      vit_layers: 27, vit_hidden: 1152, vit_patches: 576, vit_heads: 16, vit_mlp: 4304, patch_k: 588,
      llm_layers: 32, llm_hidden: 4096, q_heads: 32, kv_heads: 32, head_dim: 128, llm_mlp: 11008, vocab: 32064,
      conn_out: 4096, act_tokens: 7, act_hidden: 4096, act_blocks: 0, act_dof: 7,
      prompt_len: 276, seq_len: 880, n_decode_tokens: 7,
    },
    "vla_1b": {
      label: "~1B distilled VLA (TinyVLA/Octo-class)",
      vit_layers: 12, vit_hidden: 768, vit_patches: 256, vit_heads: 12, vit_mlp: 3072, patch_k: 588,
      llm_layers: 16, llm_hidden: 2048, q_heads: 16, kv_heads: 4, head_dim: 128, llm_mlp: 5632, vocab: 32000,
      conn_out: 2048, act_tokens: 64, act_hidden: 512, act_blocks: 12, act_dof: 14,
      prompt_len: 128, seq_len: 384, n_decode_tokens: 4,
    },
  };

  const SFU = new Set(["softmax", "norm", "act"]);
  const ATTN = new Set(["attn_score", "attn_av"]);
  const TENSORY = new Set(["gemm", "conv", "attn_score", "attn_av"]);

  function macs(o) {
    if (ATTN.has(o.type)) return o.M * o.N * o.K * o.heads;
    if (SFU.has(o.type)) return 0;
    return o.M * o.N * o.K;
  }
  function weightBytes(o) {
    if (o.has_weights === false || SFU.has(o.type)) return 0;
    return o.K * o.N;
  }
  function actInBytes(o) {
    const h = ATTN.has(o.type) ? o.heads : 1;
    return o.M * o.K * Math.max(1, h);
  }
  function actOutBytes(o) {
    const h = ATTN.has(o.type) ? o.heads : 1;
    return o.M * o.N * Math.max(1, h);
  }

  // operator-graph builder (port of workload_vla.build_vla)
  function buildVLA(d, kvBits, flowSteps, decodeBatch) {
    const B = Math.max(1, decodeBatch || 1);
    const ops = [];
    const op = (id, stage, type, o) => ops.push(Object.assign(
      { op_id: id, stage, type, M: 1, N: 1, K: 1, heads: 1, count: 1,
        reuse: "output_stationary", resident: false, has_weights: true,
        kv_read_bytes: 0 }, o));
    const H = d.vit_hidden, hd = (H / d.vit_heads) | 0;
    // ViT
    op("vit.patch_embed", "vit", "conv", { M: d.vit_patches, K: d.patch_k, N: H });
    op("vit.qkv", "vit", "gemm", { M: d.vit_patches, K: H, N: 3 * H, count: d.vit_layers });
    op("vit.qk", "vit", "attn_score", { M: d.vit_patches, K: hd, N: d.vit_patches, heads: d.vit_heads, count: d.vit_layers, has_weights: false });
    op("vit.softmax", "vit", "softmax", { M: d.vit_patches, N: d.vit_patches, heads: d.vit_heads, count: d.vit_layers, has_weights: false });
    op("vit.av", "vit", "attn_av", { M: d.vit_patches, K: d.vit_patches, N: hd, heads: d.vit_heads, count: d.vit_layers, has_weights: false });
    op("vit.o", "vit", "gemm", { M: d.vit_patches, K: H, N: H, count: d.vit_layers });
    op("vit.fc1", "vit", "gemm", { M: d.vit_patches, K: H, N: d.vit_mlp, count: d.vit_layers });
    op("vit.gelu", "vit", "act", { M: d.vit_patches, N: d.vit_mlp, count: d.vit_layers, has_weights: false });
    op("vit.fc2", "vit", "gemm", { M: d.vit_patches, K: d.vit_mlp, N: H, count: d.vit_layers });
    op("vit.norm", "vit", "norm", { M: d.vit_patches, N: H, count: 2 * d.vit_layers, has_weights: false });
    // connector
    op("conn.proj", "connector", "gemm", { M: d.vit_patches, K: H, N: d.conn_out, count: 2 });
    // LLM prefill
    const Hl = d.llm_hidden, qn = d.q_heads * d.head_dim, kvn = d.kv_heads * d.head_dim, P = d.vit_patches + d.prompt_len;
    op("pf.q", "prefill", "gemm", { M: P, K: Hl, N: qn, count: d.llm_layers });
    op("pf.kv", "prefill", "gemm", { M: P, K: Hl, N: 2 * kvn, count: d.llm_layers });
    op("pf.qk", "prefill", "attn_score", { M: P, K: d.head_dim, N: P, heads: d.q_heads, count: d.llm_layers, has_weights: false });
    op("pf.softmax", "prefill", "softmax", { M: P, N: P, heads: d.q_heads, count: d.llm_layers, has_weights: false });
    op("pf.av", "prefill", "attn_av", { M: P, K: P, N: d.head_dim, heads: d.q_heads, count: d.llm_layers, has_weights: false });
    op("pf.o", "prefill", "gemm", { M: P, K: qn, N: Hl, count: d.llm_layers });
    op("pf.gate", "prefill", "gemm", { M: P, K: Hl, N: d.llm_mlp, count: d.llm_layers });
    op("pf.up", "prefill", "gemm", { M: P, K: Hl, N: d.llm_mlp, count: d.llm_layers });
    op("pf.silu", "prefill", "act", { M: P, N: d.llm_mlp, count: d.llm_layers, has_weights: false });
    op("pf.down", "prefill", "gemm", { M: P, K: d.llm_mlp, N: Hl, count: d.llm_layers });
    op("pf.norm", "prefill", "norm", { M: P, N: Hl, count: 2 * d.llm_layers, has_weights: false });
    // LLM decode (one token; stage x n_decode_tokens)
    const kvb = Math.floor((kvBits + 7) / 8);
    const kvRead = 2 * d.seq_len * d.kv_heads * d.head_dim * kvb;
    op("dec.q", "decode", "gemv", { M: B, K: Hl, N: qn, count: d.llm_layers, reuse: "stream_once" });
    op("dec.kv", "decode", "gemv", { M: B, K: Hl, N: 2 * kvn, count: d.llm_layers, reuse: "stream_once" });
    op("dec.qk", "decode", "attn_score", { M: B, K: d.head_dim, N: d.seq_len, heads: d.q_heads, count: d.llm_layers, has_weights: false, kv_read_bytes: ((kvRead / 2) | 0) * B, reuse: "stream_once" });
    op("dec.softmax", "decode", "softmax", { M: B, N: d.seq_len, heads: d.q_heads, count: d.llm_layers, has_weights: false });
    op("dec.av", "decode", "attn_av", { M: B, K: d.seq_len, N: d.head_dim, heads: d.q_heads, count: d.llm_layers, has_weights: false, kv_read_bytes: ((kvRead / 2) | 0) * B, reuse: "stream_once" });
    op("dec.o", "decode", "gemv", { M: B, K: qn, N: Hl, count: d.llm_layers, reuse: "stream_once" });
    op("dec.gate", "decode", "gemv", { M: B, K: Hl, N: d.llm_mlp, count: d.llm_layers, reuse: "stream_once" });
    op("dec.up", "decode", "gemv", { M: B, K: Hl, N: d.llm_mlp, count: d.llm_layers, reuse: "stream_once" });
    op("dec.silu", "decode", "act", { M: B, N: d.llm_mlp, count: d.llm_layers, has_weights: false });
    op("dec.down", "decode", "gemv", { M: B, K: d.llm_mlp, N: Hl, count: d.llm_layers, reuse: "stream_once" });
    op("dec.norm", "decode", "norm", { M: B, N: Hl, count: 2 * d.llm_layers, has_weights: false });
    op("dec.lm_head", "decode", "gemv", { M: B, K: Hl, N: d.vocab, count: 1, reuse: "stream_once" });
    // action expert (flow DiT; stage x flow_steps)
    if (d.act_blocks > 0) {
      const A = d.act_hidden, Na = d.act_tokens, ah = (A / 8) | 0;
      op("act.in", "action", "gemm", { M: Na, K: d.conn_out, N: A, count: 1, resident: true });
      op("act.qkv", "action", "gemm", { M: Na, K: A, N: 3 * A, count: d.act_blocks, resident: true });
      op("act.qk", "action", "attn_score", { M: Na, K: ah, N: Na, heads: 8, count: d.act_blocks, has_weights: false });
      op("act.av", "action", "attn_av", { M: Na, K: Na, N: ah, heads: 8, count: d.act_blocks, has_weights: false });
      op("act.o", "action", "gemm", { M: Na, K: A, N: A, count: d.act_blocks, resident: true });
      op("act.mlp1", "action", "gemm", { M: Na, K: A, N: 4 * A, count: d.act_blocks, resident: true });
      op("act.mlp2", "action", "gemm", { M: Na, K: 4 * A, N: A, count: d.act_blocks, resident: true });
      op("act.timestep", "action", "act", { M: Na, N: A, count: d.act_blocks, has_weights: false });
      op("act.out", "action", "gemm", { M: Na, K: A, N: d.act_dof, count: 1, resident: true });
    }
    const mult = { vit: 1, connector: 1, prefill: 1,
                   decode: d.n_decode_tokens, action: flowSteps };
    return { ops, mult };
  }

  function tensorEff(o, c) {
    const dim = c.tensor_array_dim;
    const peak = dim * dim * c.tensor_clock_ghz * 1e9;
    const fill = o.M / (o.M + dim);
    const partial = Math.min(1.0, (o.M * o.N) / (dim * dim));
    return peak * c.util_gemm * fill * Math.max(partial, 1e-6);
  }
  function vectorEff(o, c) {
    let eff = c.vector_lanes * c.vector_dot_len * c.vector_clock_ghz * 1e9;
    if (c.tensor_fold_groups > 1 && c.decode_batch > 1)
      eff += (c.tensor_array_dim ** 2) * c.tensor_clock_ghz * 1e9 * 0.5;
    return eff;
  }

  function simOp(o, c) {
    const m = macs(o), wB = weightBytes(o), aIn = actInBytes(o), aOut = actOutBytes(o);
    const M_TENSOR_MIN = 16;
    const useTensor = TENSORY.has(o.type) && o.M >= M_TENSOR_MIN;
    let compute_t, bound;
    if (SFU.has(o.type)) {
      const elems = aOut;
      const sfuPerS = (c.tensor_array_dim ** 2) * c.tensor_clock_ghz * 1e9 * c.sfu_ratio;
      compute_t = elems / Math.max(sfuPerS, 1); bound = "sfu";
    } else if (useTensor) {
      compute_t = m / tensorEff(o, c); bound = "compute";
    } else {
      compute_t = m / vectorEff(o, c); bound = "compute";
    }
    // DRAM roof (weights fetched once; batch raises M/compute, not traffic)
    const stream = o.reuse === "stream_once";
    let wb = o.resident ? 0 : wB;
    if (stream) wb = Math.floor(wb * (1 - c.resident_weight_frac));
    const dramBytes = wb + o.kv_read_bytes;
    const util = stream ? c.dram_bw_util_decode
      : (c.dram_bw_util_compute != null ? c.dram_bw_util_compute : c.dram_bw_util_decode);
    const dram_t = dramBytes / Math.max(c.dram_bw_gbps * 1e9 * util, 1);
    // on-chip roof (acts always; weights staged for tensor GEMM / streaming)
    let onchipBytes = aIn + aOut;
    if (useTensor && !o.resident && !stream) onchipBytes += wB;
    const onchipBw = c.scratchpad_banks * c.bytes_per_bank_per_cyc * c.tensor_clock_ghz * 1e9;
    const onchip_t = onchipBytes / Math.max(onchipBw, 1);
    let t = Math.max(compute_t, dram_t, onchip_t);
    if (!SFU.has(o.type)) {
      if (t === dram_t && dram_t >= compute_t && dram_t >= onchip_t) bound = "dram";
      else if (t === onchip_t && onchip_t > compute_t) bound = "onchip";
    }
    // energy: DRAM->SRAM fill (write) + SRAM->PE (read) + MAC
    const stagedW = o.resident ? 0 : wB;
    const sramRd = aIn + stagedW, sramWr = aOut + stagedW;
    const energy = (m * c.mac_pj + sramRd * c.sram_rd_pj_per_byte
      + sramWr * c.sram_wr_pj_per_byte + dramBytes * c.dram_pj_per_byte) * 1e-12;
    return { op_id: o.op_id, stage: o.stage, type: o.type, M: o.M, N: o.N, K: o.K,
      time_s: t, bound, macs: m, dram_bytes: dramBytes, energy_j: energy, count: o.count };
  }

  function simulate(ops, mult, c) {
    const per = ops.map(o => simOp(o, c));
    const batch = Math.max(1, c.decode_batch || 1);
    const stageTime = {}, stageEnergy = {}, boundHist = {};
    for (const r of per) {
      stageTime[r.stage] = (stageTime[r.stage] || 0) + r.time_s * r.count;
      stageEnergy[r.stage] = (stageEnergy[r.stage] || 0) + r.energy_j * r.count;
      boundHist[r.bound] = (boundHist[r.bound] || 0) + r.time_s * r.count;
    }
    let e2e = 0, e2eE = 0;
    for (const s in stageTime) { e2e += stageTime[s] * (mult[s] || 1); e2eE += stageEnergy[s] * (mult[s] || 1); }
    const dim = c.tensor_array_dim;
    const tensorArea = dim * dim * c.per_mac_um2;
    const vecArea = c.vector_lanes * c.vector_dot_len * c.per_mac_um2 * 1.3;
    const sramArea = c.scratchpad_kb * c.sram_um2_per_kb;
    const areaMm2 = (tensorArea + vecArea + sramArea) * (1 + c.fixed_area_overhead_frac) / 1e6;
    const leakW = areaMm2 * c.leakage_w_per_mm2;
    e2eE += leakW * e2e;                              // leakage as an energy term
    const avgPow = e2e > 0 ? e2eE / e2e : 0;
    let peakPow = 0;                                  // worst sequential stage
    for (const s in stageTime) if (stageTime[s] > 0)
      peakPow = Math.max(peakPow, stageEnergy[s] / stageTime[s] + leakW);
    const decStep = stageTime.decode || 0, decStepE = stageEnergy.decode || 0;
    const dec = per.filter(r => r.stage === "decode");
    const decDram = dec.reduce((a, r) => a + r.dram_bytes * c.dram_pj_per_byte * 1e-12 * r.count, 0);
    const decMac = dec.reduce((a, r) => a + r.macs * c.mac_pj * 1e-12 * r.count, 0);
    return {
      stage_time_ms: Object.fromEntries(Object.entries(stageTime).map(([k, v]) => [k, v * 1e3])),
      stage_mult: mult,
      e2e_latency_ms: e2e * 1e3,
      control_hz: e2e > 0 ? 1 / e2e : 0,
      decode_ms_per_token: (decStep / batch) * 1e3,
      decode_tok_s: decStep > 0 ? batch / decStep : 0,
      energy_per_token_mj: (decStepE / batch) * 1e3,
      decode_energy_breakdown_mj: { dram: decDram / batch * 1e3, mac: decMac / batch * 1e3,
        sram: (decStepE - decDram - decMac) / batch * 1e3 },
      e2e_energy_mj: e2eE * 1e3,
      avg_power_w: avgPow, peak_power_w: peakPow,
      peak_int8_tops: 2 * dim * dim * c.tensor_clock_ghz * 1e9 / 1e12,
      area_mm2: areaMm2,
      area_breakdown_mm2: { tensor: tensorArea / 1e6, vector: vecArea / 1e6, sram: sramArea / 1e6 },
      bound_hist_ms: Object.fromEntries(Object.entries(boundHist).map(([k, v]) => [k, v * 1e3])),
      per_op: per,
    };
  }

  const API = { BASELINE, MODELS, buildVLA, simOp, simulate, macs, weightBytes };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.NPUSim = API;
})(typeof window !== "undefined" ? window : globalThis);
