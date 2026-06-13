# NPU-for-VLA roadmap

Methodology: simulator-first (roofline → cycle-approx → cycle-accurate) → RTL,
mirroring the companion h264_by_claude_code / jpeg_by_claude_code ASIC flow.

| Wave | 内容 | 验收 |
|---|---|---|
| **W0** ✅ | spec 冻结 + 3-way 架构评审(systolic/vector/hybrid 打分综合 → HV-1+ 胜)+ 项目骨架。docs/arch_spec.md、22 个可扫超参、7nm/0.7V 能耗/面积系数、代表性 ~3B VLA 算子图 | 架构选定,系数/算子图落地 |
| **W1** ✅ | roofline simulator:算子 IR + 硬件配置 + compute/DRAM/on-chip 三 roof + 能耗模型 + 全网聚合 + sweep harness。**先用已知数据点验证**(decode ~2.6GB/token → DRAM roof) | sim 跑通,decode roof 与手算一致(20.4ms/49tok/s);首次 80 点扫描出受限 Pareto |
| W2 | cycle-approximate 事件驱动 sim(PE/SRAM/DMA/NoC 占用+冲突),验证 roofline 拐点;补 peak power、DMA 延迟隐藏精度 | 关键设计点 cycle-approx 复核 roofline |
| W3 | 工具链:W8A8 PTQ 量化器 + 编译器(graph→tiled NPU schedule,与 sim 同一 IR) | 量化精度 + 调度对拍 sim |
| W4 | cycle-accurate sim,对齐 RTL 前 | — |
| W5+ | RTL(sv2v+yosys+ASAP7,复用 H.264 流程),对拍 cycle-accurate | bit/cycle 对拍 + 综合 PPA |

## W1 首次扫描结论(80 点)

- decode 正确 DRAM-bound(模型修了一个把 M=1 attention 误路由到 systolic 阵列的 bug——这正是"先验证再信"拦下的)。
- 最优可行点 256×256@1.2GHz / 546GB/s → 13.6Hz / 37mm² / 6.5W;
  面积更省的甜点 192×192@1.2GHz / 546GB/s → 11.2Hz / **22.4mm²**。
- **设计是带宽/延迟受限,非功耗受限**(6.5W ≪ 30W,余量未用)。
- **8 个自回归 decode token 是主导成本**(每 token 流 2.6GB)→ 强烈倾向 flow-matching 动作专家而非长自回归动作 token decode;这是 VLA "口味"对 NPU 的最大杠杆。

## Web 仿真器

`web/index.html` + `web/npusim_model.js`:纯客户端单页(无后端/依赖/可离线),roofline 模型 JS 移植与 Python **逐数吻合**(baseline 271.8ms/3.68Hz、recommended 89.4ms/11.2Hz)。可选模型(3B/7B/1B)、实时拖超参、看各阶段延迟/瓶颈/能耗/**算子级仿真表**。node 交叉校验 `web/npusim_model.js`。

## 待办/下一步

- 扫 n_decode_tokens(flow-only vs 自回归)与 decode_batch(并行采样)的影响。
- 加 peak power(prefill burst)与功耗门控,验证 30W 峰值。
- W2 cycle-approx 复核 prefill 利用率(小 M attention/proj 的填充损失)与 DMA 双缓冲是否真能吃满 BW。
