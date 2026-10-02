# Page-EntroKV

Hardware-aligned, entropy-weighted KV-cache eviction under grouped-query attention (GQA).

## What this paper shows

Long-context autoregressive decoding streams the full KV cache every step and is memory-bound. Under GQA several query heads share one physical KV buffer (`[N_blocks, H_KV, B, D_head]`). Head-independent top-k selection forces the engine to keep the union of divergent token sets (up to group ratio `r`). Mean pooling fixes the union but dilutes specialist retrieval heads by `(1-eps)/r`.

Page-EntroKV pools heads inside each physical group before selection:

- weights `w = softmax(-H2_sink / tau_g)` from sink-isolated Rényi-2 collision entropy, one dot product per head, computed once at prefill, no calibration
- block-max mapping `I(l,g,b) = max_{t in P_b} A_pooled(t)` and top-K_l page retention at `(l, g, P_b)`
- exact budget `UOR == 1.0` by construction, finite-context needle bound, temperature limits, per-layer cross-group accounting `K_l <= |P_ret| <= H_KV K_l`

Theory: exact UOR–disagreement identity at `r=2`, two-sided sandwich bound at general `r`, sink-suppression decay `N^{-1/tau}`, needle preservation under head specialization, largest-remainder budget conservation, page monotonicity.

## Pilot-tier status in this repo version

This version reports only what the supplied CSV logs contain. It is a pilot tier with an explicit deviation from the pre-registered Llama-3.1-8B Phase-1 scope. Log shape is consistent with `L=28`, 12 heads/layer, 2 groups (`r=6`). Hardware, engine version and seeds are not recorded in the logs.

- Structural: `experiment_1_structural_bounds_real.csv`, N=180, all `d_jaccard=0.0`, empirical UOR=1.0, Page-EntroKV UOR=1.0. Consensus null. No 4.75x bloat in this log.
- Sink: `experiment_2_sink_masquerade.csv`, 10 valid heads in layer 0 only, sink mass <0.7 percent, raw vs isolated delta <0.02 nats. Remaining layers empty and excluded. No 152/47/809 split in this log.
- NIAH: `experiment_3_niah_summary.csv`, 9 runs at 958/1474/1990 tokens x 20/50/80 percent, pooled-selection recall 1.0 vs mean 0.0. No 2499-token context, no strict all-layer column in this log.
- Page-fill: `experiment_s4_page_fill.csv`, B=16 only, slot 0 14.37 percent, slot 8 8.96 percent, slot 10 7.69 percent, rest 4.01-6.10 percent vs 6.25 percent uniform. No B={8,32,64} sweep in this log.
- LongBench: `experiment_5_longbench_summary.csv`, 4 completions (qasper 38.6, multifieldqa_en 44.2, lcc 56.1, repobench-p 44.81) with 11 tasks returning 0.0 in-log treated as harness non-completions and excluded, not zero accuracy. No dense baseline.
- No temperature sweep, no Renyi sweep, no PUR sweep, no serving tables in the supplied logs. Those remain pending under `paper/main.tex` Section 6 protocol.

## Layout

- `page_entrokv/`: core pooling, Renyi-2, block-max, largest-remainder allocator
- `baselines/`: pending patches, clearly marked, not measured here
- `evaluation/`: structural replay, NIAH pooled-selection sweep, LongBench runner with fault disclosure, guardrails stub pending
- `artifacts/logs/`: verbatim copies of the supplied CSVs used for transcription
- `artifacts/figures/`: only figures regenerable from the above logs; requested 3-class KDE / tau-alpha / strict-retention panels are pending
- `paper/`: `main.tex` pilot version plus table snippets transcribed from the same logs

## Reproduce pilot tables

```bash
pip install -r requirements.txt
python evaluation/harness_structural.py --log artifacts/logs/experiment_1_structural_bounds_real.csv
python evaluation/run_niah_sweep.py --log artifacts/logs/experiment_3_niah_summary.csv
python evaluation/run_longbench.py --log artifacts/logs/experiment_5_longbench_summary.csv
```

## Integrity rules followed here

Claims follow tables. Aggregates are recomputable. Empty cells are excluded, not zeroed. Faulted LongBench subsets are excluded, not reported as 0.00 accuracy. No 2240-row, tau/alpha, strict-retention or PUR numbers are claimed because no such rows exist in the supplied logs.

## Citation and data

Manuscript: `paper/main.tex`. Logs: `artifacts/logs/`. Harness: `evaluation/`. Release the CSVs and harness with the paper; add an anonymized link where the venue requires it.
