# `compute_metric_early_predict`: causal prediction algorithm adapted from `compute_metric`

The base algorithm (`compute_metric`) uses a symmetric +-10-frame window around each GT fall for both centroidZ drop detection and anomaly spike matching, making it a post-hoc detector. `compute_metric_early_predict` adapts this into a causal, predictive algorithm with the following key differences.

---

## Detection (`detect_falls`)

- Uses a **backward-looking 10-frame window** (not centered) so detections are causal
- Anomaly spike must occur in a window **before** the centroidZ drop, offset by a configurable `prediction_gap` -- rewarding models that fire early

| Aspect | `compute_metric` (base) | `compute_metric_early_predict` |
|---|---|---|
| CentroidZ window | 20 frames, centered (symmetric) | 10 frames, backward-looking (causal) |
| Detected frame index | Middle of the 20-frame window | End of the 10-frame window |
| Anomaly check window | +-10 around the cluster midpoint | Backward from centroidZ window start, size = `lower_bound` |
| Clustering | Done in `detect_falls` before anomaly check; midpoint is the representative | Done in `find_tpfpfn` after all detection; raw detections passed through |
| TP matching | Detection-first: for each detection, find first matching GT | Cluster-first: for each cluster, find first matching GT |
| TP matching window | `gt +- 10` (symmetric) | `[gt_first - lower_bound, gt_first - prediction_gap]` (asymmetric, pre-fall only) |

---

## Ground truth pre-processing (new cells)

The raw GT annotations mark approximate fall times. Three derived datasets are computed:

- **DS2_START_FALLS**: locates the actual "body hits ground" frame per GT fall by scoring candidates on backward centroidZ drop rate minus forward change, replacing the raw GT annotation frames with precise hit-ground frames
- **DS2_PREDICT**: expands each GT fall into the full set of frames where the person is on the ground (centroidZ < threshold), used for FP filtering
- **DS2_PREDICT_FIRSTS**: the first frame of each expanded fall region

---

## TP/FP/FN classification (`find_tpfpfn`)

Clusters raw detections within 20-frame windows (same as base), then classifies:

- **TP**: a cluster has detections in `[gt_first - lower_bound, gt_first - prediction_gap]` (causal window before the fall, not symmetric around it). That GT fall is consumed (1 TP per GT).
- **FP**: a non-TP cluster where **none** of its frames overlap any GT fall region (`gt_predict_frames` from DS2_PREDICT). Clusters that overlap a real fall but weren't early enough are ignored -- not penalized as FP. When `gt_predict_frames` is `None`, all non-TP clusters are FP (legacy behavior).
- **FN**: GT falls with no matching cluster (same as base).

The key fix vs. the previous version: FP previously used a narrow time window and didn't use DS2_PREDICT at all, causing detections during real falls to be misclassified as FP and FP counts to wobble non-monotonically as threshold increased.

---

## Weighted anomaly detection (`method='velocity'` only)

When `weighted=True`, the effective anomaly threshold is reduced proportionally to how far the estimated centroidZ has dropped from the person's standing height at the start of the detection window:

```
Z_ref = centroidZ_history[start_win]
estimated_Z = centroidZ_history[j] + centroidVz_history[j] * prediction_gap * frame_duration
ratio = max(0, (Z_ref - estimated_Z) / Z_ref)
weight = 1 + scale * ratio^power
effective_threshold = threshold / weight
```

- `Z_ref` is per-window (not a global constant) -- it's the height before each potential fall began
- `scale` (default 1.0): controls maximum extra weight; when estimated_Z = 0, weight = 1 + scale
- `power` (default 1.0): 1 = linear, 2 = quadratic relationship between drop ratio and weight
- When `weighted=False` (default) or `method='drop'`, the original binary anomaly check is used -- all existing callers are unaffected

---

## ROC evaluation

- `cal_roc_anomaly` sweeps anomaly threshold (fixed `centroidZ_thr=0.6`)
- `cal_roc_centroid` sweeps centroidZ threshold (fixed `anomaly_thr=5`)
- Both accept `gt_predict_frames` for proper FP filtering
- Both forward `weighted`, `weight_scale`, `weight_power` to `detect_falls`
- Plots: TPR-vs-FP ROC, TP-vs-FP count, TPR-vs-threshold, FP-vs-threshold

---

## Debug/visualization cells

- Per-GT-fall centroidZ + causally-aligned anomaly overlay plots
- HVRAE-vs-HVRAE_SL missed/caught fall analysis
- FP cluster analysis for each model independently
- Weighted velocity ROC: compares drop, unweighted radar, and weighted radar with various (power, scale) configs
- Weighted centroid sweep: centroid threshold sweep comparing unweighted vs weighted radar
