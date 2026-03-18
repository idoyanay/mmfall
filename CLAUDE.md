# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mmFall is a fall detection system using a 4D mmWave radar (Texas Instruments AWR1843BOOST) and a Hybrid Variational RNN AutoEncoder (HVRAE). The system is trained exclusively on normal Activities of Daily Living (ADL), then detects falls as anomalies at inference time — a semi-supervised approach requiring no fall data for training.

The primary reference for results is the notebook `src/mmfall.ipynb`. The paper is at https://arxiv.org/abs/2003.02386.

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
pip install tensorflow keras    # not in requirements.txt; install manually
```

The `venv/` uses Python 3.13. Note that `requirements.txt` only pins data/plotting dependencies (numpy, matplotlib, h5py, etc.) — TensorFlow/Keras must be installed separately.

## Running the Code

The main entry point is the Jupyter notebook:
```bash
jupyter notebook src/mmfall.ipynb
```

`src/mmfall.py` is a standalone script version of the same logic (same classes, same architecture).

Utility scripts at the project root are run directly:
```bash
# Combine multiple .npy dataset files + their .csv ground truth timesheets into one
python combine.py --filedir data/DS2/

# Visualize raw radar point cloud or processed features (3D animated)
python data_visualizer.py --gtfile <raw.npy> --binfile <processed.npy>

# Inspect processed vs predicted data point-by-point
python data_analyzer.py --test <test.npy> --prediction <pred.npy> --raw <raw.npy>
```

## Architecture Overview

### Data Pipeline

Raw `.npy` files contain per-frame radar point arrays with 15 fields each:
`[frame_id, point_id, target_id, centroidX/Y/Z, centroidVx/Vy/Vz, range, azimuth, elevation, Doppler, SNR, noise]`

`data_preproc.load_bin()` processes raw data into motion patterns of shape `(N, 10, 64, 4)`:
- Groups consecutive frames into overlapping 10-frame windows (1 second at 10 fps)
- Applies a rotation matrix (10° tilt correction) + height offset (1.8 m) to transform radar coords to ground coords
- Also rotates the centroid velocity vector to ground coordinates (no height offset) and records `centroidVz_history`
- Extracts feature vector `(delta_x, delta_y, z, Doppler)` per point — x/y are centroid-relative, z is absolute
- Runs `proposed_oversampling()` to normalize each frame to exactly 64 points: rescales existing points to preserve mean/variance, then pads to 64 with the mean vector

When `fortrain=False`, returns `(total_processed_pattern, centroidZ_his, centroidVz_his)`.

Pre-processed training data is cached as `data/normal_train_data.npy` and `data/normal_test_data.npy` (from DS0, 80/20 split).

### Models (in `autoencoder_mdl`)

All three models share the same input shape `(10, 64, 4)` — 10 frames, 64 points, 4 features.

**HVRAE** (proposed model): VAE encoder → reparameterization sampling → SimpleRNN encoder-decoder → VAE decoder outputting `(mean, log_var)` per point. Loss = Gaussian reconstruction log-likelihood + KL divergence. Output shape: `(10, 64, 8)`.

**HVRAE_SL** (baseline 1): Same as HVRAE but decoder outputs only `mean` (no `log_var`). Loss = MSE + KL divergence. Output shape: `(10, 64, 4)`.

**RAE** (baseline 2): Standard recurrent autoencoder with MSE loss, no variational components. Output shape: `(10, 64, 4)`.

All models use Adam (`lr=0.001`), trained for 5 epochs, batch size 8, no shuffle. Saved weights are in `saved_model/` as `.h5` files. At inference, models are rebuilt from code via `_build_*_architecture()` then weights are loaded with `model.load_weights()`.

### Fall Detection Logic

Inference produces a per-frame `loss_history` (anomaly score) alongside a `centroidZ_history` (body height over time).

**`compute_metric`** (symmetric, post-hoc): A fall is detected when a centroid height drop ≥ 0.6 m occurs within a 20-frame (2 sec) centered window AND an anomaly spike exceeds the threshold within the same window. TP matching uses a ±10-frame symmetric window around ground truth.

**`compute_metric_early_predict`** (causal, predictive): Uses a 10-frame backward-looking window. Reports the first frame index of each detected fall cluster (earliest prediction). TP matching is one-sided: a detection counts as TP if it occurs at or before `GT + 5 frames`, rewarding early detection. Provides two ROC sweep methods: `cal_roc_anomaly` (sweeps anomaly threshold, fixed centroid threshold=0.6) and `cal_roc_centroid` (sweeps centroid threshold, fixed anomaly=5).

`detect_falls` supports two detection methods via `method=` parameter:
- `method='drop'` (default): Original behavior — checks if centroidZ dropped by threshold within the backward window.
- `method='velocity'`: Uses centroidZ velocity to estimate where the person will be `prediction_gap` frames in the future: `estimated_Z = centroidZ[end_win] + centroidVz[end_win] * prediction_gap * frame_duration`. Two velocity sources are compared: radar hardware-reported centroidVz (rotation-corrected) and a numerical derivative of centroidZ_history via `numerical_velocity()`.

### Datasets

| Dataset | Contents | Use |
|---|---|---|
| DS0 | ~2h normal ADL, no falls | Training/testing |
| DS1 | Small mixed dataset (4 falls + other motions) | Illustration only |
| DS2 | 50 falls + 200 normal motions, labeled | ROC evaluation |

Multiple recording sessions can be merged with `combine.py` (adjusts ground truth frame indices automatically).

## Known Issues

- Second ROC figure in Cell 6 of the notebook has `lable=` (typo) instead of `label=` on one `plt.plot` call.
- `data_pre.py` uses the old `from keras.losses import mse` import (removed in newer Keras) and the deprecated `lr=` optimizer keyword — this is the older script version predating the notebook's fixes.

## General Notes and Instructions

### Notebook Editing — MANDATORY
When reading, editing, or making ANY changes to `.ipynb` files, you MUST use the `read_and_update_ipynb` skill. Do NOT use the built-in `NotebookEdit` tool or read `.ipynb` JSON directly. Always invoke the skill first. This applies to `src/mmfall.ipynb` and any other notebook in the project.

