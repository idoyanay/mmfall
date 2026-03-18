# mmPredict — Early Fall Prediction using 4D MmWave Radar

mmPredict extends the [mmFall](https://github.com/radar-lab/mmfall) project (paper: [arXiv:2003.02386](https://arxiv.org/abs/2003.02386)) from post-hoc fall **detection** to early fall **prediction**. The base project detects falls after they happen using a centered-window algorithm; mmPredict predicts falls before they complete using velocity-based extrapolation and causal anomaly detection.

## What mmFall provides (base project)

- 4D mmWave radar data pipeline (TI AWR1843BOOST point cloud → motion patterns)
- HVRAE (Hybrid Variational RNN AutoEncoder) trained on normal ADL only (semi-supervised)
- `compute_metric`: post-hoc fall detection using a centered 20-frame window
- Datasets DS0/DS1/DS2

## What mmPredict adds

- **`compute_metric_early_predict`** — causal, predictive evaluator using a backward-looking 10-frame window. Reports the earliest frame of each detected fall. TP matching rewards early detection: a detection counts as TP if it occurs at or before `GT + 5 frames`.
- **Velocity-based extrapolation** (`method='velocity'`) — uses centroidZ velocity to predict where the person will be `prediction_gap` frames in the future, enabling detection before the fall completes.
- **Weighted anomaly detection** (`weighted=True`) — lowers the effective anomaly threshold proportionally to the severity of the predicted fall, improving sensitivity to large predicted drops.

See [src/compute_metric_early_predict.md](src/compute_metric_early_predict.md) for the full algorithm description.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tensorflow keras
```

`requirements.txt` pins data/plotting dependencies (numpy, matplotlib, h5py, etc.). TensorFlow/Keras must be installed separately.

## Running

The main entry point is the Jupyter notebook:
```bash
jupyter notebook src/mmpredict.ipynb
```

`src/mmpredict.py` is an auto-generated script version of the same notebook code.

Utility scripts:
```bash
python combine.py --filedir data/DS2/         # merge multiple .npy recordings + .csv ground truth
python data_visualizer.py --gtfile <raw.npy> --binfile <processed.npy>  # 3D animated point cloud
python data_analyzer.py --test <test.npy> --prediction <pred.npy> --raw <raw.npy>  # point-by-point inspection
```

## Models

All three models share input shape `(10, 64, 4)` — 10 frames, 64 points, 4 features (`delta_x, delta_y, z, Doppler`).

| Model | Architecture | Loss | Output |
|---|---|---|---|
| **HVRAE** (proposed) | VAE + SimpleRNN encoder-decoder, decoder outputs (mean, log_var) | Gaussian reconstruction log-likelihood + KL divergence | (10, 64, 8) |
| **HVRAE_SL** (baseline) | Same as HVRAE, decoder outputs mean only | MSE + KL divergence | (10, 64, 4) |
| **RAE** (baseline) | Standard recurrent autoencoder, no variational components | MSE | (10, 64, 4) |

Trained with Adam (lr=0.001), 5 epochs, batch size 8. Saved weights in `saved_model/`.

## Datasets

| Dataset | Contents | Use |
|---|---|---|
| DS0 | ~2h normal ADL, no falls | Training/testing (80/20 split) |
| DS1 | 4 falls + other motions | Illustration |
| DS2 | 50 falls + 200 normal motions, labeled | ROC evaluation |

## Fall Detection Methods

### Base: `compute_metric` (post-hoc)

The original algorithm from the [base mmFall project](https://github.com/radar-lab/mmfall). Uses a 20-frame centered (symmetric) window for both centroidZ drop detection and anomaly matching. A fall is detected when a height drop >= 0.6 m and an anomaly spike above threshold co-occur in the same window. TP matching uses a symmetric +-10-frame window around ground truth.

### Early prediction: `compute_metric_early_predict` (causal)

Adapts the base algorithm into a causal, predictive detector:

- **Backward-looking 10-frame window** instead of centered 20-frame window
- Anomaly spike must occur **before** the centroidZ drop, offset by a configurable `prediction_gap`
- TP matching is one-sided: detection must occur at or before the fall to count as TP

Two detection methods via `method=` parameter:
- `method='drop'`: checks if centroidZ dropped by threshold within the backward window (same logic as base, but causal)
- `method='velocity'`: uses centroidZ velocity to **predict** where the person will be `prediction_gap` frames in the future, enabling earlier detection by extrapolating current downward motion

### Weighted anomaly detection (`method='velocity'` only)

When `weighted=True`, the effective anomaly threshold is lowered proportionally to the severity of the predicted fall:

```
Z_ref = centroidZ_history[start_win]       # standing height before the fall
estimated_Z = centroidZ[j] + centroidVz[j] * prediction_gap * frame_duration

ratio = max(0, (Z_ref - estimated_Z) / Z_ref)
weight = 1 + scale * ratio^power
effective_threshold = threshold / weight
```

A larger predicted drop from the person's own starting height produces a higher weight, lowering the anomaly threshold needed to confirm the fall. Parameters `scale` and `power` control the strength and shape of this relationship.

## Results

The base mmFall project achieves 98% detection out of 50 falls with just 2 false alarms using the post-hoc `compute_metric` evaluator.

The notebook includes ROC curves comparing all three models across:
- Anomaly threshold sweeps (fixed centroidZ threshold)
- CentroidZ threshold sweeps (fixed anomaly threshold)
- Drop vs velocity-based detection at various prediction gaps
- Weighted vs unweighted anomaly detection with various (power, scale) configurations
