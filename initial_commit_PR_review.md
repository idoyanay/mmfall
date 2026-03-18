# PR Review: Initial Commit → Current Version
**File:** `src/mmpredict.ipynb`
**Compared commits:** `5e41df0` (first commit) → current working version
**Date:** 2026-03-10

---

## 1. Key Algorithmic Differences

### 1.1 Model Rename: VRAE → HVRAE

The primary model was renamed from **VRAE** (Variational Recurrent Autoencoder) to **HVRAE** throughout every method, variable, saved file name, and comment:

| Original | Current |
|---|---|
| `VRAE_train` | `HVRAE_train` |
| `VRAE_predict` | `HVRAE_predict` |
| `VRAE_mdl.h5` | `HVRAE_mdl.h5` |
| `VRAE_loss_history` | `HVRAE_loss_history` |
| `VRAE_SL_*` | `HVRAE_SL_*` |

The architecture itself is structurally identical — this is a nomenclature change only.

---

### 1.2 Reparameterization Trick: Keras Backend → Native TensorFlow

The `sampling` function was rewritten to use TensorFlow directly, replacing the Keras backend API:

```python
# Original (Keras backend — static shapes):
batch_size  = K.shape(Z_mean)[0]
n_frames    = K.int_shape(Z_mean)[1]      # returns Python int (static)
n_latentdim = K.int_shape(Z_mean)[2]
epsilon     = K.random_normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0, seed=None)
Z           = Z_mean + K.exp(0.5*Z_log_var) * epsilon
return Z

# Current (native TF — fully dynamic shapes):
shape       = tf.shape(Z_mean)            # returns a tensor (dynamic)
batch_size, n_frames, n_latentdim = shape[0], shape[1], shape[2]
epsilon     = tf.random.normal(shape=(batch_size, n_frames, n_latentdim), mean=0., stddev=1.0)
return Z_mean + tf.exp(0.5 * Z_log_var) * epsilon
```

`tf.shape()` is fully dynamic and more robust to unknown batch dimensions at graph construction time. The old code also had a misleading comment `# For reproducibility, we set the seed=37` while passing `seed=None` — this was removed.

The `Lambda` layer in `_build_HVRAE_architecture` now also specifies an explicit `output_shape`:
```python
Z = Lambda(sampling, output_shape=(n_frames, n_latentdim))([Z_mean, Z_log_var])
```

---

### 1.3 Rotation Matrix (Data Preprocessing) — No Change

The tilt angle sign and the off-diagonal signs in the rotation matrix both changed, but the two changes cancel out exactly — the resulting matrix is **numerically identical**:

```python
# Original: tilt_angle = -10.0, signs [cos, -sin / sin, cos]
# cos(-10°) = 0.9848,  -sin(-10°) = +0.1736  →  [[0.9848, 0.1736], [-0.1736, 0.9848]]

# Current:  tilt_angle = +10.0, signs [cos,  sin / -sin, cos]
# cos(10°)  = 0.9848,   sin(10°)  = +0.1736  →  [[0.9848, 0.1736], [-0.1736, 0.9848]]
```

The coordinate transformation applied to every radar point is unchanged.

---

### 1.4 Model Loading Strategy: `load_model` → `build + load_weights`

The original loaded saved models using `load_model` with custom object resolution. The current version instead rebuilds the architecture from code and loads only the weights:

```python
# Original:
model = load_model('HVRAE_mdl.h5', compile=False,
                   custom_objects={'sampling': sampling_predict, 'tf': tf})

# Current:
model = self._build_HVRAE_architecture()[0]   # rebuild graph from code
model.load_weights('HVRAE_mdl.h5')            # fill with trained weights
```

This approach avoids the fragile custom object resolution required by `load_model` for Lambda layers.

> **Bug introduced:** `_build_RAE_architecture()` returns a bare `Model`, not a tuple, but `RAE_predict` calls `[0]` on it — this will raise a `TypeError` at runtime.

---

### 1.5 New Class: `compute_metric_predict`

A completely new evaluation class was added implementing a **causal/predictive** fall detection paradigm, in contrast to the centered-window approach of `compute_metric`:

| Aspect | `compute_metric` (original) | `compute_metric_predict` (new) |
|---|---|---|
| Window length | 20 frames (2 sec) | 10 frames (1 sec) |
| Window type | Centered around detected fall | Causal — looks backward from current frame |
| Fall index reported | Midpoint of fall cluster | First detected index (earliest prediction) |
| TP matching | Symmetric `±win/2` around ground truth | One-sided: detection ≤ `GT + win/2` (rewards early detection) |
| ROC methods | Single sweep on anomaly threshold | `cal_roc_anomaly` (sweep anomaly thr, fixed centroid thr=0.6) and `cal_roc_centroid` (sweep centroid thr, fixed anomaly thr=5) |

Key logic difference in `detect_falls`:
```python
# Original (centered window):
lf_edge, rh_edge = i - win_len/2, i + win_len/2
if centroidZ_history[lf_edge] - centroidZ_history[rh_edge] >= 0.6: ...

# Current (causal window):
start_win, end_win = i - (win_len-1), i
if centroidZ_history[start_win] - centroidZ_history[end_win] >= centroidZ_thr: ...
```

Key logic difference in `find_tpfpfn` (TP matching criterion):
```python
# Original (symmetric window around GT):
if int(falls_fn[j] - win_len/2) <= detected_falls_idx[i] <= int(falls_fn[j] + win_len/2): ...

# Current (one-sided — rewards early detection):
if detected_falls_idx[i] <= int(falls_fn[j] + win_len/2): ...
```

---

### 1.6 Optimizer Keyword Fix

```python
# Original (deprecated parameter name):
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Current (correct for newer Keras):
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
```

Applied to all three model trainers and both predict methods.

---

### 1.7 MSE Import Fix

```python
# Original (short alias removed in newer Keras):
from keras.losses import mse

# Current (explicit import with local alias):
from keras.losses import mean_squared_error as mse
```

---

## 2. Other Differences

### 2.1 Architecture Builder Refactoring

The original duplicated full layer-by-layer model construction inline inside each `_train` and `_predict` method. The current extracts this into three new shared private helpers:

```python
def _build_HVRAE_architecture(self)    -> (Model, Z_log_var, Z_mean)
def _build_HVRAE_SL_architecture(self) -> (Model, Z_log_var, Z_mean)
def _build_RAE_architecture(self)      -> Model
```

`_train` and `_predict` methods now delegate to these builders instead of repeating construction code.

A large commented-out block of the old inline `HVRAE_train` implementation remains in the file as a migration artifact. It contains a draft typo: `self.HHVRAE_mdl.summary()` (double H), confirming it was a work-in-progress intermediate draft.

---

### 2.2 `Flatten` Argument Inconsistency

Only `_build_HVRAE_architecture` uses `Flatten(data_format='channels_last')`, while the other two builders use `Flatten(None)` — inconsistent within the same file.

---

### 2.3 Platform Migration: Google Colab → Local Windows

```python
# Original (Google Colab):
from google.colab import drive
drive.mount('/content/drive')
project_path = '/content/drive/My Drive/Colab/'

# Current (local Windows, drive mount commented out):
# from google.colab import drive
# drive.mount('/content/drive')
project_path = 'C:/Users/idosy/uni_projects/mmfall/'
```

---

### 2.4 Dataset Naming: D0/D1/D2 → DS0/DS1/DS2

All dataset directory references updated:

| Original | Current |
|---|---|
| `data/D0/D0.npy` | `data/DS0/DS0.npy` |
| `data/D1/DS1_4falls` | `data/DS1/DS1_4falls` |
| `data/D2/D2` | `data/DS2/DS2` |

---

### 2.5 Debugging Code Added (Commented Out)

Cell 3 contains a new commented-out diagnostic block that was not in the original, likely used to debug the broken `load_model` call that led to the `load_weights` switch:

```python
## our edit
# path = r'C:/Users/idosy/uni_projects/mmfall/saved_model/HVRAE_mdl.h5'
# import os
# print(f"File size: {os.path.getsize(path)} bytes")
# with open(path, 'rb') as f:
#     header = f.read(50)
#     print(f"Header: {header}")
```

---

### 2.6 Cell 6 Extended with Predictive ROC Evaluation

Cell 6 now additionally calls `compute_metric_predict` and plots a second ROC figure comparing anomaly-threshold sweep vs centroid-threshold sweep:

```python
calculator = compute_metric_predict()
HVRAE_tpr_predict_anomaly, HVRAE_fp_total_predict_anomaly   = calculator.cal_roc_anomaly(...)
HVRAE_tpr_predict_centroid, HVRAE_fp_total_predict_centroid = calculator.cal_roc_centroid(...)
print(f"------- tpr predict: {HVRAE_tpr_predict_anomaly}")
print(f"------- fp predict:  {HVRAE_fp_total_predict_anomaly}")
```

> **Bug:** the second ROC plot has a typo on one `plt.plot` call — `lable=` instead of `label=`, so the legend entry will be silently ignored.

---

## Summary Table

| Aspect | Original (5e41df0) | Current |
|---|---|---|
| Primary model name | VRAE | HVRAE |
| Baseline 1 name | VRAE_SL | HVRAE_SL |
| Architecture builders | Inline in each method | Extracted to `_build_*` helpers |
| Sampling RNG API | `K.random_normal`, `K.int_shape` (static) | `tf.random.normal`, `tf.shape` (dynamic) |
| Optimizer keyword | `lr=` (deprecated) | `learning_rate=` |
| `mse` import | `from keras.losses import mse` | `from keras.losses import mean_squared_error as mse` |
| Model loading | `load_model` with custom objects | `build architecture + load_weights` |
| Tilt angle | −10.0° (with matching signs) | +10.0° (with matching signs) — **numerically identical matrix** |
| Platform | Google Colab | Local Windows |
| Dataset naming | D0/D1/D2 | DS0/DS1/DS2 |
| Predictive detection class | Not present | New `compute_metric_predict` |
| Predictive ROC methods | Not present | `cal_roc_anomaly`, `cal_roc_centroid` |
| Detection window (predict) | N/A | 10 frames, causal |
| TP matching criterion | Symmetric `±win/2` | One-sided (rewards early detection) |
| `RAE_predict` load | `load_model(...)` | `_build_RAE_architecture()[0]` — **bug: `[0]` invalid** |
| Second ROC figure | Not present | Added (has `lable=` typo) |
