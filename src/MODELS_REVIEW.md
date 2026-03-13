# Models Review: HVRAE vs HVRAE_SL vs RAE

## Overview

This project implements three autoencoder models for fall detection from 4D mmWave radar data. All three share the same input shape `(10, 64, 4)` — 10 frames, 64 points per frame, 4 features per point `(delta_x, delta_y, z, Doppler)`. They are trained exclusively on normal activities (ADL) and detect falls as anomalies at inference time.

The models differ in two key dimensions:
1. **Whether the encoder/decoder uses variational inference** (VAE components)
2. **Whether the reconstruction distribution models variance** (Gaussian log-likelihood vs MSE)

| Property | HVRAE (proposed) | HVRAE_SL (baseline 1) | RAE (baseline 2) |
|---|---|---|---|
| VAE encoder (q(z\|X)) | Yes | Yes | No |
| Reparameterization trick | Yes | Yes | No |
| KL divergence in loss | Yes | Yes | No |
| Decoder outputs log_var | Yes | **No** | No |
| Reconstruction loss | Gaussian log-likelihood | MSE | MSE |
| Output shape | (10, 64, **8**) | (10, 64, **4**) | (10, 64, **4**) |

---

## 1. HVRAE — Hybrid Variational RNN AutoEncoder (Proposed Model)

### Algorithm

The HVRAE is the full model proposed in the paper. It combines a Variational Autoencoder (VAE) with a Recurrent Neural Network (RNN) autoencoder in a hybrid architecture:

1. **VAE Encoder q(z|X):** The input is flattened per frame, passed through a dense layer, and projected into two vectors — `Z_mean` and `Z_log_var` — representing the parameters of the approximate posterior distribution q(z|X).
2. **Reparameterization trick:** Instead of sampling directly from q(z|X), the model samples `epsilon ~ N(0, I)` and computes `z = Z_mean + exp(0.5 * Z_log_var) * epsilon`. This makes the sampling differentiable.
3. **RNN Autoencoder:** The sampled latent sequence `z` is compressed by a SimpleRNN encoder into a single vector, then expanded back to a sequence by a SimpleRNN decoder (with time-reversal).
4. **VAE Decoder p(X|z):** The decoded sequence is projected back through dense layers into **two** output vectors per frame: `pXz_mean` and `pXz_logvar`, modeling a full Gaussian distribution over the reconstructed data.

### Loss Function

The HVRAE loss has two terms:

- **Reconstruction term (Gaussian log-likelihood):** Measures how well the predicted distribution p(X|z) explains the true data. Because the decoder outputs both mean and variance, the reconstruction error is weighted by the predicted variance: `sum(0.5 * (y_true - mean)^2 / var)`. This allows the model to express uncertainty — high-confidence predictions are penalized more for errors.
- **KL divergence:** Regularizes the latent space by pushing q(z|X) toward the prior p(z) = N(0, I): `-0.5 * sum(1 + Z_log_var - Z_mean^2 - exp(Z_log_var))`.

Total loss: `mean(log_pXz + kl_loss)` averaged over batches.

### Why It Matters

By modeling both mean **and** variance of the reconstruction, the HVRAE can capture not just "what the normal pattern looks like" but also "how certain the model is about it." Fall events produce high anomaly scores because they are both poorly reconstructed AND fall outside the learned variance envelope.

---

## 2. HVRAE_SL — HVRAE with Simplified Loss (Baseline 1)

### Algorithm

The architecture is nearly identical to HVRAE — it has the same VAE encoder with reparameterization and the same RNN autoencoder core. The critical difference is in the **decoder output**:

- The decoder outputs only `pXz_mean` (no `pXz_logvar`).
- The output shape is `(10, 64, 4)` instead of `(10, 64, 8)`.

### Loss Function

- **Reconstruction term (MSE):** Without a predicted variance, the model falls back to standard mean squared error between the true data and the predicted mean. This implicitly assumes a fixed, uniform variance across all outputs.
- **KL divergence:** Same as HVRAE — regularizes q(z|X) toward N(0, I).

Total loss: `mean(MSE + kl_loss)` averaged over batches.

### Why "Simplified Loss"

The "SL" stands for "Simplified Loss." By removing the learned variance from the decoder, the reconstruction term simplifies from a weighted Gaussian log-likelihood to plain MSE. The model can still learn a structured latent space (thanks to VAE + KL), but it loses the ability to express per-output uncertainty.

---

## 3. RAE — Recurrent AutoEncoder (Baseline 2)

### Algorithm

The RAE is a standard (non-variational) recurrent autoencoder. There is no VAE component at all:

1. **Deterministic Encoder:** Input is flattened, passed through two dense layers (64 -> 16 dims), then encoded by a SimpleRNN into a single vector.
2. **Deterministic Decoder:** The encoded vector is repeated across frames, decoded by a SimpleRNN (with time-reversal), and projected back through two dense layers to the original feature space.

There is no sampling, no reparameterization trick, and no latent distribution.

### Loss Function

Standard **MSE** between input and reconstruction. No KL divergence term.

### Simplicity vs Expressiveness

The RAE is the simplest of the three models. It learns a deterministic mapping from input to latent space and back. Without the variational component, it cannot model the distribution of normal activities — it only learns a point estimate of the "average" reconstruction.

---

## Code Differences

### Encoder: VAE vs Deterministic

**HVRAE and HVRAE_SL** — VAE encoder producing mean and log-variance, with reparameterization sampling:

```python
# mmfall.py lines 180-194 (HVRAE), lines 359-373 (HVRAE_SL) — identical
input_flatten           = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
Z_mean                  = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_mean')(input_flatten)
Z_log_var               = TimeDistributed(Dense(n_latentdim, activation=None), name='qzx_log_var')(input_flatten)

def sampling(args):
    Z_mean, Z_log_var   = args
    epsilon             = K.random_normal(shape=(...), mean=0., stddev=1.0)
    Z                   = Z_mean + K.exp(0.5*Z_log_var) * epsilon  # reparameterization trick
    return Z

Z                       = Lambda(sampling)([Z_mean, Z_log_var])
```

**RAE** — deterministic encoder, no sampling:

```python
# mmfall.py lines 517-519
encoder_feature         = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(input_flatten)
encoder_feature         = TimeDistributed(Dense(n_latentdim, activation='tanh'))(encoder_feature)
```

Note: RAE uses `activation='tanh'` on the latent projection, while HVRAE/HVRAE_SL use `activation=None` (linear) since they produce distribution parameters, not features.

### RNN Core

All three models share the same RNN autoencoder structure:

```python
# Identical in all three (mmfall.py lines 197-200, 376-379, 522-525)
encoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=False)(...)
decoder_feature         = RepeatVector(n_frames)(encoder_feature)
decoder_feature         = SimpleRNN(n_latentdim, activation='tanh', return_sequences=True)(decoder_feature)
decoder_feature         = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_feature)
```

### Decoder Output: Full Gaussian vs Mean-Only vs Direct Reconstruction

**HVRAE** — outputs both mean and log-variance, concatenated into 8 features per point:

```python
# mmfall.py lines 202-211
X_latent                = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
pXz_mean                = TimeDistributed(Dense(n_features, activation=None))(X_latent)
pXz_logvar              = TimeDistributed(Dense(n_features, activation=None))(X_latent)

pXz                     = Concatenate()([pXz_mean, pXz_logvar])       # 4+4 = 8 features
pXz                     = TimeDistributed(RepeatVector(n_points))(pXz)
outputs                 = TimeDistributed(Reshape((n_points, n_features*2)))(pXz)  # shape: (10, 64, 8)
```

**HVRAE_SL** — outputs only mean, 4 features per point:

```python
# mmfall.py lines 381-388
X_latent                = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
pXz_mean                = TimeDistributed(Dense(n_features, activation=None))(X_latent)

pXz_mean                = TimeDistributed(RepeatVector(n_points))(pXz_mean)
outputs                 = TimeDistributed(Reshape((n_points, n_features)))(pXz_mean)  # shape: (10, 64, 4)
```

**RAE** — direct reconstruction through dense layers:

```python
# mmfall.py lines 528-532
decoder_feature         = TimeDistributed(Dense(n_intermidiate, activation='tanh'))(decoder_feature)
decoder_feature         = TimeDistributed(Dense(n_points*n_features, activation='tanh'))(decoder_feature)
outputs                 = TimeDistributed(Reshape((n_points, n_features)))(decoder_feature)  # shape: (10, 64, 4)
```

Key difference: RAE's final dense layer outputs `n_points * n_features = 256` values directly (with `tanh` activation), producing a per-point reconstruction. HVRAE/HVRAE_SL output a single vector per frame that is then broadcast to all 64 points via `RepeatVector` — every point in a frame shares the same predicted mean (and variance in HVRAE).

### Loss Functions

**HVRAE** — Gaussian log-likelihood + KL:

```python
# mmfall.py lines 218-241
mean   = y_pred[:, :, :, :n_features]      # first 4 channels
logvar = y_pred[:, :, :, n_features:]       # last 4 channels
var    = K.exp(logvar)

log_pXz  = K.sum(0.5 * K.square(y_true - mean) / var, axis=-1)   # variance-weighted
kl_loss  = -0.5 * K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)
loss     = K.mean(log_pXz + kl_loss)
```

**HVRAE_SL** — MSE + KL:

```python
# mmfall.py lines 395-416
mean = y_pred  # output IS the mean (no logvar to split)

log_pXz  = mse(y_true_reshape, mean)         # standard MSE
kl_loss  = -0.5 * K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)
loss     = K.mean(log_pXz + kl_loss)
```

**RAE** — plain MSE (built-in):

```python
# mmfall.py lines 539-541
self.RAE_mdl.compile(optimizer=adam, loss=mse)  # no custom loss function
```

### Inference

**HVRAE and HVRAE_SL** require extracting `Z_mean` and `Z_log_var` from intermediate layers to compute the full loss (including KL divergence) at inference time:

```python
# mmfall.py lines 297-298 (HVRAE), lines 458-459 (HVRAE_SL)
get_z_mean_model    = Model(inputs=model.input, outputs=model.get_layer('qzx_mean').output)
get_z_log_var_model = Model(inputs=model.input, outputs=model.get_layer('qzx_log_var').output)

# Per-pattern loop computes custom loss
current_loss = HVRAE_loss(pattern, current_prediction, predicted_z_mean, predicted_z_log_var)
```

**RAE** uses the built-in MSE loss directly via `test_on_batch`:

```python
# mmfall.py lines 564-567
for pattern in inferencedata:
    pattern      = np.expand_dims(pattern, axis=0)
    current_loss = model.test_on_batch(pattern, pattern)  # built-in MSE
```

---

## Summary

| Aspect | HVRAE | HVRAE_SL | RAE |
|---|---|---|---|
| Encoder type | Variational (mean + log_var + sampling) | Variational (mean + log_var + sampling) | Deterministic (dense layers) |
| Decoder output | mean + log_var per feature | mean only | direct reconstruction |
| Output channels | 8 (4 mean + 4 log_var) | 4 (mean) | 4 (reconstruction) |
| Per-point output | Shared across all 64 points (RepeatVector) | Shared across all 64 points (RepeatVector) | Unique per point (Dense -> 256) |
| Reconstruction loss | Gaussian log-likelihood (variance-weighted) | MSE | MSE |
| KL regularization | Yes | Yes | No |
| Inference complexity | Custom numpy loss + intermediate layer extraction | Custom numpy loss + intermediate layer extraction | Built-in `test_on_batch` |
| Final activation | None (linear, for distribution params) | None (linear, for mean) | tanh |

The HVRAE is the most expressive model — it learns a structured latent space (via VAE) and models reconstruction uncertainty (via learned variance). HVRAE_SL ablates the learned variance to test whether the Gaussian log-likelihood provides benefit over MSE. RAE ablates the entire variational framework to test whether the probabilistic latent space matters at all.
