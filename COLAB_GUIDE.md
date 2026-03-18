# Running mmPredict on Google Colab

## 1. Upload the Repository to Google Drive

Upload the entire `mmpredict/` project folder to your Google Drive. The expected structure:

```
My Drive/
  mmpredict/
    data/
      DS0/DS0.npy
      DS1/DS1_4falls.npy, DS1_4normal.npy, ...
      DS2/DS2.npy, DS2.csv, ...
      normal_train_data.npy
      normal_test_data.npy
    saved_model/
      HVRAE_mdl.h5
      HVRAE_SL_mdl.h5
      RAE_mdl.h5
    src/
      mmpredict.ipynb
```

## 2. Setup Cell (Mount Drive + Keras Compatibility)

Add a new **first cell** at the top of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

# Install Keras 2 compatibility layer for TF 2.16+
# This is needed because Colab ships TF 2.16+ with Keras 3,
# which breaks the standalone `from keras.*` imports and .h5 weight loading.
!pip install tf_keras
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

> **Why `tf_keras`?** Colab runs Python 3.12+ which only supports TF 2.16+. TF 2.16+ ships Keras 3 by default, which is incompatible with the notebook's standalone `from keras.*` imports and `.h5` weight files. Setting `TF_USE_LEGACY_KERAS=1` tells TensorFlow to use the Keras 2 API, so all existing imports and model loading work without modification.

> **Important:** The `os.environ` line must run **before** `import tensorflow` or any `from keras.*` imports. Place it in the very first cell.

## 3. Change `project_path`

The notebook defines `project_path` as a local absolute path. Update it to point to your Drive location.

**Find (in the cell that sets `project_path`):**
```python
project_path = '/Users/idoyanay/projects/personal_projects/mmpredict/'
```

**Replace with:**
```python
project_path = '/content/drive/MyDrive/mmpredict/'
```

Adjust the path if you placed the folder somewhere other than the Drive root.

This single change propagates to all data loading and model loading paths in the notebook, since they are all built relative to `project_path`:
- `data/DS1/DS1_4falls.npy`
- `data/DS1/DS1_4normal.npy`
- `data/DS2/DS2.npy` and `DS2.csv`
- `saved_model/HVRAE_mdl.h5`, `HVRAE_SL_mdl.h5`, `RAE_mdl.h5`

## 4. GPU/TPU Accelerator

After opening the notebook in Colab:

1. Go to **Runtime > Change runtime type**
2. Select **GPU** (T4, A100, etc.) or **TPU**
3. Click **Save**

No code changes are needed to use the GPU — TensorFlow automatically detects and uses it. You can verify with:

```python
print(tf.config.list_physical_devices('GPU'))
```

### If using TPU

TPU requires a distribution strategy wrapper. Add this before model creation:

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = autoencoder_mdl(model_dir=project_path + 'saved_model/')
```

> TPU does not support `.h5` weight files well. You may need to convert saved weights to the SavedModel format or retrain on the TPU.

## 5. Imports (Fallback — Only If `tf_keras` Doesn't Work)

If for some reason `TF_USE_LEGACY_KERAS=1` doesn't resolve import errors, manually replace all standalone Keras imports:

```python
# Old
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Reshape, \
    TimeDistributed, LSTM, RepeatVector, SimpleRNN, Activation
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.losses import mean_squared_error as mse
from keras.utils import plot_model

# New
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Reshape, \
    TimeDistributed, LSTM, RepeatVector, SimpleRNN, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.utils import plot_model
```

> Note: `plot_model` requires `pydot` and `graphviz` (already available on Colab).

## 6. Training (Optional)

If you want to retrain instead of using the pre-trained weights, uncomment the training cells in the notebook. The relevant commented-out lines are:

```python
# train_data, test_data = data_preproc().load_bin(project_path + 'data/DS0/DS0.npy', fortrain=True)
# np.save(project_path + 'data/normal_train_data', np.array(train_data))
# np.save(project_path + 'data/normal_test_data', np.array(test_data))

# train_data = np.load(project_path + 'data/normal_train_data.npy', allow_pickle=True)
# test_data  = np.load(project_path + 'data/normal_test_data.npy', allow_pickle=True)
# model = autoencoder_mdl(model_dir=(project_path + 'saved_model/'))
# model.HVRAE_train(train_data, test_data)
```

Make sure `data/DS0/DS0.npy` is uploaded to Drive if you want to preprocess from raw data, or that `normal_train_data.npy` / `normal_test_data.npy` are uploaded if you want to skip preprocessing.

## 7. Summary of All Required Changes

| What | Where | Change |
|---|---|---|
| Mount Drive | New first cell | `drive.mount('/content/drive')` |
| Keras 2 compat | Same first cell, before any imports | `!pip install tf_keras` + `os.environ["TF_USE_LEGACY_KERAS"] = "1"` |
| `project_path` | Cell that defines it | Change to `'/content/drive/MyDrive/mmpredict/'` |
| Runtime type | Colab menu | Runtime > Change runtime type > GPU |
