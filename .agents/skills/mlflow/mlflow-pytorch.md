# mlflow.pytorch

The `mlflow.pytorch` module provides an API for logging and loading PyTorch models. This module exports PyTorch models with the following flavors:

- **PyTorch (native) format**: This is the main flavor that can be loaded back into PyTorch.
- **mlflow.pyfunc**: Produced for use by generic pyfunc-based deployment tools and batch inference.

## Classes and Functions

### MlflowModelCheckpointCallback

```python
class mlflow.pytorch.MlflowModelCheckpointCallback(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch'
)
```

Callback for auto-logging pytorch-lightning model checkpoints to MLflow. This callback implementation only supports pytorch-lightning >= 1.6.0.

**Bases**: `pytorch_lightning.callbacks.callback.Callback`, `mlflow.utils.checkpoint_utils.MlflowModelCheckpointCallbackBase`

#### Parameters

- **monitor**: In automatic model checkpointing, the metric name to monitor if you set `model_checkpoint_save_best_only` to True.
- **save_best_only**: If True, automatic model checkpointing only saves when the model is considered the "best" model according to the quantity monitored and previous checkpoint model is overwritten.
- **mode**: one of {"min", "max"}. In automatic model checkpointing, if save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity.
- **save_weights_only**: In automatic model checkpointing, if True, then only the model's weights will be saved. Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint too.
- **save_freq**: "epoch" or integer. When using "epoch", the callback saves the model after each epoch. When using integer, the callback saves the model at end of this many batches. Note that if the saving isn't aligned to epochs, the monitored metric may potentially be less reliable (it could reflect as little as 1 batch, since the metrics get reset every epoch). Defaults to "epoch".

#### Example

```python
import mlflow
from mlflow.pytorch import MlflowModelCheckpointCallback
from pytorch_lightning import Trainer

mlflow.pytorch.autolog(checkpoint=True)

model = MyLightningModuleNet()  # A custom-pytorch lightning model
train_loader = create_train_dataset_loader()

mlflow_checkpoint_callback = MlflowModelCheckpointCallback()

trainer = Trainer(callbacks=[mlflow_checkpoint_callback])

with mlflow.start_run() as run:
    trainer.fit(model, train_loader)
```

#### Methods

**on_fit_start(trainer, pl_module)**
- Called when fit begins.

**on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)**
- Called when the train batch ends.
- Note: The value `outputs["loss"]` here will be the normalized value w.r.t `accumulate_grad_batches` of the loss returned from `training_step`.

**on_train_epoch_end(trainer, pl_module)**
- Called when the train epoch ends.

To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the `pytorch_lightning.core.LightningModule` and access them in this hook:

```python
class MyLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []

    def training_step(self):
        loss = ...
        self.training_step_outputs.append(loss)
        return loss


class MyCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        pl_module.log("training_epoch_mean", epoch_mean)
        # free up the memory
        pl_module.training_step_outputs.clear()
```

**save_checkpoint(filepath)**
- Saves the model checkpoint at the specified filepath.

---

### autolog()

```python
mlflow.pytorch.autolog(
    log_every_n_epoch=1,
    log_every_n_step=None,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
    checkpoint=True,
    checkpoint_monitor='val_loss',
    checkpoint_mode='min',
    checkpoint_save_best_only=True,
    checkpoint_save_weights_only=False,
    checkpoint_save_freq='epoch',
    log_model_signatures=True
)
```

Enables (or disables) and configures autologging from PyTorch Lightning to MLflow.

Autologging is performed when you call the `fit` method of `pytorch_lightning.Trainer()`.

Explore the complete [PyTorch MNIST](https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST) example for an expansive example with implementation of additional lightning steps.

**Note**: Full autologging is only supported for PyTorch Lightning models, i.e., models that subclass `pytorch_lightning.LightningModule`. Autologging support for vanilla PyTorch (i.e., models that only subclass `torch.nn.Module`) only autologs calls to `torch.utils.tensorboard.SummaryWriter`'s `add_scalar` and `add_hparams` methods to mlflow. In this case, there's also no notion of an "epoch".

**Autologging Compatibility Note**: Autologging is known to be compatible with the following package versions: `2.2.2` <= `torch` <= `2.10.0`. Autologging may not succeed when used with package versions outside of this range.

#### Parameters

- **log_every_n_epoch**: If specified, logs metrics once every `n` epochs. By default, metrics are logged after every epoch.
- **log_every_n_step**: If specified, logs batch metrics once every `n` training step. By default, metrics are not logged for steps. Note that setting this to 1 can cause performance issues and is not recommended. Metrics are logged against Lightning's global step number, and when multiple optimizers are used it is assumed that all optimizers are stepped in each training step.
- **log_models**: If `True`, trained models are logged as MLflow model artifacts. If `False`, trained models are not logged.
- **log_datasets**: If `True`, dataset information is logged to MLflow Tracking. If `False`, dataset information is not logged.
- **disable**: If `True`, disables the PyTorch Lightning autologging integration. If `False`, enables the PyTorch Lightning autologging integration.
- **exclusive**: If `True`, autologged content is not logged to user-created fluent runs. If `False`, autologged content is logged to the active fluent run, which may be user-created.
- **disable_for_unsupported_versions**: If `True`, disable autologging for versions of pytorch and pytorch-lightning that have not been tested against this version of the MLflow client or are incompatible.
- **silent**: If `True`, suppress all event logs and warnings from MLflow during PyTorch Lightning autologging. If `False`, show all events and warnings during PyTorch Lightning autologging.
- **registered_model_name**: If given, each time a model is trained, it is registered as a new model version of the registered model with this name. The registered model is created if it does not already exist.
- **extra_tags**: A dictionary of extra tags to set on each managed run created by autologging.
- **checkpoint**: Enable automatic model checkpointing, this feature only supports pytorch-lightning >= 1.6.0.
- **checkpoint_monitor**: In automatic model checkpointing, the metric name to monitor if you set `model_checkpoint_save_best_only` to True.
- **checkpoint_mode**: one of {"min", "max"}. In automatic model checkpointing, if save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity.
- **checkpoint_save_best_only**: If True, automatic model checkpointing only saves when the model is considered the "best" model according to the quantity monitored and previous checkpoint model is overwritten.
- **checkpoint_save_weights_only**: In automatic model checkpointing, if True, then only the model's weights will be saved. Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint too.
- **checkpoint_save_freq**: "epoch" or integer. When using "epoch", the callback saves the model after each epoch. When using integer, the callback saves the model at end of this many batches. Note that if the saving isn't aligned to epochs, the monitored metric may potentially be less reliable (it could reflect as little as 1 batch, since the metrics get reset every epoch). Defaults to "epoch".
- **log_model_signatures**: Whether to log model signature when `log_model` is True.

#### Example

```python
import os

import lightning as L
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

import mlflow.pytorch
from mlflow import MlflowClient


class MNISTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


# Initialize our model.
mnist_model = MNISTModel()

# Load MNIST dataset.
train_ds = MNIST(
    os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)
# Only take a subset of the data for faster training.
indices = torch.arange(32)
train_ds = Subset(train_ds, indices)
train_loader = DataLoader(train_ds, batch_size=8)

# Initialize a trainer.
trainer = L.Trainer(max_epochs=3)

# Auto log all MLflow entities
mlflow.pytorch.autolog()

# Train the model.
with mlflow.start_run() as run:
    trainer.fit(mnist_model, train_loader)

# Fetch the auto logged parameters and metrics.
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
```

---

### get_default_conda_env()

```python
mlflow.pytorch.get_default_conda_env()
```

Gets the default Conda environment as a dictionary for MLflow Models produced by calls to `save_model()` and `log_model()`.

#### Returns

The default Conda environment as a dictionary for MLflow Models produced by calls to `save_model()` and `log_model()`.

#### Example

```python
import mlflow

# Log PyTorch model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, name="model", signature=signature)

# Fetch the associated conda environment
env = mlflow.pytorch.get_default_conda_env()
print(f"conda env: {env}")
```

#### Output

```
conda env {
    'name': 'mlflow-env',
    'channels': ['conda-forge'],
    'dependencies': ['python=3.8.15',
                     {'pip': ['torch==1.5.1',
                              'mlflow',
                              'cloudpickle==1.6.0']}]
}
```

---

### get_default_pip_requirements()

```python
mlflow.pytorch.get_default_pip_requirements()
```

Gets a list of default pip requirements for MLflow Models produced by this flavor. Calls to `save_model()` and `log_model()` produce a pip environment that, at minimum, contains these requirements.

#### Returns

A list of default pip requirements for MLflow Models produced by this flavor.

---

### load_checkpoint()

```python
mlflow.pytorch.load_checkpoint(
    model_class,
    run_id=None,
    epoch=None,
    global_step=None,
    kwargs=None
)
```

If you enable "checkpoint" in autologging, during pytorch-lightning model training execution, checkpointed models are logged as MLflow artifacts. Using this API, you can load the checkpointed model.

If you want to load the latest checkpoint, set both `epoch` and `global_step` to None. If "checkpoint_save_freq" is set to "epoch" in autologging, you can set `epoch` param to the epoch of the checkpoint to load specific epoch checkpoint. If "checkpoint_save_freq" is set to an integer in autologging, you can set `global_step` param to the global step of the checkpoint to load specific global step checkpoint. `epoch` param and `global_step` cannot be set together.

#### Parameters

- **model_class**: The class of the training model, the class should inherit 'pytorch_lightning.LightningModule'.
- **run_id**: The id of the run which model is logged to. If not provided, current active run is used.
- **epoch**: The epoch of the checkpoint to be loaded, if you set "checkpoint_save_freq" to "epoch".
- **global_step**: The global step of the checkpoint to be loaded, if you set "checkpoint_save_freq" to an integer.
- **kwargs**: Any extra kwargs needed to init the model.

#### Returns

The instance of a pytorch-lightning model restored from the specified checkpoint.

#### Example

```python
import mlflow

mlflow.pytorch.autolog(checkpoint=True)

model = MyLightningModuleNet()  # A custom-pytorch lightning model
train_loader = create_train_dataset_loader()
trainer = Trainer()

with mlflow.start_run() as run:
    trainer.fit(model, train_loader)

run_id = run.info.run_id

# load latest checkpoint model
latest_checkpoint_model = mlflow.pytorch.load_checkpoint(MyLightningModuleNet, run_id)

# load history checkpoint model logged in second epoch
checkpoint_model = mlflow.pytorch.load_checkpoint(MyLightningModuleNet, run_id, epoch=2)
```

---

### load_model()

```python
mlflow.pytorch.load_model(
    model_uri,
    dst_path=None,
    **kwargs
)
```

Load a PyTorch model from a local file or a run.

#### Parameters

- **model_uri**: The location, in URI format, of the MLflow model, for example:
  - `/Users/me/path/to/local/model`
  - `relative/path/to/local/model`
  - `s3://my_bucket/path/to/model`
  - `runs:/<mlflow_run_id>/run-relative/path/to/model`
  - `models:/<model_name>/<model_version>`
  - `models:/<model_name>/<stage>`

  For more information about supported URI schemes, see [Referencing Artifacts](https://www.mlflow.org/docs/latest/concepts.html#artifact-locations).

- **dst_path**: The local filesystem path to which to download the model artifact. This directory must already exist. If unspecified, a local output path will be created.
- **kwargs**: kwargs to pass to `torch.load` method.

#### Returns

A PyTorch model.

#### Example

```python
import torch
import mlflow.pytorch


model = nn.Linear(1, 1)

# Log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, name="model")

# Inference after loading the logged model
model_uri = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
for x in [4.0, 6.0, 30.0]:
    X = torch.Tensor([[x]])
    y_pred = loaded_model(X)
    print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}")
```

#### Output

```
predict X: 4.0, y_pred: 7.57
predict X: 6.0, y_pred: 11.64
predict X: 30.0, y_pred: 60.48
```

---

### log_model()

```python
mlflow.pytorch.log_model(
    pytorch_model,
    artifact_path: str | None = None,
    conda_env=None,
    code_paths=None,
    pickle_module=None,
    registered_model_name=None,
    signature: mlflow.models.signature.ModelSignature = None,
    input_example = None,
    await_registration_for=300,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    name: str | None = None,
    params: dict[str, typing.Any] | None = None,
    tags: dict[str, typing.Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    export_model: bool = False,
    serialization_format: Literal['pickle', 'pt2'] = 'pickle',
    **kwargs
)
```

Log a PyTorch model as an MLflow artifact for the current run.

**Warning**: Log the model with a signature to avoid inference errors. If the model is logged without a signature, the MLflow Model Server relies on the default inferred data type from NumPy. However, PyTorch often expects different defaults, particularly when parsing floats. You must include the signature to ensure that the model is logged with the correct data type so that the MLflow model server can correctly provide valid input.

#### Parameters

- **pytorch_model**: PyTorch model to be saved. Can be either an eager model (subclass of `torch.nn.Module`) or scripted model prepared via `torch.jit.script` or `torch.jit.trace`.

  The model accepts a single `torch.FloatTensor` as input and produces a single output tensor.

  If saving an eager model, any code dependencies of the model's class, including the class definition itself, should be included in one of the following locations:
  - The package(s) listed in the model's Conda environment, specified by the `conda_env` parameter.
  - One or more of the files specified by the `code_paths` parameter.

- **artifact_path**: Deprecated. Use `name` instead.
- **conda_env**: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. If provided, this describes the environment this model should be run in. At a minimum, it should specify the dependencies contained in `get_default_conda_env()`. If `None`, a conda environment with pip requirements inferred by `mlflow.models.infer_pip_requirements()` is added to the model. If the requirement inference fails, it falls back to using `get_default_pip_requirements()`. pip requirements from `conda_env` are written to a pip `requirements.txt` file and the full conda environment is written to `conda.yaml`. Example conda environment dictionary:

  ```python
  {
      "name": "mlflow-env",
      "channels": ["conda-forge"],
      "dependencies": [
          "python=3.8.15",
          {
              "pip": [
                  "torch==x.y.z"
              ],
          },
      ],
  }
  ```

- **code_paths**: A list of local filesystem paths to Python file dependencies (or directories containing file dependencies). These files are prepended to the system path when the model is loaded. Files declared as dependencies for a given model should have relative imports declared from a common root path if multiple files are defined with import dependencies between them to avoid import errors when loading the model.

  For a detailed explanation of `code_paths` functionality, recommended usage patterns and limitations, see the [code_paths usage guide](https://mlflow.org/docs/latest/model/dependencies.html?highlight=code_paths#saving-extra-code-with-an-mlflow-model).

- **pickle_module**: The module that PyTorch should use to serialize ("pickle") the specified `pytorch_model`. This is passed as the `pickle_module` parameter to `torch.save()`. By default, this module is also used to deserialize ("unpickle") the PyTorch model at load time.
- **registered_model_name**: If given, create a model version under `registered_model_name`, also create a registered model if one with the given name does not exist.
- **signature**: an instance of the `ModelSignature` class that describes the model's inputs and outputs. If not specified but an `input_example` is supplied, a signature will be automatically inferred based on the supplied input example and model. To disable automatic signature inference when providing an input example, set `signature` to `False`. To manually infer a model signature, call `infer_signature()` on datasets with valid model inputs, such as a training dataset with the target column omitted, and valid model outputs, like model predictions made on the training dataset, for example:

  ```python
  from mlflow.models import infer_signature

  train = df.drop_column("target_label")
  predictions = ...  # compute model predictions
  signature = infer_signature(train, predictions)
  ```

- **input_example**: one or several instances of valid model input. The input example is used as a hint of what data to feed the model. It will be converted to a Pandas DataFrame and then serialized to json using the Pandas split-oriented format, or a numpy array where the example will be serialized to json by converting it to a list. Bytes are base64-encoded. When the `signature` parameter is `None`, the input example is used to infer a model signature.
- **await_registration_for**: Number of seconds to wait for the model version to finish being created and is in `READY` status. By default, the function waits for five minutes. Specify 0 or None to skip waiting.
- **extra_files**: A list containing the paths to corresponding extra files, if `None`, no extra files are added to the model. Remote URIs are resolved to absolute filesystem paths. For example, consider the following `extra_files` list:

  ```python
  extra_files = ["s3://my-bucket/path/to/my_file1", "/local-path/to/my_file2"]
  ```

  In this case, the `"my_file1"` extra file is downloaded from S3. Model paths will be ["extra_files/my_file1", "extra_files/my_file2"] in the model directory.

- **pip_requirements**: Either an iterable of pip requirement strings (e.g. `["torch", "-r requirements.txt", "-c constraints.txt"]`) or the string path to a pip requirements file on the local filesystem (e.g. `"requirements.txt"`). If provided, this describes the environment this model should be run in. If `None`, a default list of requirements is inferred by `mlflow.models.infer_pip_requirements()` from the current software environment. If the requirement inference fails, it falls back to using `get_default_pip_requirements()`. Both requirements and constraints are automatically parsed and written to `requirements.txt` and `constraints.txt` files, respectively, and stored as part of the model. Requirements are also written to the `pip` section of the model's conda environment (`conda.yaml`) file.
- **extra_pip_requirements**: Either an iterable of pip requirement strings (e.g. `["pandas", "-r requirements.txt", "-c constraints.txt"]`) or the string path to a pip requirements file on the local filesystem (e.g. `"requirements.txt"`). If provided, this describes additional pip requirements that are appended to a default set of pip requirements generated automatically based on the user's current software environment. Both requirements and constraints are automatically parsed and written to `requirements.txt` and `constraints.txt` files, respectively, and stored as part of the model. Requirements are also written to the `pip` section of the model's conda environment (`conda.yaml`) file.

  **Warning**: The following arguments can't be specified at the same time: `conda_env`, `pip_requirements`, `extra_pip_requirements`.

  This [example](https://github.com/mlflow/mlflow/blob/master/examples/pip_requirements/pip_requirements.py) demonstrates how to specify pip requirements using `pip_requirements` and `extra_pip_requirements`.

- **metadata**: Custom metadata dictionary passed to the model and stored in the MLmodel file.
- **name**: Model name.
- **params**: A dictionary of parameters to log with the model.
- **tags**: A dictionary of tags to log with the model.
- **model_type**: The type of the model.
- **step**: The step at which to log the model outputs and metrics.
- **model_id**: The ID of the model.
- **export_model**: If set to True, save the model as "pt2" format. This argument is deprecated. For details, see documentation of `serialization_format` argument.
- **serialization_format**: The serialization format used to save the PyTorch model. Accepted values are "pickle" and "pt2".
  - When set to "pickle", the model is serialized using either pickle or cloudpickle, depending on the `pickle_module` parameter.
  - When set to "pt2", the model is saved using torch.export.save, which exports the model as a traced graph. This is a safer serialization format that prevents executing arbitrary code during deserialization.
  - Note that "pt2" format requires `input_example` (used to trace the model graph by virtually executing model.forward) and only supports Numpy array / Tensor or a list of Numpy arrays / Tensors as inputs. For details, see https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/pt2_archive.html.

- **kwargs**: kwargs to pass to `torch.save` method.

#### Returns

A `ModelInfo` instance that contains the metadata of the logged model.

#### Example

```python
import numpy as np
import torch
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

# Define model, loss, and optimizer
model = nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Create training data with relationship y = 2X
X = torch.arange(1.0, 26.0).reshape(-1, 1)
y = X * 2

# Training loop
epochs = 250
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing X to the model
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Create model signature
signature = infer_signature(X.numpy(), model(X).detach().numpy())

# Log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, name="model")

    # convert to scripted model and log the model
    scripted_pytorch_model = torch.jit.script(model)
    mlflow.pytorch.log_model(scripted_pytorch_model, name="scripted_model")

# Fetch the logged model artifacts
print(f"run_id: {run.info.run_id}")
for artifact_path in ["model/data", "scripted_model/data"]:
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
    ]
    print(f"artifacts: {artifacts}")
```

#### Output

```
run_id: 1a1ec9e413ce48e9abf9aec20efd6f71
artifacts: ['model/data/model.pth',
            'model/data/pickle_module_info.txt']
artifacts: ['scripted_model/data/model.pth',
            'scripted_model/data/pickle_module_info.txt']
```

---

### save_model()

```python
mlflow.pytorch.save_model(
    pytorch_model,
    path,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    pickle_module=None,
    signature: mlflow.models.signature.ModelSignature = None,
    input_example = None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    export_model: bool = False,
    serialization_format: Literal['pickle', 'pt2'] = 'pickle',
    **kwargs
)
```

Save a PyTorch model to a path on the local file system.

#### Parameters

- **pytorch_model**: PyTorch model to be saved. Can be either an eager model (subclass of `torch.nn.Module`) or a scripted model prepared via `torch.jit.script` or `torch.jit.trace`.

  To save an eager model, any code dependencies of the model's class, including the class definition itself, should be included in one of the following locations:
  - The package(s) listed in the model's Conda environment, specified by the `conda_env` parameter.
  - One or more of the files specified by the `code_paths` parameter.

- **path**: Local path where the model is to be saved.
- **conda_env**: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. If provided, this describes the environment this model should be run in. Same behavior as in `log_model()`.
- **mlflow_model**: `mlflow.models.Model` this flavor is being added to.
- **code_paths**: A list of local filesystem paths to Python file dependencies (or directories containing file dependencies). See `log_model()` for full description.
- **pickle_module**: The module that PyTorch should use to serialize ("pickle") the specified `pytorch_model`. This is passed as the `pickle_module` parameter to `torch.save()`. By default, this module is also used to deserialize ("unpickle") the model at loading time.
- **signature**: an instance of the `ModelSignature` class. See `log_model()` for full description.
- **input_example**: one or several instances of valid model input. See `log_model()` for full description.
- **extra_files**: A list containing the paths to corresponding extra files. See `log_model()` for full description.
- **pip_requirements**: Either an iterable of pip requirement strings or the string path to a pip requirements file on the local filesystem. See `log_model()` for full description.
- **extra_pip_requirements**: Either an iterable of pip requirement strings or the string path to a pip requirements file on the local filesystem. See `log_model()` for full description.
- **metadata**: Custom metadata dictionary passed to the model and stored in the MLmodel file.
- **export_model**: If set to True, save the model as "pt2" format. This argument is deprecated. For details, see documentation of `serialization_format` argument.
- **serialization_format**: The serialization format used to save the PyTorch model. Accepted values are "pickle" and "pt2". See `log_model()` for full description.
- **kwargs**: kwargs to pass to `torch.save` method.

#### Example

```python
import os
import mlflow
import torch


model = nn.Linear(1, 1)

# Save PyTorch models to current working directory
with mlflow.start_run() as run:
    mlflow.pytorch.save_model(model, "model")

    # Convert to a scripted model and save it
    scripted_pytorch_model = torch.jit.script(model)
    mlflow.pytorch.save_model(scripted_pytorch_model, "scripted_model")

# Load each saved model for inference
for model_path in ["model", "scripted_model"]:
    model_uri = f"{os.getcwd()}/{model_path}"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    print(f"Loaded {model_path}:")
    for x in [6.0, 8.0, 12.0, 30.0]:
        X = torch.Tensor([[x]])
        y_pred = loaded_model(X)
        print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}")
    print("--")
```

#### Output

```
Loaded model:
predict X: 6.0, y_pred: 11.90
predict X: 8.0, y_pred: 15.92
predict X: 12.0, y_pred: 23.96
predict X: 30.0, y_pred: 60.13
--
Loaded scripted_model:
predict X: 6.0, y_pred: 11.90
predict X: 8.0, y_pred: 15.92
predict X: 12.0, y_pred: 23.96
predict X: 30.0, y_pred: 60.13
```
