# `torch.optim`

Created On: Jun 13, 2025 | Last Updated On: Jan 26, 2026

`torch.optim` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general enough so that more sophisticated ones can also be integrated in the future.

## How to Use an Optimizer

To use `torch.optim`, construct an optimizer object that holds the current state and updates parameters based on the computed gradients.

### Constructing It

To construct an `Optimizer`, provide an iterable containing the parameters to optimize. These can be parameters directly, or named parameters as tuples of `(str, Parameter)`. You can then specify optimizer-specific options such as learning rate, weight decay, and momentum.

Example:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

Named parameters example:

```python
optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([("layer0", var1), ("layer1", var2)], lr=0.0001)
```

### Per-Parameter Options

Optimizers also support specifying per-parameter options. To do this, instead of passing an iterable of Variables, pass in an iterable of dicts. Each dict defines a separate parameter group and should contain a `params` key with a list of parameters belonging to it. Other keys should match the keyword arguments accepted by the optimizer and will be used as optimization options for that group.

For example, this is useful when specifying per-layer learning rates:

```python
optim.SGD([
    {"params": model.base.parameters(), "lr": 1e-2},
    {"params": model.classifier.parameters()},
], lr=1e-3, momentum=0.9)

optim.SGD([
    {"params": model.base.named_parameters(), "lr": 1e-2},
    {"params": model.classifier.named_parameters()},
], lr=1e-3, momentum=0.9)
```

This means that `model.base` parameters will use a learning rate of `1e-2`, whereas `model.classifier` parameters will stick to the default learning rate of `1e-3`. A momentum of `0.9` will be used for all parameters.

> Note
>
> You can still pass options as keyword arguments. They will be used as defaults in groups that do not override them. This is useful when you only want to vary a single option while keeping all others consistent between parameter groups.

Also consider the following example related to distinct penalization of parameters. Remember that `parameters()` returns an iterable that contains all learnable parameters, including biases and other parameters that may prefer distinct penalization. To address this, one can specify individual penalization weights for each parameter group:

```python
bias_params = [p for name, p in self.named_parameters() if "bias" in name]
others = [p for name, p in self.named_parameters() if "bias" not in name]

optim.SGD([
    {"params": others},
    {"params": bias_params, "weight_decay": 0},
], weight_decay=1e-2, lr=1e-2)
```

In this manner, bias terms are isolated from non-bias terms, and a `weight_decay` of `0` is set specifically for the bias terms to avoid any penalization for this group.

### Taking an Optimization Step

All optimizers implement a `step()` method that updates the parameters. It can be used in two ways.

#### `optimizer.step()`

This is a simplified version supported by most optimizers. The function can be called once the gradients are computed using `backward()`.

Example:

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

#### `optimizer.step(closure)`

Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.

Example:

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)
```

## Base Class

### `class torch.optim.Optimizer(params, defaults)`

`[source]`

Base class for all optimizers.

> Warning
>
> Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. Examples of objects that do not satisfy those properties are sets and iterators over values of dictionaries.

Parameters:

- `params` (`iterable`): An iterable of `torch.Tensor` values or dicts. Specifies what Tensors should be optimized.
- `defaults` (`dict[str, Any]`): A dict containing default values of optimization options, used when a parameter group does not specify them.

### Optimizer Methods

- `Optimizer.add_param_group`: Add a param group to the optimizer's `param_groups`.
- `Optimizer.load_state_dict`: Load the optimizer state.
- `Optimizer.register_load_state_dict_pre_hook`: Register a `load_state_dict` pre-hook which will be called before `load_state_dict()` is called. It should have the documented signature.
- `Optimizer.register_load_state_dict_post_hook`: Register a `load_state_dict` post-hook which will be called after `load_state_dict()` is called. It should have the documented signature.
- `Optimizer.state_dict`: Return the state of the optimizer as a dict.
- `Optimizer.register_state_dict_pre_hook`: Register a state-dict pre-hook which will be called before `state_dict()` is called.
- `Optimizer.register_state_dict_post_hook`: Register a state-dict post-hook which will be called after `state_dict()` is called.
- `Optimizer.step`: Perform a single optimization step to update parameters.
- `Optimizer.register_step_pre_hook`: Register an optimizer step pre-hook which will be called before `optimizer.step()`.
- `Optimizer.register_step_post_hook`: Register an optimizer step post-hook which will be called after `optimizer.step()`.
- `Optimizer.zero_grad`: Reset the gradients of all optimized `torch.Tensor` values.

## Algorithms

- `Adadelta`: Implements Adadelta algorithm.
- `Adafactor`: Implements Adafactor algorithm.
- `Adagrad`: Implements Adagrad algorithm.
- `Adam`: Implements Adam algorithm.
- `AdamW`: Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance.
- `SparseAdam`: Implements a masked version of the Adam algorithm suitable for sparse gradients.
- `Adamax`: Implements Adamax algorithm, a variant of Adam based on infinity norm.
- `ASGD`: Implements Averaged Stochastic Gradient Descent.
- `LBFGS`: Implements L-BFGS algorithm.
- `Muon`: Implements Muon algorithm.
- `NAdam`: Implements NAdam algorithm.
- `RAdam`: Implements RAdam algorithm.
- `RMSprop`: Implements RMSprop algorithm.
- `Rprop`: Implements the resilient backpropagation algorithm.
- `SGD`: Implements stochastic gradient descent, optionally with momentum.

Many of these algorithms have multiple implementations optimized for performance, readability, or generality, so the default behavior is to prefer the generally fastest implementation for the current device when the user has not requested a specific backend.

There are three major categories of implementations: for-loop, `foreach` (multi-tensor), and fused. The most straightforward implementations are for-loops over the parameters with large chunks of computation. For-looping is usually slower than `foreach` implementations, which combine parameters into a multi-tensor and run the large chunks of computation at once, thereby saving many sequential kernel calls. Some optimizers have even faster fused implementations, which fuse the large chunks of computation into a single kernel. In rough terms, the expected performance order is:

$$
	ext{fused} > \text{foreach} > \text{for-loop}
$$

When applicable, PyTorch defaults to `foreach` over for-loop. Applicable means the `foreach` implementation is available, the user has not specified implementation-specific kwargs such as `fused`, `foreach`, or `differentiable`, and all tensors are native. While fused can be even faster than `foreach`, those implementations are newer and still getting bake-in time.

### Available and Default Implementations

| Algorithm | Default | Has `foreach`? | Has `fused`? |
| --- | --- | --- | --- |
| `Adadelta` | `foreach` | yes | no |
| `Adafactor` | `for-loop` | no | no |
| `Adagrad` | `foreach` | yes | yes (CPU only) |
| `Adam` | `foreach` | yes | yes |
| `AdamW` | `foreach` | yes | yes |
| `SparseAdam` | `for-loop` | no | no |
| `Adamax` | `foreach` | yes | no |
| `ASGD` | `foreach` | yes | no |
| `LBFGS` | `for-loop` | no | no |
| `Muon` | `for-loop` | no | no |
| `NAdam` | `foreach` | yes | no |
| `RAdam` | `foreach` | yes | no |
| `RMSprop` | `foreach` | yes | no |
| `Rprop` | `foreach` | yes | no |
| `SGD` | `foreach` | yes | yes |

### Stability Status for Fused Implementations

| Algorithm | CPU | CUDA | MPS |
| --- | --- | --- | --- |
| `Adadelta` | unsupported | unsupported | unsupported |
| `Adafactor` | unsupported | unsupported | unsupported |
| `Adagrad` | beta | unsupported | unsupported |
| `Adam` | beta | stable | beta |
| `AdamW` | beta | stable | beta |
| `SparseAdam` | unsupported | unsupported | unsupported |
| `Adamax` | unsupported | unsupported | unsupported |
| `ASGD` | unsupported | unsupported | unsupported |
| `LBFGS` | unsupported | unsupported | unsupported |
| `Muon` | unsupported | unsupported | unsupported |
| `NAdam` | unsupported | unsupported | unsupported |
| `RAdam` | unsupported | unsupported | unsupported |
| `RMSprop` | unsupported | unsupported | unsupported |
| `Rprop` | unsupported | unsupported | unsupported |
| `SGD` | beta | beta | beta |

## How to Adjust Learning Rate

`torch.optim.lr_scheduler.LRScheduler` provides several methods to adjust the learning rate based on the number of epochs. `torch.optim.lr_scheduler.ReduceLROnPlateau` allows dynamic learning-rate reduction based on validation measurements.

Learning-rate scheduling should be applied after the optimizer update. For example:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

Most learning-rate schedulers can be called back-to-back, also referred to as chaining schedulers. The result is that each scheduler is applied one after the other on the learning rate obtained by the one preceding it.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler1.step()
    scheduler2.step()
```

Template used throughout the documentation:

```python
scheduler = ...
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

> Warning
>
> Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before the optimizer update. PyTorch 1.1.0 changed this behavior in a backward-compatibility-breaking way. If you call `scheduler.step()` before `optimizer.step()`, this will skip the first value of the learning-rate schedule.

### Scheduler Reference

- `lr_scheduler.LRScheduler`: Base class for all learning-rate schedulers.
- `lr_scheduler.LambdaLR`: Sets the initial learning rate.
- `lr_scheduler.MultiplicativeLR`: Multiply the learning rate of each parameter group by the factor given in the specified function.
- `lr_scheduler.StepLR`: Decays the learning rate of each parameter group by `gamma` every `step_size` epochs.
- `lr_scheduler.MultiStepLR`: Decays the learning rate of each parameter group by `gamma` once the number of epochs reaches one of the milestones.
- `lr_scheduler.ConstantLR`: Multiply the learning rate of each parameter group by a small constant factor.
- `lr_scheduler.LinearLR`: Decays the learning rate of each parameter group by a linearly changing small multiplicative factor.
- `lr_scheduler.ExponentialLR`: Decays the learning rate of each parameter group by `gamma` every epoch.
- `lr_scheduler.PolynomialLR`: Decays the learning rate of each parameter group using a polynomial function in the given `total_iters`.
- `lr_scheduler.CosineAnnealingLR`: Set the learning rate of each parameter group using a cosine annealing schedule.
- `lr_scheduler.ChainedScheduler`: Chains a list of learning-rate schedulers.
- `lr_scheduler.SequentialLR`: Contains a list of schedulers expected to be called sequentially during the optimization process.
- `lr_scheduler.ReduceLROnPlateau`: Reduce learning rate when a metric has stopped improving.
- `lr_scheduler.CyclicLR`: Sets the learning rate of each parameter group according to cyclical learning-rate policy (CLR).
- `lr_scheduler.OneCycleLR`: Sets the learning rate of each parameter group according to the 1cycle learning-rate policy.
- `lr_scheduler.CosineAnnealingWarmRestarts`: Set the learning rate of each parameter group using a cosine annealing schedule.

## How to Utilize Named Parameters to Load Optimizer State Dict

The function `load_state_dict()` stores the optional `param_names` content from the loaded state dict if present. However, the process of loading the optimizer state is not affected, as the order of the parameters matters to maintain compatibility when ordering differs. To utilize the loaded parameter names from the loaded state dict, a custom `register_load_state_dict_pre_hook` needs to be implemented according to the desired behavior.

This can be useful, for instance, when the model architecture changes, but the weights and optimizer states need to remain unchanged.

### Example: Duplicating Optimizer State for Two Experts

```python
class OneLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x):
        return self.fc(x)


model = OneLayerModel()
optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
# training..
torch.save(optimizer.state_dict(), PATH)
```

Let's say that `model` implements an expert (MoE), and we want to duplicate it and resume training for two experts, both initialized the same way as the `fc` layer. For the following `model2`, we create two layers identical to `fc` and resume training by loading the model weights and optimizer states from `model` into both `fc1` and `fc2` of `model2`.

```python
class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(3, 4)

    def forward(self, x):
        return (self.fc1(x) + self.fc2(x)) / 2


model2 = TwoLayerModel()
# adapt and load model weights..
optimizer2 = optim.SGD(model2.named_parameters(), lr=0.01, momentum=0.9)
```

To load the state dict for `optimizer2` with the state dict of the previous optimizer such that both `fc1` and `fc2` will be initialized with a copy of `fc` optimizer states, we can use the following hook:

```python
def adapt_state_dict_ids(optimizer, state_dict):
    adapted_state_dict = deepcopy(optimizer.state_dict())
    # Copy setup parameters (lr, weight_decay, etc.), in case they differ in the loaded state dict.
    for k, v in state_dict["param_groups"][0].items():
        if k not in ["params", "param_names"]:
            adapted_state_dict["param_groups"][0][k] = v

    lookup_dict = {
        "fc1.weight": "fc.weight",
        "fc1.bias": "fc.bias",
        "fc2.weight": "fc.weight",
        "fc2.bias": "fc.bias",
    }
    clone_deepcopy = lambda d: {
        k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v))
        for k, v in d.items()
    }
    for param_id, param_name in zip(
        optimizer.state_dict()["param_groups"][0]["params"],
        optimizer.state_dict()["param_groups"][0]["param_names"],
    ):
        name_in_loaded = lookup_dict[param_name]
        index_in_loaded_list = state_dict["param_groups"][0]["param_names"].index(name_in_loaded)
        id_in_loaded = state_dict["param_groups"][0]["params"][index_in_loaded_list]
        # Copy the state of the corresponding parameter
        if id_in_loaded in state_dict["state"]:
            adapted_state_dict["state"][param_id] = clone_deepcopy(state_dict["state"][id_in_loaded])

    return adapted_state_dict


optimizer2.register_load_state_dict_pre_hook(adapt_state_dict_ids)
optimizer2.load_state_dict(torch.load(PATH))  # The previous optimizer saved state_dict
```

This ensures that the adapted state dict with the correct states for the layers of `model2` will be used during model loading. Note that this code is designed specifically for this example, for example assuming a single parameter group, and other cases might require different adaptations.

### Example: Handling Missing Parameters in a Changed Model

The following example shows how to handle missing parameters in a loaded state dict when the model structure changes. The `Model_bypass` adds a new bypass layer, which is not present in the original `Model1`. To resume training, a custom `adapt_state_dict_missing_param` hook is used to adapt the optimizer state dict, ensuring existing parameters are mapped correctly, while missing ones such as the bypass layer remain unchanged and are trained from scratch.

```python
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x) + x


model = Model1()
optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
# training..
torch.save(optimizer.state_dict(), PATH)


class Model_bypass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.bypass = nn.Linear(5, 5, bias=False)
        torch.nn.init.eye_(self.bypass.weight)

    def forward(self, x):
        return self.fc(x) + self.bypass(x)


model2 = Model_bypass()
optimizer2 = optim.SGD(model2.named_parameters(), lr=0.01, momentum=0.9)


def adapt_state_dict_missing_param(optimizer, state_dict):
    adapted_state_dict = deepcopy(optimizer.state_dict())
    # Copy setup parameters (lr, weight_decay, etc.), in case they differ in the loaded state dict.
    for k, v in state_dict["param_groups"][0].items():
        if k not in ["params", "param_names"]:
            adapted_state_dict["param_groups"][0][k] = v

    lookup_dict = {
        "fc.weight": "fc.weight",
        "fc.bias": "fc.bias",
        "bypass.weight": None,
    }

    clone_deepcopy = lambda d: {
        k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v))
        for k, v in d.items()
    }
    for param_id, param_name in zip(
        optimizer.state_dict()["param_groups"][0]["params"],
        optimizer.state_dict()["param_groups"][0]["param_names"],
    ):
        name_in_loaded = lookup_dict[param_name]
        if name_in_loaded in state_dict["param_groups"][0]["param_names"]:
            index_in_loaded_list = state_dict["param_groups"][0]["param_names"].index(name_in_loaded)
            id_in_loaded = state_dict["param_groups"][0]["params"][index_in_loaded_list]
            # Copy the state of the corresponding parameter
            if id_in_loaded in state_dict["state"]:
                adapted_state_dict["state"][param_id] = clone_deepcopy(state_dict["state"][id_in_loaded])

    return adapted_state_dict


optimizer2.register_load_state_dict_pre_hook(adapt_state_dict_ids)
optimizer2.load_state_dict(torch.load(PATH))  # The previous optimizer saved state_dict
```

### Example: Matching by Parameter Names

Instead of loading a state according to parameter order, the following hook can be used to load according to parameter names:

```python
def names_matching(optimizer, state_dict):
    assert len(state_dict["param_groups"]) == len(optimizer.state_dict()["param_groups"])
    adapted_state_dict = deepcopy(optimizer.state_dict())
    for g_ind in range(len(state_dict["param_groups"])):
        assert len(state_dict["param_groups"][g_ind]["params"]) == len(
            optimizer.state_dict()["param_groups"][g_ind]["params"]
        )

        for k, v in state_dict["param_groups"][g_ind].items():
            if k not in ["params", "param_names"]:
                adapted_state_dict["param_groups"][g_ind][k] = v

        for param_id, param_name in zip(
            optimizer.state_dict()["param_groups"][g_ind]["params"],
            optimizer.state_dict()["param_groups"][g_ind]["param_names"],
        ):
            index_in_loaded_list = state_dict["param_groups"][g_ind]["param_names"].index(param_name)
            id_in_loaded = state_dict["param_groups"][g_ind]["params"][index_in_loaded_list]
            # Copy the state of the corresponding parameter
            if id_in_loaded in state_dict["state"]:
                adapted_state_dict["state"][param_id] = deepcopy(state_dict["state"][id_in_loaded])

    return adapted_state_dict
```

## Weight Averaging (SWA and EMA)

`torch.optim.swa_utils.AveragedModel` implements Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA). `torch.optim.swa_utils.SWALR` implements the SWA learning-rate scheduler and `torch.optim.swa_utils.update_bn()` is a utility function used to update SWA or EMA batch-normalization statistics at the end of training.

SWA has been proposed in Averaging Weights Leads to Wider Optima and Better Generalization.

EMA is a widely known technique to reduce training time by reducing the number of weight updates needed. It is a variation of Polyak averaging, but uses exponential weights instead of equal weights across iterations.

### Constructing Averaged Models

The `AveragedModel` class serves to compute the weights of the SWA or EMA model.

You can create an SWA averaged model by running:

```python
averaged_model = AveragedModel(model)
```

EMA models are constructed by specifying the `multi_avg_fn` argument:

```python
decay = 0.999
averaged_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
```

Decay is a parameter between `0` and `1` that controls how fast the averaged parameters are decayed. If not provided to `torch.optim.swa_utils.get_ema_multi_avg_fn()`, the default is `0.999`. Decay should be close to `1.0`, as smaller values can cause optimization convergence issues.

`torch.optim.swa_utils.get_ema_multi_avg_fn()` returns a function that applies the following EMA equation to the weights:

$$
W_0^{EMA} = W_0^{model}
$$

$$
W_{t+1}^{EMA} = decay \times W_t^{EMA} + (1 - decay) \times W_{t+1}^{model}
$$

where $W_t^{EMA}$ is the EMA parameter at step $t$, $W_t^{model}$ is the model parameter at step $t$, and `decay` is the EMA decay rate.

Here the `model` can be an arbitrary `torch.nn.Module` object. `averaged_model` will keep track of the running averages of the parameters of the model. To update these averages, use `update_parameters()` after `optimizer.step()`:

```python
averaged_model.update_parameters(model)
```

For SWA and EMA, this call is usually done right after the optimizer step. In the case of SWA, this is usually skipped for some number of steps at the beginning of training.

### Custom Averaging Strategies

By default, `torch.optim.swa_utils.AveragedModel` computes a running equal average of the parameters that you provide, but you can also use custom averaging functions with the `avg_fn` or `multi_avg_fn` parameters:

- `avg_fn` allows defining a function operating on each parameter tuple `(averaged_parameter, model_parameter)` and should return the new averaged parameter.
- `multi_avg_fn` allows defining more efficient operations acting on a tuple of parameter lists, `(averaged_parameter_list, model_parameter_list)`, for example using the `torch._foreach*` functions. This function must update the averaged parameters in-place.

In the following example `ema_model` computes an exponential moving average using the `avg_fn` parameter:

```python
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
    0.9 * averaged_model_parameter + 0.1 * model_parameter
)
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
```

In the following example `ema_model` computes an exponential moving average using the more efficient `multi_avg_fn` parameter:

```python
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))
```

### SWA Learning Rate Schedules

Typically, in SWA the learning rate is set to a high constant value. `SWALR` is a learning-rate scheduler that anneals the learning rate to a fixed value and then keeps it constant. For example, the following code creates a scheduler that linearly anneals the learning rate from its initial value to `0.05` in `5` epochs within each parameter group:

```python
swa_scheduler = torch.optim.swa_utils.SWALR(
    optimizer,
    anneal_strategy="linear",
    anneal_epochs=5,
    swa_lr=0.05,
)
```

You can also use cosine annealing to a fixed value instead of linear annealing by setting `anneal_strategy="cos"`.

### Taking Care of Batch Normalization

`update_bn()` is a utility function that allows computing the batch-normalization statistics for the SWA model on a given dataloader at the end of training:

```python
torch.optim.swa_utils.update_bn(loader, swa_model)
```

`update_bn()` applies the `swa_model` to every element in the dataloader and computes the activation statistics for each batch-normalization layer in the model.

> Warning
>
> `update_bn()` assumes that each batch in the dataloader is either a tensor or a list of tensors where the first element is the tensor that the network should be applied to. If your dataloader has a different structure, you can update batch-normalization statistics by doing a forward pass with the SWA model on each element of the dataset.

### Putting It All Together: SWA

In the example below, `swa_model` is the SWA model that accumulates the averages of the weights. We train the model for a total of `300` epochs and switch to the SWA learning-rate schedule and start to collect SWA averages at epoch `160`:

```python
loader, optimizer, model, loss_fn = ...
swa_model = torch.optim.swa_utils.AveragedModel(model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
swa_start = 160
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(300):
    for input, target in loader:
        optimizer.zero_grad()
        loss_fn(model(input), target).backward()
        optimizer.step()
    if epoch > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(loader, swa_model)
# Use swa_model to make predictions on test data
preds = swa_model(test_input)
```

### Putting It All Together: EMA

In the example below, `ema_model` is the EMA model that accumulates the exponentially decayed averages of the weights with a decay rate of `0.999`. We train the model for a total of `300` epochs and start to collect EMA averages immediately.

```python
loader, optimizer, model, loss_fn = ...
ema_model = torch.optim.swa_utils.AveragedModel(
    model,
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
)

for epoch in range(300):
    for input, target in loader:
        optimizer.zero_grad()
        loss_fn(model(input), target).backward()
        optimizer.step()
        ema_model.update_parameters(model)

# Update bn statistics for the ema_model at the end
torch.optim.swa_utils.update_bn(loader, ema_model)
# Use ema_model to make predictions on test data
preds = ema_model(test_input)
```

## SWA Utilities

- `swa_utils.AveragedModel`: Implements averaged model support for Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).
- `swa_utils.SWALR`: Anneals the learning rate in each parameter group to a fixed value.

### `torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.999)`

`[source]`

Get the function applying exponential moving average (EMA) across multiple params.

The EMA is computed as:

$$
W_0^{EMA} = W_0^{model}
$$

$$
W_{t+1}^{EMA} = decay \times W_t^{EMA} + (1 - decay) \times W_{t+1}^{model}
$$

where $W_t^{EMA}$ is the EMA parameter at step $t$, $W_t^{model}$ is the model parameter at step $t$, and `decay` is the decay rate.

Parameters:

- `decay` (`float`): Decay rate for EMA. Must be in the range `[0, 1]`. Default: `0.999`.

Returns:

- A function that updates EMA parameters given current model parameters.

Return type:

- `Callable`

### `torch.optim.swa_utils.update_bn(loader, model, device=None)`

`[source]`

Update BatchNorm `running_mean` and `running_var` buffers in the model.

It performs one pass over data in `loader` to estimate the activation statistics for BatchNorm layers in the model.

Parameters:

- `loader` (`torch.utils.data.DataLoader`): Dataset loader used to compute the activation statistics. Each data batch should be either a tensor, or a list or tuple whose first element is a tensor containing data.
- `model` (`torch.nn.Module`): Model for which we seek to update BatchNorm statistics.
- `device` (`torch.device`, optional): If set, data will be transferred to `device` before being passed into `model`.

Example:

```python
loader, model = ...
torch.optim.swa_utils.update_bn(loader, model)
```

> Note
>
> The `update_bn` utility assumes that each data batch in `loader` is either a tensor or a list or tuple of tensors. In the latter case it is assumed that `model.forward()` should be called on the first element of the list or tuple corresponding to the data batch.
