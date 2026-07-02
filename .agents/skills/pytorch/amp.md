Automatic Mixed Precision

Created On: Sep 15, 2020 | Last Updated: Jan 30, 2025 | Last Verified: Nov 05, 2024

Author: Michael Carilli

torch.cuda.amp provides convenience methods for mixed precision, where some operations use the torch.float32 (float) datatype and other operations use torch.float16 (half). Some ops, like linear layers and convolutions, are much faster in float16 or bfloat16. Other ops, like reductions, often require the dynamic range of float32. Mixed precision tries to match each op to its appropriate datatype, which can reduce your network’s runtime and memory footprint.

Ordinarily, “automatic mixed precision training” uses torch.autocast and torch.cuda.amp.GradScaler together.

This recipe measures the performance of a simple network in default precision, then walks through adding autocast and GradScaler to run the same network in mixed precision with improved performance.

You may download and run this recipe as a standalone Python script. The only requirements are PyTorch 1.6 or later and a CUDA-capable GPU.

Mixed precision primarily benefits Tensor Core-enabled architectures (Volta, Turing, Ampere). This recipe should show significant (2-3X) speedup on those architectures. On earlier architectures (Kepler, Maxwell, Pascal), you may observe a modest speedup. Run nvidia-smi to display your GPU’s architecture.

import torch, time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

A simple network

The following sequence of linear layers and ReLUs should show a speedup with mixed precision.

def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

batch_size, in_size, out_size, and num_layers are chosen to be large enough to saturate the GPU with work. Typically, mixed precision provides the greatest speedup when the GPU is saturated. Small networks may be CPU bound, in which case mixed precision won’t improve performance. Sizes are also chosen such that linear layers’ participating dimensions are multiples of 8, to permit Tensor Core usage on Tensor Core-capable GPUs (see Troubleshooting below).

Exercise: Vary participating sizes and see how the mixed precision speedup changes.

batch_size = 512 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change inputs' ``dtype`` when enabling mixed precision.
data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()

Default Precision

Without torch.cuda.amp, the following simple network executes all ops in default precision (torch.float32):

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Default precision:")

Adding torch.autocast

Instances of torch.autocast serve as context managers that allow regions of your script to run in mixed precision.

In these regions, CUDA ops run in a dtype chosen by autocast to improve performance while maintaining accuracy. See the Autocast Op Reference for details on what precision autocast chooses for each op, and under what circumstances.

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under ``autocast``.
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers ``autocast`` to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
            assert loss.dtype is torch.float32

        # Exits ``autocast`` before backward().
        # Backward passes under ``autocast`` are not recommended.
        # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

Adding GradScaler

Gradient scaling helps prevent gradients with small magnitudes from flushing to zero (“underflowing”) when training with mixed precision.

torch.cuda.amp.GradScaler performs the steps of gradient scaling conveniently.

# Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
# If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
# The same ``GradScaler`` instance should be used for the entire convergence run.
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
scaler = torch.amp.GradScaler("cuda")

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)

        # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
        # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)

        # Updates the scale for next iteration.
        scaler.update()

        opt.zero_grad() # set_to_none=True here can modestly improve performance

All together: “Automatic Mixed Precision”

(The following also demonstrates enabled, an optional convenience argument to autocast and GradScaler. If False, autocast and GradScaler‘s calls become no-ops. This allows switching between default precision and mixed precision without if/else statements.)

use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Mixed precision:")

Inspecting/modifying gradients (e.g., clipping)

All gradients produced by scaler.scale(loss).backward() are scaled. If you wish to modify or inspect the parameters’ .grad attributes between backward() and scaler.step(optimizer), you should unscale them first using scaler.unscale_(optimizer).

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned parameters in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance

Saving/Resuming

To save/resume Amp-enabled runs with bitwise accuracy, use scaler.state_dict and scaler.load_state_dict.

When saving, save the scaler state dict alongside the usual model and optimizer state dicts. Do this either at the beginning of an iteration before any forward passes, or at the end of an iteration after scaler.update().

checkpoint = {"model": net.state_dict(),
              "optimizer": opt.state_dict(),
              "scaler": scaler.state_dict()}
# Write checkpoint as desired, e.g.,
# torch.save(checkpoint, "filename")

When resuming, load the scaler state dict alongside the model and optimizer state dicts. Read checkpoint as desired, for example:

dev = torch.cuda.current_device()
checkpoint = torch.load("filename",
                        map_location = lambda storage, loc: storage.cuda(dev))

net.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
scaler.load_state_dict(checkpoint["scaler"])

If a checkpoint was created from a run without Amp, and you want to resume training with Amp, load model and optimizer states from the checkpoint as usual. The checkpoint won’t contain a saved scaler state, so use a fresh instance of GradScaler.

If a checkpoint was created from a run with Amp and you want to resume training without Amp, load model and optimizer states from the checkpoint as usual, and ignore the saved scaler state.
Inference/Evaluation

autocast may be used by itself to wrap inference or evaluation forward passes. GradScaler is not necessary.
Advanced topics

See the Automatic Mixed Precision Examples for advanced use cases including:

    Gradient accumulation

    Gradient penalty/double backward

    Networks with multiple models, optimizers, or losses

    Multiple GPUs (torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel)

    Custom autograd functions (subclasses of torch.autograd.Function)

If you perform multiple convergence runs in the same script, each run should use a dedicated fresh GradScaler instance. GradScaler instances are lightweight.

If you’re registering a custom C++ op with the dispatcher, see the autocast section of the dispatcher tutorial.
Troubleshooting
Speedup with Amp is minor

    Your network may fail to saturate the GPU(s) with work, and is therefore CPU bound. Amp’s effect on GPU performance won’t matter.

        A rough rule of thumb to saturate the GPU is to increase batch and/or network size(s) as much as you can without running OOM.

        Try to avoid excessive CPU-GPU synchronization (.item() calls, or printing values from CUDA tensors).

        Try to avoid sequences of many small CUDA ops (coalesce these into a few large CUDA ops if you can).

    Your network may be GPU compute bound (lots of matmuls/convolutions) but your GPU does not have Tensor Cores. In this case a reduced speedup is expected.

    The matmul dimensions are not Tensor Core-friendly. Make sure matmuls participating sizes are multiples of 8. (For NLP models with encoders/decoders, this can be subtle. Also, convolutions used to have similar size constraints for Tensor Core use, but for CuDNN versions 7.3 and later, no such constraints exist. See here for guidance.)

Loss is inf/NaN

First, check if your network fits an advanced use case. See also Prefer binary_cross_entropy_with_logits over binary_cross_entropy.

If you’re confident your Amp usage is correct, you may need to file an issue, but before doing so, it’s helpful to gather the following information:

    Disable autocast or GradScaler individually (by passing enabled=False to their constructor) and see if infs/NaNs persist.

    If you suspect part of your network (e.g., a complicated loss function) overflows , run that forward region in float32 and see if infs/NaN``s persist. `The autocast docstring <https://pytorch.org/docs/stable/amp.html#torch.autocast>`_'s last code snippet shows forcing a subregion to run in ``float32 (by locally disabling autocast and casting the subregion’s inputs).

Type mismatch error (may manifest as CUDNN_STATUS_BAD_PARAM)

Autocast tries to cover all ops that benefit from or require casting. Ops that receive explicit coverage are chosen based on numerical properties, but also on experience. If you see a type mismatch error in an autocast enabled forward region or a backward pass following that region, it’s possible autocast missed an op.

Please file an issue with the error backtrace. export TORCH_SHOW_CPP_STACKTRACES=1 before running your script to provide fine-grained information on which backend op is failing.

Automatic Mixed Precision examples

Created On: Feb 13, 2020 | Last Updated On: Sep 13, 2024

Ordinarily, “automatic mixed precision training” means training with torch.autocast and torch.amp.GradScaler together.

Instances of torch.autocast enable autocasting for chosen regions. Autocasting automatically chooses the precision for operations to improve performance while maintaining accuracy.

Instances of torch.amp.GradScaler help perform the steps of gradient scaling conveniently. Gradient scaling improves convergence for networks with float16 (by default on CUDA and XPU) gradients by minimizing gradient underflow, as explained here.

torch.autocast and torch.amp.GradScaler are modular. In the samples below, each is used as its individual documentation suggests.

(Samples here are illustrative. See the Automatic Mixed Precision recipe for a runnable walkthrough.)

    Typical Mixed Precision Training

    Working with Unscaled Gradients

        Gradient clipping

    Working with Scaled Gradients

        Gradient accumulation

        Gradient penalty

    Working with Multiple Models, Losses, and Optimizers

    Working with Multiple GPUs

        DataParallel in a single process

        DistributedDataParallel, one GPU per process

        DistributedDataParallel, multiple GPUs per process

    Autocast and Custom Autograd Functions

        Functions with multiple inputs or autocastable ops

        Functions that need a particular dtype

Typical Mixed Precision Training

# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

Working with Unscaled Gradients

All gradients produced by scaler.scale(loss).backward() are scaled. If you wish to modify or inspect the parameters’ .grad attributes between backward() and scaler.step(optimizer), you should unscale them first. For example, gradient clipping manipulates a set of gradients such that their global norm (see torch.nn.utils.clip_grad_norm_()) or maximum magnitude (see torch.nn.utils.clip_grad_value_()) is <=<= some user-imposed threshold. If you attempted to clip without unscaling, the gradients’ norm/maximum magnitude would also be scaled, so your requested threshold (which was meant to be the threshold for unscaled gradients) would be invalid.

scaler.unscale_(optimizer) unscales gradients held by optimizer’s assigned parameters. If your model or models contain other parameters that were assigned to another optimizer (say optimizer2), you may call scaler.unscale_(optimizer2) separately to unscale those parameters’ gradients as well.
Gradient clipping

Calling scaler.unscale_(optimizer) before clipping enables you to clip unscaled gradients as usual:

scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

scaler records that scaler.unscale_(optimizer) was already called for this optimizer this iteration, so scaler.step(optimizer) knows not to redundantly unscale gradients before (internally) calling optimizer.step().

Warning

unscale_ should only be called once per optimizer per step call, and only after all gradients for that optimizer’s assigned parameters have been accumulated. Calling unscale_ twice for a given optimizer between each step triggers a RuntimeError.
Working with Scaled Gradients
Gradient accumulation

Gradient accumulation adds gradients over an effective batch of size batch_per_iter * iters_to_accumulate (* num_procs if distributed). The scale should be calibrated for the effective batch, which means inf/NaN checking, step skipping if inf/NaN grads are found, and scale updates should occur at effective-batch granularity. Also, grads should remain scaled, and the scale factor should remain constant, while grads for a given effective batch are accumulated. If grads are unscaled (or the scale factor changes) before accumulation is complete, the next backward pass will add scaled grads to unscaled grads (or grads scaled by a different factor) after which it’s impossible to recover the accumulated unscaled grads step must apply.

Therefore, if you want to unscale_ grads (e.g., to allow clipping unscaled grads), call unscale_ just before step, after all (scaled) grads for the upcoming step have been accumulated. Also, only call update at the end of iterations where you called step for a full effective batch:

scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

Gradient penalty

A gradient penalty implementation commonly creates gradients using torch.autograd.grad(), combines them to create the penalty value, and adds the penalty value to the loss.

Here’s an ordinary example of an L2 penalty without gradient scaling or autocasting:

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # Creates gradients
        grad_params = torch.autograd.grad(outputs=loss,
                                          inputs=model.parameters(),
                                          create_graph=True)

        # Computes the penalty term and adds it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        loss = loss + grad_norm

        loss.backward()

        # clip gradients here, if desired

        optimizer.step()

To implement a gradient penalty with gradient scaling, the outputs Tensor(s) passed to torch.autograd.grad() should be scaled. The resulting gradients will therefore be scaled, and should be unscaled before being combined to create the penalty value.

Also, the penalty term computation is part of the forward pass, and therefore should be inside an autocast context.

Here’s how that looks for the same L2 penalty:

scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales the loss for autograd.grad's backward pass, producing scaled_grad_params
        scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                 inputs=model.parameters(),
                                                 create_graph=True)

        # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
        # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
        inv_scale = 1./scaler.get_scale()
        grad_params = [p * inv_scale for p in scaled_grad_params]

        # Computes the penalty term and adds it to the loss
        with autocast(device_type='cuda', dtype=torch.float16):
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

        # Applies scaling to the backward call as usual.
        # Accumulates leaf gradients that are correctly scaled.
        scaler.scale(loss).backward()

        # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

        # step() and update() proceed as usual.
        scaler.step(optimizer)
        scaler.update()

Working with Multiple Models, Losses, and Optimizers

If your network has multiple losses, you must call scaler.scale on each of them individually. If your network has multiple optimizers, you may call scaler.unscale_ on any of them individually, and you must call scaler.step on each of them individually.

However, scaler.update should only be called once, after all optimizers used this iteration have been stepped:

scaler = torch.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

        # (retain_graph here is unrelated to amp, it's present because in this
        # example, both backward() calls share some sections of graph.)
        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        # You can choose which optimizers receive explicit unscaling, if you
        # want to inspect or modify the gradients of the params they own.
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()

Each optimizer checks its gradients for infs/NaNs and makes an independent decision whether or not to skip the step. This may result in one optimizer skipping the step while the other one does not. Since step skipping occurs rarely (every several hundred iterations) this should not impede convergence. If you observe poor convergence after adding gradient scaling to a multiple-optimizer model, please report a bug.
Working with Multiple GPUs

The issues described here only affect autocast. GradScaler‘s usage is unchanged.
DataParallel in a single process

Even if torch.nn.DataParallel spawns threads to run the forward pass on each device. The autocast state is propagated in each one and the following will work:

model = MyModel()
dp_model = nn.DataParallel(model)

# Sets autocast in the main thread
with autocast(device_type='cuda', dtype=torch.float16):
    # dp_model's internal threads will autocast.
    output = dp_model(input)
    # loss_fn also autocast
    loss = loss_fn(output)

DistributedDataParallel, one GPU per process

torch.nn.parallel.DistributedDataParallel’s documentation recommends one GPU per process for best performance. In this case, DistributedDataParallel does not spawn threads internally, so usages of autocast and GradScaler are not affected.
DistributedDataParallel, multiple GPUs per process

Here torch.nn.parallel.DistributedDataParallel may spawn a side thread to run the forward pass on each device, like torch.nn.DataParallel. The fix is the same: apply autocast as part of your model’s forward method to ensure it’s enabled in side threads.
Autocast and Custom Autograd Functions

If your network uses custom autograd functions (subclasses of torch.autograd.Function), changes are required for autocast compatibility if any function

    takes multiple floating-point Tensor inputs,

    wraps any autocastable op (see the Autocast Op Reference), or

    requires a particular dtype (for example, if it wraps CUDA extensions that were only compiled for dtype).

In all cases, if you’re importing the function and can’t alter its definition, a safe fallback is to disable autocast and force execution in float32 ( or dtype) at any points of use where errors occur:

with autocast(device_type='cuda', dtype=torch.float16):
    ...
    with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        output = imported_function(input1.float(), input2.float())

If you’re the function’s author (or can alter its definition) a better solution is to use the torch.amp.custom_fwd() and torch.amp.custom_bwd() decorators as shown in the relevant case below.
Functions with multiple inputs or autocastable ops

Apply custom_fwd and custom_bwd (with no arguments) to forward and backward respectively. These ensure forward executes with the current autocast state and backward executes with the same autocast state as forward (which can prevent type mismatch errors):

class MyMM(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)

Now MyMM can be invoked anywhere, without disabling autocast or manually casting inputs:

mymm = MyMM.apply

with autocast(device_type='cuda', dtype=torch.float16):
    output = mymm(input1, input2)

Functions that need a particular dtype

Consider a custom function that requires torch.float32 inputs. Apply custom_fwd(device_type='cuda', cast_inputs=torch.float32) to forward and custom_bwd(device_type='cuda') to backward. If forward runs in an autocast-enabled region, the decorators cast floating-point Tensor inputs to float32 on designated device assigned by the argument device_type, CUDA in this example, and locally disable autocast during forward and backward:

class MyFloat32Func(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ...
        return fwd_output
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        ...

Now MyFloat32Func can be invoked anywhere, without manually disabling autocast or casting inputs:

func = MyFloat32Func.apply

with autocast(device_type='cuda', dtype=torch.float16):
    # func will run in float32, regardless of the surrounding autocast state
    output = func(input)
