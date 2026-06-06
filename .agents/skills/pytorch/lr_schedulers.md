Learning rate schedulers play a crucial role in training deep learning models. They adjust the learning rate during training, helping to converge faster and avoid local minima. PyTorch provides a variety of learning rate schedulers, each with its unique characteristics and use cases (https://medium.com/@benjybo7/10-must-have-schedulers-that-will-boost-your-models-performance-2ff0c446ac98). 

1. StepLR

StepLR reduces the learning rate by a factor of gamma every step_size epochs. The idea is to lower the learning rate at regular intervals, allowing the model to take larger steps initially and then fine-tune with smaller steps. It works well with many models like ResNet and VGG for image classification and models like DeepSpeech for speech recognition.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(50):
    # Training code here
    scheduler.step()

```

2. MultiStepLR

MultiStepLR decreases the learning rate by gamma at specified epochs, allowing more flexible learning rate adjustments at specific points in training. This scheduler is often used in training models like Faster R-CNN for object detection and Transformer for sequence modeling.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

for epoch in range(100):
    # Training code here
    scheduler.step()
```

3. ExponentialLR

ExponentialLR reduces the learning rate exponentially at each epoch by a factor of gamma, providing a smooth and continuous decay of the learning rate. It is useful in training Generative Adversarial Networks (GANs) and deep Q-networks in reinforcement learning.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

for epoch in range(100):
    # Training code here
    scheduler.step()

```

4. CosineAnnealingLR

CosineAnnealingLR adjusts the learning rate following a cosine curve, decreasing it to a minimum value and then restarting. This strategy mimics a warm restart, allowing the model to escape local minima. It is likely the default scheduler and best you should try, as it is effective in training a wide variety of models, such as Restormer for image restoration and ResNet++ in image classifiction.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(100):
    # Training code here
    scheduler.step()
```

5. ReduceLROnPlateau

ReduceLROnPlateau reduces the learning rate when a metric has stopped improving, adapting based on validation performance. It should only be used if the you have very low label noise in your validation data, making the metrics there very reliable.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

for epoch in range(100):
    # Training code here
    validation_loss = validate()
    scheduler.step(validation_loss)
```

6. CyclicLR

CyclicLR cycles the learning rate between two boundaries with a constant frequency, helping to avoid saddle points. This scheduler is effective in cases where you suspect to have local minima, where you model can get “stuck” in a certain prediction pattern. This scheduler will help you escape the local minima and hopefully find better ones.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=2000, mode='triangular')

for epoch in range(100):
    for batch in dataloader:
        # Training code here
        scheduler.step()
```

7. OneCycleLR

OneCycleLR sets the learning rate according to the 1cycle policy, increasing the learning rate from an initial value to some maximum value and then decreasing it. This allows the model to explore a larger learning rate initially and then fine-tune the weights with a lower learning rate. It can be useful when using pretrained models with new prediction heads, such as YOLO in object detection and Llama models in natural language processing.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(dataloader), epochs=10)

for epoch in range(10):
    for batch in dataloader:
        # Training code here
        scheduler.step()
```

8. CosineAnnealingWarmRestarts

CosineAnnealingWarmRestarts restarts the learning rate from a high value following a cosine annealing schedule at each restart. Combining the benefits of CosineAnnealingLR with warm restarts, this helps to escape local minima by periodically increasing the learning rate. It is effective in training models which may get stuck in local minima, such as ConvNextV2 for image classification and VideoMAE for self supervised video pretraining.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

for epoch in range(50):
    # Training code here
    scheduler.step(epoch)
```

9. PolynomialLR

PolynomialLR decays the learning rate using a polynomial function until it reaches a minimum value, providing a gradual reduction in the learning rate. This can help in fine-tuning the model towards the end of training. It is a bit similar to ExponentialLR, but with a slightly different decay pattern. It has been used in models such as the original Transformer for sequence modeling.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

for epoch in range(num_epochs):
    # Training code here
    scheduler.step()
```

10. LinearLR

LinearLR adjusts the learning rate linearly over a predefined number of iterations or epochs, providing a simple and steady decrease or increase in the learning rate. This isn’t used a lot, as it tends to decay the learning rate too fast, causing premature convergance, perhaps not even to any minima.

```python
import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

for epoch in range(50):
    # Training code here
    scheduler.step()
```

Conclusion

Schedulers are essential tools for optimizing the training process of deep learning models. By leveraging different learning rate schedulers in PyTorch, you can enhance the performance of your models and achieve better results. Best thing to do to understand them better is to plot the actual learning rate schedules for yourself, and try them out.
It is important to know that you will likely only be able to tell which scheduler is best for you after a complete training run, as many schedulers keep learning rate high on purpose until the end. This which can cause intermediate “bad” performance metrics in the middle of training, but “superior” metrics at the end.