# %%
import torch

from cs336_basics.optimizer import MySGD

# %%


def run_optimizer(weights: torch.nn.Parameter, lr: float = 1, steps: int = 10) -> list[float]:
  opt = MySGD([weights], lr=lr)
  losses = []
  for t in range(steps):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # Compute a scalar loss value.
    losses.append(loss.cpu().item())  # Store the loss value for later inspection.
    print(loss.cpu().item())
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step.
  return losses


# %%
origin_weights = 5 * torch.randn((10, 10))
lrs = [1e1, 1e2, 1e3]
weights_list = [torch.nn.Parameter(origin_weights.clone()) for _ in lrs]
res = {}
steps = 10
for lr, weights in zip(lrs, weights_list):
  print(f"Running optimizer with lr={lr}")
  losses = run_optimizer(weights, lr=lr, steps=steps)
  res[lr] = losses
# %%
print(f"| {' | '.join(f'lr={x}' for x in lrs)} |")
print(f"| {' | '.join('---' for _ in lrs)} |")
for step in range(steps):
  data = [res[lr][step] for lr in lrs]
  print(f"| {' | '.join(f'{x}' for x in data)} |")
