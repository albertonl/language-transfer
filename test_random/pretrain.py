import os

# CUDA ONLY
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import pytorch_warmup as warmup

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from models.DecoderTransformer import DecoderTransformer

DATASET_PATH = "datasets/mc4_garbage_train_6M.bin"
MODEL_PATH = "pretrained/mc4_garbage_scratch_6M.pt"
NUM_SEQUENCES = 6656

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

set_seed(42) # to ensure reproducibility

def make_batch(batch_size, device):
    inputs = None
    outputs = None

    with open(DATASET_PATH, 'rb') as entry:
        for seq_len in iter(lambda: entry.read(4), b''):
            sequence = entry.read(int.from_bytes(seq_len, byteorder="little"))

            # Output = Input shifted one token to the right
            input_sequence = torch.tensor(bytearray(sequence)[:-1], dtype=torch.long, device=device).unsqueeze(0)
            output_sequence = torch.tensor(bytearray(sequence)[1:], dtype=torch.long, device=device).unsqueeze(0)

            if inputs is None and outputs is None:
                inputs = input_sequence.clone().to(device)
                outputs = output_sequence.clone().to(device)
            elif isinstance(inputs, torch.Tensor) and isinstance(outputs, torch.Tensor):
                inputs = torch.cat((inputs, input_sequence), dim=0)
                outputs = torch.cat((outputs, output_sequence), dim=0)

            if inputs.size()[0] == batch_size:
                yield inputs, outputs
                # yield torch.tensor(inputs).to(device), torch.tensor(outputs).to(device)
                inputs = None
                outputs = None
            elif len(inputs) > batch_size:
                raise RuntimeError(f"Length of batch ({len(inputs)}) exceeds maximum batch size ({batch_size})")
    
    return 0, 0 # a return value of 0 indicates EOF (also end of epoch)

vocab_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DecoderTransformer(num_layers=1, num_heads=1)
model.to(device)

# Hyperparameters
batch_size = 512
initial_lr = 0
peak_lr = 2e-4
end_lr = 2e-5
warmup_steps = 3000
decay_steps = 11445
eval_steps = 5000
num_epochs = 3

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
num_steps = num_epochs * (NUM_SEQUENCES // batch_size)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=end_lr)
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_steps)

# Plot data
plot_steps = []
plot_losses = []

model.train()
step = 0

for epoch in range(num_epochs):
    print(f"In epoch {epoch+1}")
    for inputs, outputs in make_batch(batch_size, device):
        if not isinstance(inputs, torch.Tensor) and not isinstance(outputs, torch.Tensor) and inputs == 0 and outputs == 0:
            break

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), outputs.view(-1))
        
        if step % eval_steps == 0:
            print(f'Step [{step}/{num_steps}], loss: {loss.item():.4f}, perplexity: {torch.exp(loss).item():.2f}')
            plot_steps.append(step)
            plot_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

        with warmup_scheduler.dampening():
            scheduler.step()
        
        step += 1

        if step == num_steps:
            break
    if step == num_steps:
        break

print(f'Step [{step}/{num_steps}], loss: {loss.item():.4f}, perplexity: {torch.exp(loss).item():.2f}')

# Save state dict to file
torch.save(model, MODEL_PATH)

# Plot losses
plt.plot(plot_steps, plot_losses, label="Loss")
plt.xlabel("Step of training")
plt.legend(loc="upper right")
plt.show()