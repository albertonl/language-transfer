import os

# CUDA ONLY
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import pytorch_warmup as warmup

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

from models.DecoderTransformer import DecoderTransformer

DATASET_PATH = ''
MODEL_PATH = ''
NUM_SEQUENCES = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

# set_seed(42) # to ensure reproducibility

def config(size, lang):
    global DATASET_PATH
    global MODEL_PATH
    global NUM_SEQUENCES

    DATASET_PATH = f"datasets/mc4_{lang}_train_{size}.bin"
    MODEL_PATH = f"pretrained/mc4_{lang}_scratch_{size}.pt"

    with open(f"datasets/mc4_garbage_train_{size}.stats", 'r') as stats:
        line = stats.readline() # first line is discarded
        line = stats.readline().split()

        if line[-2] == 'sequences:':
            NUM_SEQUENCES = int(line[-1])

def make_batch(batch_size, max_len, device):
    inputs = None
    outputs = None

    with open(DATASET_PATH, 'rb') as entry:
        for seq_len in iter(lambda: entry.read(4), b''):
            sequence = entry.read(int.from_bytes(seq_len, byteorder="little"))

            # Output = Input shifted one token to the right
            input_sequence = torch.tensor(bytearray(sequence)[:-1], dtype=torch.long, device=device)
            output_sequence = torch.tensor(bytearray(sequence)[1:], dtype=torch.long, device=device)

            input_sequence = F.pad(input_sequence, (0, max_len - input_sequence.size()[-1]), 'constant', 0).unsqueeze(0)
            output_sequence = F.pad(output_sequence, (0, max_len - output_sequence.size()[-1]), 'constant', 0).unsqueeze(0)

            if inputs is None and outputs is None:
                inputs = input_sequence.clone().to(device)
                outputs = output_sequence.clone().to(device)
            elif isinstance(inputs, torch.Tensor) and isinstance(outputs, torch.Tensor):
                inputs = torch.cat((inputs, input_sequence), dim=0)
                outputs = torch.cat((outputs, output_sequence), dim=0)

            if inputs.size()[0] == batch_size:
                yield inputs, outputs
                
                inputs = None
                outputs = None
            elif len(inputs) > batch_size:
                raise RuntimeError(f"Length of batch ({len(inputs)}) exceeds maximum batch size ({batch_size})")
    
    return 0, 0 # a return value of 0 indicates EOF (also end of epoch)

def pretrain(device='cuda', num_layers=10, num_heads=10, batch_size=512,
             max_seq_len=1024, peak_lr=2e-4, end_lr=2e-5, weight_decay=0.1, warmup_steps=3000, decay_steps=11445,
             eval_count=10, num_epochs=3):
    
    num_steps = num_epochs * (NUM_SEQUENCES // batch_size)
    eval_steps = num_steps // eval_count
    model = DecoderTransformer(num_layers=num_layers, num_heads=num_heads, device=device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=end_lr)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_steps)

    # Training statistics
    step_intervals, losses, perplexities = [], [], []

    model.train()
    step = 0

    for epoch in range(num_epochs):
        print(f'In epoch {epoch+1}')
        for inputs, outputs in make_batch(batch_size, max_seq_len, device):
            if not isinstance(inputs, torch.Tensor) and not isinstance(outputs, torch.Tensor) and inputs == 0 and outputs == 0:
                break
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), outputs.view(-1))

            if step % eval_steps == 0:
                print(f'Step [{step}/{num_steps}], loss: {loss.item():.4f}, perplexity: {torch.exp(loss).item():.2f}')
                step_intervals.append(step)
                losses.append(loss.item())
                perplexities.append(torch.exp(loss).item())
            
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
    
    print('-- TRAINING SUMMARY --')
    print(f'Step Intervals: {step_intervals}')
    print(f'Evolution of loss: {losses}')
    print(f'Evolution of perplexity: {perplexities}')

    # Save state dict to file
    torch.save(model, MODEL_PATH)

    # Plot loss
    plt.plot(step_intervals, losses, label='Cross-Entropy Loss')
    plt.xlabel('Step of training')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    return model

if __name__ == '__main__':
    set_seed(42) # to ensure reproducibility

    parser = argparse.ArgumentParser(prog='pretrain')
    parser.add_argument('-s', '--size', choices=['6M', '60M', '189M', '600M', '6B'], help='Size of the dataset', required=True)
    parser.add_argument('-l', '--lang', default='garbage', help='Language of the pretraining dataset')
    args = parser.parse_args()

    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config(args.size, args.lang)
    model = pretrain(num_layers=10, num_heads=10, device=device)
