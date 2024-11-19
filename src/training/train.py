import os
import time
import torch
import json
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from src.util import get_time_remaining
from src.visualization import visualize_loss

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')
RESULTS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results')

def model_forward(model, batch, device):
  sequence = batch['input_ids'].to(device)
  input_ids = sequence[:, :-1]
  target_ids = sequence[:, 1:]
  _, loss = model(input_ids, target_ids)
  return loss

def train_model(model, train_dataset, val_dataset, num_epochs=10, starting_epoch=0):
  
  # Training Setup
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  
  # Results
  train_results = {
    'num_epoch_steps': len(train_dataset),
    'num_epochs': 0,
    'losses': []
  }
  val_results = {
    'num_epoch_steps': len(train_dataset),
    'num_epochs': 0,
    'losses': []
  }
  record_steps = len(train_dataset) // 100 # Only validates/saves model losses 100 times (for performance/memory reasons)
  
  # Training Loop
  print(f"Training {model.name} [Device: {device}]")
  for epoch in range(starting_epoch, num_epochs):
    
    start_time = time.time()
    
    train_loss = 0.0
    val_loss = 0.0
    
    train_results['num_epochs'] = epoch
    val_results['num_epochs'] = epoch
    
    for step, batch in enumerate(train_dataset):
      
      model.train()
      
      optimizer.zero_grad()
      
      train_loss = model_forward(model, batch, device)
      train_loss.backward()
      optimizer.step()
      
      train_loss = train_loss.item()
      
      if step % record_steps == 0 or step == len(train_dataset) - 1:
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
          for val_batch in val_dataset:
            batch_val_loss = model_forward(model, val_batch, device)
            batch_val_loss = batch_val_loss.item()
            total_val_loss += batch_val_loss
        val_loss = total_val_loss / len(val_dataset)
        
        # Write both train and val losses at the same step
        total_step = epoch * len(train_dataset) + step
        val_results['losses'].append((total_step, val_loss))
        train_results['losses'].append((total_step, train_loss))
      
      if step <= 1000 or step % 100 == 0 or step == len(train_dataset) - 1:
        time_remaining = get_time_remaining(start_time, step, len(train_dataset))
        print(f"\r\tEpoch {epoch}/{num_epochs} ({100 * epoch / num_epochs:.0f}%) | Step {step}/{len(train_dataset)} | Train Loss: {train_loss:.4f} | Most Recent Val Loss: {val_loss:.4f} | Time Remaining: {time_remaining}", end='')
    
    print(f"\nEpoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    
    torch.save(model.state_dict(), f'{MODEL_BASE_DIR}/{model.name}_epoch_{epoch}.pt')
    
    with open(f'{RESULTS_BASE_DIR}/{model.name}_train_results.json', 'w') as f:
      json.dump(train_results, f)
    with open(f'{RESULTS_BASE_DIR}/{model.name}_val_results.json', 'w') as f:
      json.dump(val_results, f)
    
    visualize_loss((train_results, "Train"), (val_results, "Test"), title=f"{model.name} Training Losses")