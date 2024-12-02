import os
import sys
import subprocess

# Add parent directory to path

path = os.path.abspath(os.path.join(os.getcwd(), '../'))
if path not in sys.path:
  sys.path.append(path)
  
# Install requirements
    
try:
  subprocess.check_call(["pip", "install", "-r", "../requirements.txt", "--quiet"])
  print("Requirements installed")
except:
  print("Failed to install requirements")

# Basic experiment setup

import torch
from src.models import GPT, GDM, Config
from src.training import train_model
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer

DEFAULT_VOCAB_SIZE = 10002
  
GPT_CONFIG = Config(
  
  model_type='GPT',
  
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  
  d_embed=512,
  n_layer=1,
  n_head=8,
  
  dropout=0.1,
  
  attn_kernel_fn='softmax',
  
  use_ff=True,
  use_ppe=False,
  use_nto=False
)

GDM_CONFIG = Config(
  
  model_type='GDM',
  
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  
  d_embed=512,
  n_layer=1,
  n_head=8,
  
  dropout=0.1,
  
  attn_kernel_fn='softmax',
  
  use_ff=True,
  use_ppe=False,
  use_nto=False
)
  
def run_experiment(config, seed=0):
  
  torch.manual_seed(seed)
  
  if config.model_type == "GPT":
    model = GPT(config)
  elif config.model_type == "GDM":
    model = GDM(config)
  else:
    raise ValueError("Invalid model type")

  tokenizer = TinyStoriesTokenizer()
  train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=config.context_size)
  val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=config.context_size)
  
  train_model(model, train_dataset, val_dataset)