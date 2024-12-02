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
  d_embed=512,
  n_layer=1,
  n_head=8,
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  use_ff=True,
  attn_kernel_fn='softmax',
  dropout=0.1,
  next_target_only=False,
  use_ppe_attn=False
)

GDM_CONFIG = Config(
  d_embed=512,
  n_layer=1,
  n_head=8,
  context_size=256,
  vocab_size=DEFAULT_VOCAB_SIZE,
  use_ff=False,
  attn_kernel_fn='softmax',
  next_target_only=False
)
  
def train_model_with_config(config, model_type, seed=0):
  
  torch.manual_seed(seed)
  
  if model_type == "GPT":
    model = GPT(config)
  elif model_type == "GDM":
    model = GDM(config)
  else:
    raise ValueError("Invalid model type")

  tokenizer = TinyStoriesTokenizer()
  train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=config.context_size)
  val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=config.context_size)
  
  train_model(model, train_dataset, val_dataset)
  
def train_GPT(config, seed=0):
  train_model_with_config(config, "GPT", seed)
  
def train_GDM(config, seed=0):
  train_model_with_config(config, "GDM", seed)