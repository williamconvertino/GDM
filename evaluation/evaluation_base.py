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
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer
from src.util import load_mrm
from src.evaluation import evaluate_model_generation

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
  
def evaluate_model_with_config(config, model_type, max_epoch=None):
  
  if model_type == "GPT":
    model = GPT(config)
  elif model_type == "GDM":
    model = GDM(config)
  else:
    raise ValueError("Invalid model type")

  model = load_mrm(model, max_epoch=max_epoch)

  tokenizer = TinyStoriesTokenizer()
  val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=config.context_size)
  
  evaluate_model_generation(model, tokenizer, val_dataset)
  
def evaluate_GPT(config, max_epoch=None):
  evaluate_model_with_config(config, "GPT", max_epoch)
  
def evaluate_GDM(config, max_epoch=None):
  evaluate_model_with_config(config, "GDM", max_epoch)