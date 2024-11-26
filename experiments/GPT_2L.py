import notebook_setup
import torch
from src.models import PGD, GPT, gdGPT, GPTConfig
from src.datasets import TinyStoriesDataset
from src.tokenizers import TinyStoriesTokenizer
from src.training import train_model

torch.manual_seed(42)

tokenizer = TinyStoriesTokenizer()

CONTEXT_SIZE = 256
D_EMBED = 512
N_LAYER = 2
N_HEAD = 8

VOCAB_SIZE = len(tokenizer)

config = GPTConfig(
  context_size=CONTEXT_SIZE,
  d_embed=D_EMBED,
  n_layer=N_LAYER,
  n_head=N_HEAD,
  vocab_size=VOCAB_SIZE
)

model = GPT(config)

train_dataset = TinyStoriesDataset(tokenizer, 'train', context_size=CONTEXT_SIZE)
val_dataset = TinyStoriesDataset(tokenizer, 'val', context_size=CONTEXT_SIZE)

train_model(model, train_dataset, val_dataset)