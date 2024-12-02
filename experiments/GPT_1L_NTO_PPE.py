import experiment_base
from experiment_base import train_GPT, train_GDM, GPT_CONFIG, GDM_CONFIG

config = GPT_CONFIG
config.n_layer = 2
config.next_target_only = True
config.use_ppe_attn = True
train_GPT(config)