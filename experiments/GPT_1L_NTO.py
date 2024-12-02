import experiment_base
from experiment_base import train_GPT, train_GDM, GPT_CONFIG, GDM_CONFIG

config = GPT_CONFIG
config.next_target_only = True
train_GPT(config)