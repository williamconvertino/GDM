import experiment_base
from experiment_base import train_GPT, train_GDM, GPT_CONFIG, GDM_CONFIG

config = GDM_CONFIG
config.next_target_only = True
config.use_ff = True
train_GDM(config)