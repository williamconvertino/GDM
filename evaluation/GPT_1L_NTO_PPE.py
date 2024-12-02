import evaluation_base
from evaluation_base import GPT_CONFIG, GDM_CONFIG, evaluate_GDM, evaluate_GPT

config = GPT_CONFIG
config.next_target_only = True
config.use_ppe_attn = True
evaluate_GPT(config)