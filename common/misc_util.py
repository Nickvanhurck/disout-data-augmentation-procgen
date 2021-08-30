import gym
import numpy as np
import torch
import torch.nn as nn

def set_global_seeds(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def set_global_log_levels(level):
  gym.logger.set_level(level)

def orthogonal_init(module, gain=torch.nn.init.calculate_gain('relu')):
  if isinstance(module, nn.Linear) or isinstance(module, torch.nn.Conv2d):
    torch.nn.init.orthogonal_(module.weight.data, gain)
    torch.nn.init.constant_(module.bias.data, 0)
  return module

def xavier_uniform_init(module, gain=1.0):
  if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
    torch.nn.init.xavier_uniform_(module.weight.data, gain)
    torch.nn.init.constant_(module.bias.data, 0)
  return module

def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
  lr = init_lr * (1 - (timesteps / max_timesteps))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer

def get_n_params(model):
  return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'
