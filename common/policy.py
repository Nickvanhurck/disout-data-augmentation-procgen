import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .misc_util import orthogonal_init
from .model import GRU

class CategoricalPolicy(nn.Module):
  def __init__(self,
               embedder,
               recurrent,
               action_size):
    """
    embedder: (torch.Tensor) model to extract the embedding for observation
    action_size: number of the categorical actions
    """
    super(CategoricalPolicy, self).__init__()
    self.embedder = embedder
    # small scale weight-initialization in policy enhances the stability        
    self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
    self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
    
    self.recurrent = recurrent
    if self.recurrent:
      self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)
  
  def is_recurrent(self):
    return self.recurrent
  
  def forward(self, x, hx, masks):
    hidden = self.embedder(x)
    if self.recurrent:
      hidden, hx = self.gru(hidden, hx, masks)
    logits = self.fc_policy(hidden)
    log_probs = F.log_softmax(logits, dim=1)
    p = Categorical(logits=log_probs)
    v = self.fc_value(hidden).reshape(-1)
    return p, v, hx
  
  def act(self, x, hx, masks, deterministic=False):
    # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
    # dist = self.dist(actor_features)
    
    hidden = self.embedder(x)
    if self.recurrent:
      hidden, hx = self.gru(hidden, hx, masks)
    logits = self.fc_policy(hidden)
    log_probs = F.log_softmax(logits, dim=1)
    dist = Categorical(logits=log_probs)
    value = self.fc_value(hidden).reshape(-1)
    
    if deterministic:
      action = dist.mode()
    else:
      # don't know if this unsqueeze is exactly necessary?
      action = dist.sample().unsqueeze(-1)
    
    action_log_probs = dist.log_prob(action)
    dist_entropy = dist.entropy().mean()
    
    return value, action, action_log_probs, hx
  
  def get_value(self, x, hx, masks):
    hidden = self.embedder(x)
    value = self.fc_value(hidden).reshape(-1)
    return value
  
  def evaluate_actions(self, x, hx, masks, action):
    hidden = self.embedder(x)
    if self.recurrent:
      hidden, hx = self.gru(hidden, hx, masks)
    logits = self.fc_policy(hidden)
    log_probs = F.log_softmax(logits, dim=1)
    dist = Categorical(logits=log_probs)
    value = self.fc_value(hidden).reshape(-1)
    
    action_log_probs = dist.log_prob(action)
    dist_entropy = dist.entropy().mean()
    
    return value, action_log_probs, dist_entropy, hx

def conv1x1(in_plane, out_plane, stride=1):
  """
  1x1 convolutional layer
  """
  return nn.Conv2d(in_plane, out_plane,
                   kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

def linear(in_features, out_features):
  return nn.Linear(in_features, out_features)

class BasicBlock(nn.Module):
  
  def __init__(self, in_plane, out_plane, stride=1, downsample=None,
               dist_prob=None, block_size=None, alpha=None, nr_steps=None):
    super(BasicBlock, self).__init__()
    self.downsample = downsample
    
    self.bn1 = nn.BatchNorm2d(in_plane)
    self.relu1 = nn.ReLU(inplace=True)
    
    self.conv1 = conv3x3(in_plane, out_plane, stride)
    self.bn2 = nn.BatchNorm2d(out_plane)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out_plane, out_plane)
  
  def forward(self, x):
    residual = x
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv1(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.conv2(x)
    
    if self.downsample:
      residual = self.downsample(residual)
    
    out = x + residual
    return out
