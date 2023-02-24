import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.attention import SpatialTransformer
from typing import List


class UNet(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 n_res_blocks: int,
                 attention_levels: List[int],
                 channel_multipliers: List[int],
                 n_heads: int,
                 tf_layers: int = 1,
                 d_cond: int = 768):
        super().__init__()
