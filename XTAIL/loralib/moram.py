import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class MoRAM_Linear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        module_name=None,
        lora_threshold: float = 1.0,
        temp: float = 0.01,
        topk: Optional[int] = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        self.module_name = module_name
        self.rank = r
        self.init_rank = r
        self.in_features = in_features
        self.out_features = out_features
        self.temp = float(temp)

        self.lora_threshold = float(lora_threshold)
        self.topk = int(topk) if topk is not None else r

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_scaling = 1
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.old_lora_A = None
        self.old_lora_B = None
        self.old_rank = 0

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def set_old_rank(self, old_rank):
        self.old_rank = old_rank

    def increment_task(self):
        if self.lora_A is not None:
            if self.old_lora_A is not None:
                self.old_lora_A = nn.Parameter(torch.cat([
                    self.old_lora_A.data.cuda().clone(),
                    self.lora_A.data.cuda().clone(),
                ], dim=0))
                self.old_lora_B = nn.Parameter(torch.cat([
                    self.old_lora_B.data.cuda().clone(),
                    self.lora_B.data.cuda().clone(),
                ], dim=1))
            else:
                self.old_lora_A = nn.Parameter(self.lora_A.data.cuda().clone())
                self.old_lora_B = nn.Parameter(self.lora_B.data.cuda().clone())

            self.old_lora_A.requires_grad = False
            self.old_lora_B.requires_grad = False

            self.old_rank += self.rank

        if self.init_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.init_rank, self.in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.init_rank)))
            self.register_buffer("gate", torch.zeros(1, self.init_rank, device=self.weight.device, dtype=self.weight.dtype))
            self.n_gate = 0
            self.lora_scaling = 1
            self.rank = self.init_rank
            self.r = self.init_rank
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.first_iter = True
            
    def forward(self, x: torch.Tensor, get_cur_feat=False, **kwargs):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.old_lora_A is None and self.lora_A is None:
            return result, 0

        if self.old_lora_A is not None and self.lora_A is not None:
            combined_lora_A = torch.cat([self.old_lora_A, self.lora_A], dim=0)
            B_combined = torch.cat(
                [self.old_lora_B, self.lora_B * self.lora_scaling], dim=1
            )
        elif self.old_lora_A is not None:
            combined_lora_A = self.old_lora_A
            B_combined = self.old_lora_B
        elif self.lora_A is not None:
            combined_lora_A = self.lora_A
            B_combined = self.lora_B * self.lora_scaling
        else:
            return result, 0

        xd = self.lora_dropout(x)
        projection = xd @ combined_lora_A.T

        normed_projection = F.normalize(projection, p=2, dim=-1)

        k = max(self.topk, 2)
        topk_values, topk_indices = torch.topk(normed_projection, k=k, dim=-1)
        masked_logits = torch.full_like(normed_projection, float('-inf'))
        masked_logits.scatter_(-1, topk_indices, topk_values)
        weights = F.softmax(masked_logits / self.temp, dim=-1)

        if not self.training and self.lora_threshold > 0:
            gate_mask = (normed_projection >= self.lora_threshold).float()
            weights = weights * gate_mask

        weighted_proj = projection * weights
        lora_delta = torch.matmul(weighted_proj, B_combined.T)

        result = result + lora_delta

        loss = 0

        return result, loss

    # --- Backup: two A projections (separate xd@old_A.T and xd@new_A.T) + dense B matmul ---
    # def forward_backup(self, x: torch.Tensor, get_cur_feat=False, **kwargs):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
    #
    #     result = F.linear(x, T(self.weight), bias=self.bias)
    #
    #     if self.old_lora_A is None and self.lora_A is None:
    #         return result, 0
    #
    #     if self.old_lora_A is not None and self.lora_A is not None:
    #         combined_lora_A = torch.cat([self.old_lora_A, self.lora_A], dim=0)
    #     elif self.old_lora_A is not None:
    #         combined_lora_A = self.old_lora_A
    #     elif self.lora_A is not None:
    #         combined_lora_A = self.lora_A
    #     else:
    #         return result, 0
    #
    #     projection = (self.lora_dropout(x) @ combined_lora_A.T)
    #
    #     normed_projection = F.normalize(projection, p=2, dim=-1)
    #
    #     topk_values, topk_indices = torch.topk(normed_projection, k=max(self.topk, 2), dim=-1)
    #     masked_logits = torch.full_like(normed_projection, float('-inf'))
    #     masked_logits.scatter_(-1, topk_indices, topk_values)
    #     weights = F.softmax(masked_logits / self.temp, dim=-1)
    #
    #     if not self.training and self.lora_threshold > 0:
    #         gate_mask = (normed_projection >= self.lora_threshold).float()
    #         weights = weights * gate_mask
    #
    #     if self.old_lora_A is not None and self.lora_A is not None:
    #         old_weight, new_weight = torch.split(weights, [self.old_rank, self.init_rank], dim=-1)
    #     elif self.old_lora_A is not None:
    #         old_weight, new_weight = weights, None
    #     elif self.lora_A is not None:
    #         old_weight, new_weight = None, weights
    #     else:
    #         old_weight, new_weight = None, None
    #
    #     if new_weight is not None:
    #         if "out_proj" in self.module_name:
    #             self.gate.data = (
    #                 self.gate.data * self.n_gate + torch.mean(new_weight.detach().clone(), dim=0)
    #             ) / (self.n_gate + 1)
    #         else:
    #             self.gate.data = (
    #                 self.gate.data * self.n_gate + torch.mean(new_weight.detach().clone(), dim=(0, 1))
    #             ) / (self.n_gate + 1)
    #         self.n_gate += 1
    #     self.first_iter = False
    #
    #     if self.old_lora_A is not None and self.old_lora_B is not None and old_weight is not None:
    #         old_lora_contribution = (self.lora_dropout(x) @ self.old_lora_A.T) * old_weight @ self.old_lora_B.T
    #     else:
    #         old_lora_contribution = torch.zeros_like(result)
    #
    #     if self.lora_A is not None and self.lora_B is not None and new_weight is not None:
    #         lora_contribution = (
    #             (self.lora_dropout(x) @ self.lora_A.T) * new_weight @ self.lora_B.T
    #         ) * self.lora_scaling
    #     else:
    #         lora_contribution = torch.zeros_like(result)
    #
    #     result = result + old_lora_contribution + lora_contribution
    #
    #     if self.training:
    #         total_ranks = weights.shape[-1]
    #         P = weights.reshape(-1, total_ranks).mean(dim=0)
    #         loss = total_ranks * (P ** 2).sum()
    #     else:
    #         loss = 0
    #
    #     return result, loss

