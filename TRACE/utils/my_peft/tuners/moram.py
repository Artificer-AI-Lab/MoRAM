# MoRAM (Mixture-of-Ranks Adapter) linear layer — continual-learning router over LoRA rank experts.
# Builds on the TreeLoRA-style Linear base in lora.py (lora_A/lora_B + loranew_A/loranew_B).

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import transpose
from .lora import Linear


class MoRAMLinear(Linear):
    """Mixture-of-ranks adapter layer: official-style top-k routing on L2-normalized rank projections."""

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        r_sum = kwargs.pop("r_sum", 0)
        self.moram_topk: Optional[int] = kwargs.pop("moram_topk", None)
        super().__init__(
            adapter_name,
            in_features,
            out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            r_sum=r_sum,
            **kwargs,
        )

        self.router_temperature = 0.01
        self.use_moram = True
        self.moram_infer_lora_a_thresh = 0.0

    def set_router_temperature(self, temperature: float):
        self.router_temperature = max(float(temperature), 1e-6)

    def set_moram_infer_lora_a_threshold(self, thresh: float = 0.0):
        """Inference-only gate on L2-normalized rank projections (>=). Same threshold for frozen and current-task ranks; 0 disables."""
        self.moram_infer_lora_a_thresh = max(0.0, float(thresh))

    def increment_task(self, adapter_name: Optional[str] = None):
        adapter_name = adapter_name or self.active_adapter
        if adapter_name not in self.loranew_A:
            return

        device = self.loranew_A[adapter_name].weight.device
        dtype = self.loranew_A[adapter_name].weight.dtype
        new_r = self.r.get(adapter_name, 0)
        if new_r == 0:
            return

        old_linear_A = self.lora_A[adapter_name]
        old_linear_B = self.lora_B[adapter_name]
        old_r = old_linear_A.out_features
        total_r = old_r + new_r

        new_A = nn.Linear(self.in_features, total_r, bias=False).to(device=device, dtype=dtype)
        new_B = nn.Linear(total_r, self.out_features, bias=False).to(device=device, dtype=dtype)

        if old_r > 0:
            new_A.weight.data[:old_r] = old_linear_A.weight.data.to(dtype)
            new_B.weight.data[:, :old_r] = old_linear_B.weight.data.to(dtype)
        new_A.weight.data[old_r:] = self.loranew_A[adapter_name].weight.data.to(dtype)
        new_B.weight.data[:, old_r:] = self.loranew_B[adapter_name].weight.data.to(dtype)

        self.lora_A[adapter_name] = new_A
        self.lora_B[adapter_name] = new_B
        for param in self.lora_A[adapter_name].parameters():
            param.requires_grad = False
        for param in self.lora_B[adapter_name].parameters():
            param.requires_grad = False

        nn.init.kaiming_uniform_(self.loranew_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B[adapter_name].weight)

    def forward(self, x: torch.Tensor):
        """L2-normalize projections, top-k sparse logits, softmax / τ, optional inference gate, Δ = (w⊙z)Bᵀ."""
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        adapter = self.active_adapter
        if adapter not in self.loranew_A.keys():
            return result.to(previous_dtype)

        dropped = self.lora_dropout[adapter](x.to(self.loranew_A[adapter].weight.dtype))
        rank_per_task = self.r[adapter]
        scale = self.scaling[adapter]

        W_old_A = self.lora_A[adapter].weight
        W_new_A = self.loranew_A[adapter].weight
        total_old_r = W_old_A.shape[0]

        if total_old_r > 0:
            W_A = torch.cat([W_old_A, W_new_A], dim=0)
        else:
            W_A = W_new_A

        projection = F.linear(dropped, W_A)
        normed = F.normalize(projection, p=2, dim=-1, eps=1e-8)

        topk_ref = self.moram_topk if self.moram_topk is not None else rank_per_task
        k = max(int(topk_ref), 2)
        r_all = normed.shape[-1]
        k_eff = min(k, r_all)
        topk_val, topk_idx = torch.topk(normed, k=k_eff, dim=-1)
        masked_logits = normed.new_full(normed.shape, float("-inf"))
        masked_logits.scatter_(-1, topk_idx, topk_val)
        weights = F.softmax(masked_logits / self.router_temperature, dim=-1)

        if not self.training:
            gate = torch.ones_like(weights, dtype=weights.dtype)
            t = self.moram_infer_lora_a_thresh
            if t > 0:
                if total_old_r > 0:
                    gate[..., :total_old_r] = (
                        normed[..., :total_old_r] >= t
                    ).to(weights.dtype)
                if rank_per_task > 0:
                    gate[..., total_old_r:] = (
                        normed[..., total_old_r:] >= t
                    ).to(weights.dtype)
            weights = weights * gate

        weighted_proj = projection * weights

        W_B_old = self.lora_B[adapter].weight
        W_B_new = self.loranew_B[adapter].weight * scale
        if total_old_r > 0:
            B_combined = torch.cat([W_B_old, W_B_new], dim=1)
        else:
            B_combined = W_B_new

        delta = F.linear(weighted_proj, B_combined)
        result = result + delta.to(result.dtype)
        return result.to(previous_dtype)
