import os
import numpy as np
from open_clip import create_model_and_transforms, get_tokenizer
import torch
from typing import Optional, Tuple

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import loralib as lora
from torch.nn.parameter import Parameter
import torch.nn as nn
import utils
import logging
from torch.utils.tensorboard import SummaryWriter
import copy
import json

# Define a subclass to override the forward method
class moram_CLIPModel(torch.nn.Module):
    def __init__(self, original_model):
        super(moram_CLIPModel, self).__init__()
        
        self.output_dict = original_model.output_dict
        self.visual = moram_VisionTransformer(original_model.visual)

        self.transformer = moram_Transformer(original_model.transformer)
        # fall back to default context length if missing
        self.context_length = getattr(original_model, "context_length", 77)
        self.vocab_size = original_model.vocab_size
        self.token_embedding = original_model.token_embedding
        self.positional_embedding = original_model.positional_embedding
        self.ln_final = original_model.ln_final
        self.text_projection = original_model.text_projection
        self.attn_mask = original_model.attn_mask

        self.logit_scale = original_model.logit_scale
        # fall back to None if missing
        self.logit_bias = getattr(original_model, "logit_bias", None)
        
        # Initialize visual and textual buffers
        self.register_buffer("visual_prototypes", None)  # Initially empty
        self.register_buffer("textual_prototypes", None)  # Initially empty


    def update_prototypes(self, visual_prototypes=None, textual_prototypes=None):
        if visual_prototypes is not None:
            if self.visual_prototypes is None:
                self.visual_prototypes = visual_prototypes
            else:
                self.visual_prototypes = torch.cat([self.visual_prototypes, visual_prototypes], dim=0)
            print(f"visual prototype updated: {self.visual_prototypes.shape}")
        
        if textual_prototypes is not None:
            if self.textual_prototypes is None:
                self.textual_prototypes = textual_prototypes
            else:
                self.textual_prototypes = torch.cat([self.textual_prototypes, textual_prototypes], dim=0)
            print(f"texual prototype updated: {self.textual_prototypes.shape}")
            
    def encode_text(self, text, normalize=False, reg_prename=False, get_cur_feat=False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, prename_loss = self.transformer(x, attn_mask=self.attn_mask, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        output = F.normalize(x, dim=-1) if normalize else x
        
        return output, prename_loss
        
    def encode_image(self, image, normalize=False, get_feat=False,get_cur_feat=False):
        features, loss = self.visual(image, get_cur_feat=get_cur_feat)
        output = F.normalize(features, dim=-1) if normalize else features
        return output, loss


    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        get_feat=False,get_cur_feat=False
    ):
        features, _ = self.visual(image, get_feat=get_feat, get_cur_feat=get_cur_feat) if image is not None else None
        image_features = F.normalize(features, dim=-1) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()
        
        
class moram_VisionTransformer(torch.nn.Module):
    def __init__(self, original_model):
        super(moram_VisionTransformer, self).__init__()
        
        self.output_tokens = original_model.output_tokens
        self.image_size = original_model.image_size
        self.patch_size = original_model.patch_size
        self.grid_size = original_model.grid_size
        self.output_dim = original_model.output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = original_model.input_patchnorm

        if self.input_patchnorm:
            self.patchnorm_pre_ln = original_model.pathnorm_pre_ln
            self.conv1 = original_model.conv1
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = original_model.conv1

        # class embeddings and positional embeddings
        self.class_embedding = original_model.class_embedding
        self.positional_embedding = original_model.positional_embedding

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = original_model.patch_dropout

        self.ln_pre = original_model.ln_pre
        self.transformer = moram_Transformer(original_model.transformer)

        self.global_average_pool = original_model.global_average_pool
        self.attn_pool = original_model.attn_pool
        self.ln_post = original_model.ln_post
        self.proj = original_model.proj
        
    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

        
    def forward(
            self, x: torch.Tensor,
            get_feat=False,get_cur_feat=False
        ):

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, loss = self.transformer(x, get_feat=get_feat,get_cur_feat=get_cur_feat)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        return pooled, loss

class moram_Transformer(nn.Module):
    def __init__(self, original_model):
        super(moram_Transformer, self).__init__()

        super().__init__()
        self.width = original_model.width
        self.layers = original_model.layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            moram_ResidualAttentionBlock(original_model.resblocks[i])
            for i in range(len(original_model.resblocks))
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, reg_prename=False, get_feat=False,get_cur_feat=False):
        losses = []
        for r in self.resblocks:
            x, prename_loss = r(x, attn_mask=attn_mask, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)

        return x, sum(losses)/len(losses) if len(losses) > 0 else 0

class moram_ResidualAttentionBlock(nn.Module):
    def __init__(self, original_model):
        super(moram_ResidualAttentionBlock, self).__init__()

        self.ln_1 = original_model.ln_1
        self.attn = copy.deepcopy(original_model.attn)
        self.ls_1 = original_model.ls_1

        self.ln_2 = original_model.ln_2
        self.mlp = copy.deepcopy(original_model.mlp)
        self.ls_2 = original_model.ls_2

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            reg_prename=False, get_cur_feat=False
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        if isinstance(self.attn, nn.MultiheadAttention):
            assert reg_prename == False, "MultiheadAttention does not support PreName"
            return self.attn(
                q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
            )[0], 0
        else:
            attn_output, loss = self.attn(
                q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask, 
                reg_prename=reg_prename, get_cur_feat=get_cur_feat
            )
            return attn_output, loss

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            reg_prename=False, get_cur_feat=False,
            **kwargs
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        losses = []
        attn_output, prename_loss = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, 
                                                    attn_mask=attn_mask, reg_prename=reg_prename,
                                                    get_cur_feat=get_cur_feat)
        losses.append(prename_loss)
        x = q_x + self.ls_1(attn_output)
        
        x_mlp = self.ln_2(x)
        
        for layer in self.mlp:
            if isinstance(layer, lora.MoRAM_Linear):
                x_mlp, prename_loss = layer(x_mlp, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
                losses.append(prename_loss)
            else:
                x_mlp = layer(x_mlp)
        x = x + self.ls_2(x_mlp)

        return x, sum(losses)/len(losses) if len(losses) > 0 else 0

class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            module_name: str,  # New parameter to hold the module name
            enable_lora: list = ['q', 'k', 'v', 'o'],
            r: int = 0,
            lora_temp: float = 0.01,
            lora_threshold: float = 1.0,
            lora_topk: Optional[int] = None,
            coverage: Optional[float] = None,
            **kwargs
        ):
        super(PlainMultiheadAttentionLoRA, self).__init__()

        self.module_name = module_name  # Store the module name

        self.lora_temp = float(lora_temp)
        self.lora_threshold = float(lora_threshold)
        self.lora_topk = int(lora_topk) if lora_topk is not None else r
        coverage = 0.99 if coverage is None else float(coverage)
        self.router_coverage = min(max(coverage, 0.0), 1.0)

        # Copy essential parameters from the existing MultiheadAttention
        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = self.embed_dim // self.num_heads
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.dropout = existing_mha.dropout

        # Initialize the projection layers (q_proj, k_proj, v_proj, out_proj)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        # Initialize parameters from the existing attention layer
        self._init_from_existing_mha(existing_mha)

        # LoRA-ify the layers specified in `enable_lora`
        self._apply_lora(enable_lora, r)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention
        
    def _init_from_existing_mha(self, existing_mha):
        """
        Initializes the new attention module's weights with the existing MultiheadAttention weights.
        """
        # Extract existing weights and biases
        existing_weight = existing_mha.in_proj_weight.data
        existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

        # Initialize q_proj
        self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
        if existing_bias is not None:
            self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

        # Initialize k_proj
        self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
        if existing_bias is not None:
            self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

        # Initialize v_proj
        self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
        if existing_bias is not None:
            self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])

        # Initialize out_proj
        self.out_proj.weight.data.copy_(existing_mha.out_proj.weight.data)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(existing_mha.out_proj.bias.data)

    def _apply_lora(self, enable_lora, r):
        """
        Converts the specified projections into LoRA-enhanced projections.
        Ensure the pretrained weights are copied to the LoRA layers.
        """
        lora_kwargs = {
            'temp': self.lora_temp,
            'lora_threshold': self.lora_threshold,
            'topk': self.lora_topk,
        }
        if 'q' in enable_lora:
            lora_q_proj = lora.MoRAM_Linear(
                self.q_proj.in_features,
                self.q_proj.out_features,
                r=r,
                module_name=self.module_name + ".q_proj",
                **lora_kwargs,
            )
            lora_q_proj.weight.data.copy_(self.q_proj.weight.data)
            if self.q_proj.bias is not None:
                lora_q_proj.bias.data.copy_(self.q_proj.bias.data)
            self.q_proj = lora_q_proj

        if 'k' in enable_lora:
            lora_k_proj = lora.MoRAM_Linear(
                self.k_proj.in_features,
                self.k_proj.out_features,
                r=r,
                module_name=self.module_name + ".k_proj",
                **lora_kwargs,
            )
            lora_k_proj.weight.data.copy_(self.k_proj.weight.data)
            if self.k_proj.bias is not None:
                lora_k_proj.bias.data.copy_(self.k_proj.bias.data)
            self.k_proj = lora_k_proj

        if 'v' in enable_lora:
            lora_v_proj = lora.MoRAM_Linear(
                self.v_proj.in_features,
                self.v_proj.out_features,
                r=r,
                module_name=self.module_name + ".v_proj",
                **lora_kwargs,
            )
            lora_v_proj.weight.data.copy_(self.v_proj.weight.data)
            if self.v_proj.bias is not None:
                lora_v_proj.bias.data.copy_(self.v_proj.bias.data)
            self.v_proj = lora_v_proj

        if 'o' in enable_lora:
            lora_out_proj = lora.MoRAM_Linear(
                self.out_proj.in_features,
                self.out_proj.out_features,
                r=r,
                module_name=self.module_name + ".out_proj",
                **lora_kwargs,
            )
            lora_out_proj.weight.data.copy_(self.out_proj.weight.data)
            if self.out_proj.bias is not None:
                lora_out_proj.bias.data.copy_(self.out_proj.bias.data)
            self.out_proj = lora_out_proj
            
    def forward(self, query, key, value, need_weights=False, key_padding_mask=None, attn_mask=None, is_causal=False, average_attn_weights=True,
                reg_prename=False, get_cur_feat=False
                ):
        """
        Forward pass with LoRA-augmented attention layers.
        """        
        losses =[]
        
        is_batched = query.dim() == 3
        
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        # Compute the projections using LoRA-augmented K and V
        tgt_len, bsz, embed_dim = query.shape
        source_len, _, _ = key.shape
        
        num_heads = self.num_heads
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        
        if isinstance(self.q_proj, lora.MoRAM_Linear):
            q, prename_loss = self.q_proj(query, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            q = self.q_proj(query)
            
        if isinstance(self.k_proj, lora.MoRAM_Linear):
            k, prename_loss = self.k_proj(key, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            k = self.k_proj(key)
            
        if isinstance(self.v_proj, lora.MoRAM_Linear):
            v, prename_loss = self.v_proj(value, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            v = self.v_proj(value)

        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        
        source_len = k.size(1)
        
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout
            
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, source_len, head_dim)
        v = v.view(bsz, num_heads, source_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        if isinstance(self.out_proj, lora.MoRAM_Linear):
            attn_output, prename_loss = self.out_proj(attn_output, reg_prename=reg_prename, get_cur_feat=get_cur_feat)
            losses.append(prename_loss)
        else:
            attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            
        
        return attn_output, sum(losses)/len(losses) if len(losses) > 0 else 0

    def train(self, mode: bool = True):
        """
        Overrides the train method to ensure LoRA-specific behavior is applied when training.
        """
        super().train(mode)
        # LoRA-specific behavior can be added here if needed.


            
def apply_lora(cfg, model, encoder=''):
    """
    Applies LoRA to the specified modules within a given model based on the configuration.

    Args:
    - cfg: Configuration dictionary containing settings like target modules, rank, etc.
    - model: The model (CLIP) to which LoRA should be applied.
    """
    lora_temp = float(cfg.get('temp', 0.01))
    lora_threshold = float(cfg.get('prune_thre', 1.0))
    lora_topk = int(cfg.get('topk', cfg['rank']))
    router_coverage = float(cfg.get('router_coverage', 0.99))
    router_coverage = min(max(router_coverage, 0.0), 1.0)

    enable_lora = []
    if 'q_proj' in cfg['target_modules']:
        enable_lora.append('q')
    if 'k_proj' in cfg['target_modules']:
        enable_lora.append('k')
    if 'v_proj' in cfg['target_modules']:
        enable_lora.append('v')
    if 'out_proj' in cfg['target_modules']:
        enable_lora.append('o')

    for i, block in enumerate(model.resblocks):
        for name, submodule in block.named_children():
            full_name = f"{encoder}transformer.resblocks.{i}.{name}"
            
            # Apply LoRA to attention modules
            if any(['proj' in m for m in cfg['target_modules']]):
                if isinstance(submodule, nn.MultiheadAttention):
                    # Replace with LoRA-enhanced MultiheadAttention
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        existing_mha=submodule, 
                        module_name=full_name,  # Pass the module name for tracking
                        enable_lora=enable_lora, 
                        r=cfg['rank'], 
                        lora_temp=lora_temp,
                        lora_threshold=lora_threshold,
                        lora_topk=lora_topk,
                        coverage=router_coverage,
                    )
                    setattr(block, name, new_multi_head_lora)

            # Apply LoRA to FFN layers
            if any(['ffn' in m for m in cfg['target_modules']]):
                if 'mlp' in name:
                    ffn_target_modules = []
                    if 'ffn_in' in cfg['target_modules']:
                        ffn_target_modules.append('c_fc')
                    if 'ffn_out' in cfg['target_modules']:
                        ffn_target_modules.append('c_proj')

                    for ffn_name, ffn_submodule in submodule.named_children():
                        ffn_full_name = f"{full_name}.{ffn_name}"
                        if ffn_name in ffn_target_modules and isinstance(ffn_submodule, nn.Linear):
                            # Apply LoRA to this specific linear layer (c_fc or c_proj)
                            lora_layer = lora.MoRAM_Linear(
                                ffn_submodule.in_features,
                                ffn_submodule.out_features,
                                r=cfg['rank'],
                                merge_weights=False,
                                module_name=ffn_full_name,
                                temp=lora_temp,
                                lora_threshold=lora_threshold,
                                topk=lora_topk,
                            )
                            # Initialize LoRA layer with the existing weights
                            lora_layer.weight.data = ffn_submodule.weight.data.clone()
                            if ffn_submodule.bias is not None:
                                setattr(lora_layer, 'bias', Parameter(ffn_submodule.bias.data.clone()))

                            # Replace the original linear layer with the LoRA-augmented one
                            setattr(submodule, ffn_name, lora_layer)



def get_moram(cfg):
    logging.info(f"Loading CLIP (backbone: {cfg['backbone_type']}.{cfg['pretrained_weight']})")

    clip_model, train_trfm, test_trfm = create_model_and_transforms(cfg['backbone_type'], pretrained=cfg['pretrained_weight'])
    tokenizer = get_tokenizer(cfg['backbone_type'])
    if cfg['zero_shot']:
        return clip_model, train_trfm, test_trfm, tokenizer
    
    clip_params = sum(p.numel() for p in clip_model.parameters())
    logging.info(f"Total number of CLIP parameters: {clip_params}")
    
    clip_model = clip_model.to('cuda')
    clip_model = moram_CLIPModel(clip_model)
    clip_model = clip_model.to('cuda')
    
    # Apply LoRA to each MultiheadAttention layer in the model
    if cfg['target_encoder'] == 'vision':
        apply_lora(cfg, clip_model.visual.transformer, encoder='visual.')
    elif cfg['target_encoder'] == 'text':
        apply_lora(cfg, clip_model.transformer)
    elif cfg['target_encoder'] == 'all':
        apply_lora(cfg, clip_model.visual.transformer, encoder='visual.')
        apply_lora(cfg, clip_model.transformer)
    else:
        raise ValueError(f"Invalid target_encoder: {cfg['target_encoder']}")
            
    model = clip_model
    # Freeze all parameters except the LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'lora_' in name:
            param.requires_grad = True
        
        if 'weight' in name or 'bias' in name:
            param.requires_grad = False
            
        if 'old' in name:
            param.requires_grad = False
        
            

    model = model.to('cuda')
    
    return model, train_trfm, test_trfm, tokenizer

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param  


def finetune(args, model, tokenizer, dataset_name, dataset, train_loader):
    model = model.cuda()
    try:
        if len(args.seen_classes) > 0 and not args.mtil_eval:
            for name, module in model.named_modules():
                if isinstance(module, lora.MoRAM_Linear):
                    module.increment_task()
            logging.info('finished increment task, saved old lora, and initialized new lora')
    except Exception:
        if len(args.seen_classes) > 0:
            for name, module in model.named_modules():
                if isinstance(module, lora.MoRAM_Linear):
                    module.increment_task()
            logging.info('finished increment task, saved old lora, and initialized new lora')

    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'lora_' in name:
            param.requires_grad = True
        if 'old' in name:
            param.requires_grad = False

    ckpt_path = os.path.join(args.save, f"{dataset_name}.pth") if args.save is not None else None
    train_model(args, model, tokenizer, dataset_name, dataset, train_loader)

    if args.save is not None:
        if isinstance(model, torch.nn.DataParallel):
            to_save_model = model.module
        else:
            to_save_model = model
        path = ckpt_path if ckpt_path is not None else os.path.join(args.save, f"{dataset_name}.pth")
        utils.torch_save_lora(to_save_model, path, lora.MoRAM_Linear)

    template = dataset.template[0]
    class_names = dataset.classnames
    args.seen_classes += [template.format(l) for l in class_names]
    
    
def train_model(args, model, tokenizer, dataset_name, dataset, train_loader):
    writer = SummaryWriter(log_dir=os.path.join("runs", args.save, dataset_name))
    
    loss_interval = len(train_loader) 
    num_batches = len(train_loader)
    total_iterations = args.iterations  # 1000
    logging.info(f"Iterations per epoch: {num_batches}")
    logging.info(f"Total iterations: {total_iterations}")

    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    logging.info(f'The number of Total Trainable Parameters------------------: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logging.info(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.9999)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    logging.info(f"Using devices {devices}")
    model = torch.nn.DataParallel(model, device_ids=devices) 

    template = dataset.template[0]
    class_names = dataset.classnames
    seen_classes = copy.deepcopy(args.seen_classes)

    if len(seen_classes) > 0:
        with torch.no_grad():
            texts = tokenizer(seen_classes).cuda()
            if isinstance(model, torch.nn.DataParallel):
                class_embeddings, _ = model.module.encode_text(texts)
            else:
                class_embeddings, _ = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            seen_embeddings = class_embeddings.cuda()
    
    if args.target_encoder == 'vision':
        with torch.no_grad():
            texts = [template.format(l) for l in class_names]
            texts = tokenizer(texts).cuda()
            if isinstance(model, torch.nn.DataParallel):
                class_embeddings, _ = model.module.encode_text(texts)
            else:
                class_embeddings, _ = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            cur_embeddings = class_embeddings.cuda()
        
        if len(seen_classes) > 0:
            embeddings = torch.cat([seen_embeddings, cur_embeddings], dim=0).cuda()
        else:
            embeddings = cur_embeddings.cuda()
        print(len(seen_classes), len(texts), embeddings.shape)

    correct, total = 0, 0
    for iteration in tqdm(range(total_iterations)):
        if iteration % num_batches == 0:
            data_iter = iter(train_loader)

        model.train()
        scheduler(iteration)

        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()
        labels += len(seen_classes)

        sparse_losses = 0

        # Text embeddings for contrastive loss when training text or both encoders
        if args.target_encoder == 'text' or args.target_encoder == 'all':
            texts = [template.format(l) for l in class_names]
            texts = tokenizer(texts).cuda()
            if isinstance(model, torch.nn.DataParallel):
                class_embeddings, _ = model.module.encode_text(texts)
            else:
                class_embeddings, _ = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            cur_embeddings = class_embeddings.cuda()

            if len(seen_classes) > 0:
                embeddings = torch.cat([seen_embeddings, cur_embeddings], dim=0).cuda()
            else:
                embeddings = cur_embeddings.cuda()

            embeddings = embeddings.cuda()

        out, _ = model.module.encode_image(images)
        out = out / out.norm(dim=-1, keepdim=True)

        logits_per_image = logit_scale.exp() * out @ embeddings.t()
    
        if len(seen_classes) > 0:
            logits_per_image[:, :len(seen_classes)] = float('-inf')
        
        ce_loss = F.cross_entropy(logits_per_image, labels.long(), label_smoothing=args.ls)
        
        anchor_labels = labels.contiguous().view(-1, 1)
        contrast_labels = torch.arange(embeddings.shape[0]).view(-1,1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda().t()
        t_logits = logit_scale.exp() * embeddings @ out.t()
        loss_ce_t = F.cross_entropy(t_logits, mask)

        if args.target_encoder == 'vision' or args.target_encoder == 'all':
            loss = ce_loss + loss_ce_t
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        _, preds = torch.max(logits_per_image, dim=1)
        correct += preds.eq(labels.expand_as(preds)).cpu().sum()
        total += len(labels)
        
        writer.add_scalar(f'{dataset_name}/train', loss.item(), iteration)
        writer.add_scalar(f'{dataset_name}/CE_i', ce_loss.item(), iteration)
        writer.add_scalar(f'{dataset_name}/CE_t', loss_ce_t.item(), iteration)
        writer.add_scalar(f'{dataset_name}/LR', optimizer.param_groups[0]['lr'], iteration)

        if iteration % loss_interval == 0:
            logging.info(
                f"Loss: {loss.item()}, ce_loss: {ce_loss.item()}, ce_t_loss: {loss_ce_t.item()}, sparse: {sparse_losses}"
            )
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            logging.info(f"Train accuracy: {train_acc}%")
            correct, total = 0, 0
