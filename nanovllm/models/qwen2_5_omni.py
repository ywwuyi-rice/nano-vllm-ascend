# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


"""
Simplified implementation of the Qwen2.5-Omni multimodal model.
This module implements the thinker stage for multimodal understanding.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import NPURMSNorm as RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rotary_emb_mrope
from nanovllm.utils.logger import init_logger
from nanovllm.utils.parallel_state import get_tp_group

logger = init_logger(__name__)


def get_llm_pos_ids_for_vision(
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: torch.Tensor,
        grid_ws: torch.Tensor,
) -> torch.Tensor:
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index_tensor = (
        torch.tensor(t_index, dtype=torch.long).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    )
    _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> List[List[int]]:
    if lst.numel() == 0:
        return []

    data_list = lst.detach().cpu().tolist()
    max_val = int(torch.max(lst).item())

    ranges: List[List[int]] = [[] for _ in range((max_val // interval) + 1)]

    for num in data_list:
        index = int(num // interval)
        ranges[index].append(num)

    return ranges


class Qwen2_5OmniTextAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position: int,
            rms_norm_eps: float,
            qkv_bias: bool,
            head_dim: int | None,
            rope_theta: float,
            rope_scaling: tuple | None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rotary_emb_mrope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen2_5OmniTextMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2_5OmniTextDecoderLayer(nn.Module):

    def __init__(
            self,
            config,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = None

        self.self_attn = Qwen2_5OmniTextAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=rope_scaling,
        )
        self.mlp = Qwen2_5OmniTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2_5OmniTextModel(nn.Module):

    def __init__(
            self,
            config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2_5OmniTextDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            positions: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2_5OmniTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen2_5OmniTextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            positions: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Qwen2_5OmniForConditionalGeneration(nn.Module):
    """
    Qwen2.5-Omni model for conditional generation.
    This implements the thinker stage with support for audio and vision inputs.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.thinker_config = getattr(config, "thinker_config", config)

        self.text_config = getattr(self.thinker_config, "text_config", self.thinker_config)
        self.vision_config = getattr(self.thinker_config, "vision_config", None)

        self.language_model = Qwen2_5OmniTextForCausalLM(self.text_config)

        self.packed_modules_mapping = {
            "mlp.gate_proj": ("mlp.gate_up_proj", 0),
            "mlp.up_proj": ("mlp.gate_up_proj", 1),
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
        }

        logger.info("[Qwen2_5OmniForConditionalGeneration] Initialization complete")
        logger.info(f"  - Language model: {type(self.language_model).__name__}")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(input_ids)

    def get_mrope_input_positions(
            self,
            input_tokens: List[int],
            grid_thw: List[List[int]] | torch.Tensor,
            audio_feature_lengths: List[int] | None = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Get MRoPE input positions for audio and vision modalities.

        Args:
            input_tokens: Input token IDs
            grid_thw: Grid dimensions (temporal, height, width)
            audio_feature_lengths: Lengths of audio features

        Returns:
            Tuple of (positions, delta)
        """
        thinker_config = self.thinker_config
        audio_token_id = getattr(thinker_config, "audio_token_index", -1)
        image_token_id = getattr(thinker_config, "image_token_index", -1)
        video_token_id = getattr(thinker_config, "video_token_index", -1)
        audio_start_token_id = getattr(thinker_config, "audio_start_token_id", -1)
        audio_end_token_id = getattr(thinker_config, "audio_end_token_id", -1)
        vision_start_token_id = getattr(thinker_config, "vision_start_token_id", -1)
        vision_end_token_id = getattr(thinker_config, "vision_end_token_id", -1)
        spatial_merge_size = getattr(thinker_config.vision_config, "spatial_merge_size", 2)
        tokens_per_second = getattr(thinker_config.vision_config, "tokens_per_second", 25)

        if isinstance(grid_thw, list):
            grid_thw = torch.tensor(grid_thw)

        src_item = input_tokens
        audio_seqlens = audio_feature_lengths
        audio_idx = 0
        video_idx = 0
        image_idx = 0
        new_src_item: List[int] = []
        llm_pos_ids_list: List[torch.Tensor] = []

        idx = 0
        while idx < len(src_item):
            new_src_item_len = len(new_src_item)
            start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            if src_item[idx] not in [audio_token_id, video_token_id, image_token_id]:
                new_src_item.append(src_item[idx])
                llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
                llm_pos_ids_list.append(llm_pos_ids)
            elif src_item[idx] == audio_token_id:
                if audio_seqlens is not None:
                    audio_seqlen = audio_seqlens[audio_idx]
                    place_num = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
                    new_src_item.extend([audio_token_id] * place_num)
                    llm_pos_ids = torch.arange(place_num, dtype=torch.long).expand(3, -1) + start_idx
                    llm_pos_ids_list.append(llm_pos_ids)
                    audio_idx += 1
                else:
                    new_src_item.append(src_item[idx])
                    llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
                    llm_pos_ids_list.append(llm_pos_ids)
            elif src_item[idx] == image_token_id:
                grid_t = grid_thw[image_idx][0]
                grid_hs = grid_thw[:, 1]
                grid_ws = grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    start_idx, image_idx, spatial_merge_size, t_index.tolist(), grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = grid_thw[image_idx].prod() // (spatial_merge_size ** 2)
                new_src_item.extend([image_token_id] * vision_seqlen)
                image_idx += 1
            elif src_item[idx] == video_token_id:
                grid_t = grid_thw[video_idx][0]
                grid_hs = grid_thw[:, 1]
                grid_ws = grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    start_idx, video_idx, spatial_merge_size, t_index.tolist(), grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = grid_thw[video_idx].prod() // (spatial_merge_size ** 2)
                new_src_item.extend([video_token_id] * vision_seqlen)
                video_idx += 1
            idx += len(new_src_item) - new_src_item_len

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        mrope_position_delta = torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item)

        return llm_positions, mrope_position_delta

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            positions: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            sequence_lengths: list[int] | None = None,
            vision_slices_per_seq: list[list[dict]] | None = None,
            **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids and inputs_embeds cannot be None simultaneously")
            inputs_embeds = self.get_input_embeddings(input_ids)

        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states):
        """Compute logits (delegate to language model)"""
        return self.language_model.compute_logits(hidden_states)


def load_qwen2_5_omni_model(model_path, config):
    """
    Load Qwen2.5-Omni model

    Args:
        model_path: Model path
        config: Configuration object

    Returns:
        model: Qwen2_5OmniForConditionalGeneration instance
    """
    hf_config = config.hf_config

    model = Qwen2_5OmniForConditionalGeneration(hf_config)

    from nanovllm.utils.loader import load_model

    logger.info("[load_qwen2_5_omni_model] Loading Qwen2.5-Omni weights...")

    def name_mapping(weight_name: str) -> str | None:
        if weight_name.startswith("language_model."):
            sub_name = weight_name[len("language_model."):]
            text_model_prefixes = (
                "model.",
                "embed_tokens.",
                "layers.",
                "norm.",
                "rotary_emb.",
            )
            if sub_name.startswith(text_model_prefixes):
                if sub_name.startswith("model."):
                    sub_name = sub_name[len("model."):]
                sub_name = "language_model.model." + sub_name
            elif sub_name.startswith("lm_head."):
                sub_name = "language_model.lm_head." + sub_name[len("lm_head."):]
            else:
                sub_name = "language_model." + sub_name
            return sub_name
        if weight_name.startswith("visual.") and not weight_name.startswith("visual.audio_tower"):
            sub_name = weight_name[len("visual."):]
            return "visual.vision." + sub_name
        return None

    load_model(model, model_path, name_mapping=name_mapping)
    return model
