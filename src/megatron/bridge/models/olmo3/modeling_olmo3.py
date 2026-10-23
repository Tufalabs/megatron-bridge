# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OLMo 3 architecture extensions; all attention kernels remain in Megatron/TE."""

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.distributed as dist
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer import ModuleSpec, TransformerConfig, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch.distributed.nn.functional import all_reduce

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig as BridgeTransformerConfig
from megatron.bridge.utils.import_utils import safe_import_from


TEColumnParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEColumnParallelLinear")
TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")
TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")
TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")


if TYPE_CHECKING:
    from megatron.bridge.models.olmo3.olmo3_provider import Olmo3ModelProvider


def olmo3_layer_spec(config: "Olmo3ModelProvider | Olmo3ModelConfig") -> ModuleSpec:
    """Create OLMo 3's post-norm attention/MLP blocks, without any prenorm."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=Olmo3DotProductAttention,
                    q_layernorm=Olmo3QNorm,
                    k_layernorm=Olmo3KNorm,
                    linear_proj=Olmo3PostNormLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=Olmo3PostNormLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


class Olmo3PostNormLinear(TERowParallelLinear):
    """Row-parallel projection followed by RMSNorm using the checkpoint's epsilon."""

    def __init__(self, input_size: int, output_size: int, *, config: TransformerConfig, **kwargs: object) -> None:
        super().__init__(input_size, output_size, config=config, **kwargs)
        # TENorm's default epsilon is 1e-5; OLMo 3 checkpoints can require 1e-6.
        self.post_layernorm = TENorm(config, output_size, eps=config.layernorm_epsilon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the output norm before the block's residual addition."""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


class Olmo3FullDimRMSNorm(torch.nn.Module):
    """Full projected-width Q/K RMSNorm with sharded weights and global statistics.

    TENorm has no cross-TP statistic reduction with sharded affine weights.
    A local norm or a non-differentiable reduction loses cross-rank denominator
    gradients. The autograd-aware
    reduction is needed in BOTH directions: a rank's Q/K gradient depends on
    the denominator contributions from every other TP rank.
    """

    def __init__(self, global_dim: int, config: TransformerConfig, eps: float) -> None:
        super().__init__()
        self.global_dim = global_dim
        groups = getattr(config, "_pg_collection", None)
        self.tp_group = groups.tp if groups is not None else get_tensor_model_parallel_group(check_initialized=False)
        self.tp_size = config.tensor_model_parallel_size
        if global_dim % self.tp_size:
            raise ValueError("Q/K normalization width must be divisible by TP size")
        self.eps = eps
        device = torch.get_default_device()
        if device.type != "meta":
            device = "cpu" if getattr(config, "use_cpu_initialization", True) else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(
            torch.ones(global_dim // self.tp_size, dtype=config.params_dtype, device=device)
        )
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize all local heads using the full projection's RMS."""
        shape, dtype = x.shape, x.dtype
        x = x.reshape(*shape[:2], -1).float()
        squares = x.square().sum(-1, keepdim=True)
        if self.tp_size > 1:
            if self.tp_group is None:
                raise RuntimeError("Tensor parallel process group must be initialized")
            squares = all_reduce(squares, op=dist.ReduceOp.SUM, group=self.tp_group)
        normalized = x * torch.rsqrt(squares / self.global_dim + self.eps)
        # OLMo 3 applies the learned weight in FP32, then casts once. Casting
        # before the multiply silently changes trained (non-unit) Q/K norms.
        return (normalized * self.weight).to(dtype).reshape(shape)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> dict:
        """Describe norm-weight shards for native distributed checkpoints."""
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"] if metadata else None,
        )


class Olmo3QNorm:
    """Construct a full-width query norm from Megatron's per-head norm API."""

    def __new__(cls, hidden_size: int, config: TransformerConfig, eps: float = 1e-6) -> Olmo3FullDimRMSNorm:
        return Olmo3FullDimRMSNorm(hidden_size * config.num_attention_heads, config, eps)


class Olmo3KNorm:
    """Construct a full-width key norm from Megatron's per-head norm API."""

    def __new__(cls, hidden_size: int, config: TransformerConfig, eps: float = 1e-6) -> Olmo3FullDimRMSNorm:
        return Olmo3FullDimRMSNorm(hidden_size * config.num_query_groups, config, eps)


class Olmo3RotaryEmbedding(RotaryEmbedding):
    """Generate default angles for local layers and optional YaRN for full layers."""

    def __init__(self, config: "Olmo3ModelProvider | Olmo3ModelConfig") -> None:
        kwargs = dict(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_base=config.rotary_base,
            use_cpu_initialization=config.use_cpu_initialization,
        )
        super().__init__(**kwargs)
        rope = config.olmo3_rope_parameters
        if rope.get("rope_type", "default") == "yarn":
            self.full_rope = YarnRotaryEmbedding(
                **kwargs,
                scaling_factor=rope["factor"],
                original_max_position_embeddings=rope["original_max_position_embeddings"],
                beta_fast=rope.get("beta_fast", 32.0),
                beta_slow=rope.get("beta_slow", 1.0),
                correction_range_round_to_int=rope.get("truncate", True),
            )
        else:
            self.full_rope = None

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Return stacked local/global angles, retaining Megatron's packing and CP handling."""
        local = super().forward(max_seq_len, offset, packed_seq, cp_group)
        full = self.full_rope(max_seq_len, offset, packed_seq, cp_group)[0] if self.full_rope else local
        return torch.stack((local, full))


class Olmo3SelfAttention(SelfAttention):
    """Select RoPE and sliding-window settings independently for every layer."""

    def __init__(
        self,
        config: "Olmo3ModelProvider | Olmo3ModelConfig",
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        **kwargs: object,
    ) -> None:
        config = copy.copy(config)
        config.qk_layernorm = True
        self.is_sliding = config.olmo3_layer_types[layer_number - 1] == "sliding_attention"
        config.window_size = (config.olmo3_sliding_window - 1, 0) if self.is_sliding else None
        config.window_attn_skip_freq = None
        rope = config.olmo3_rope_parameters
        config.yarn_rotary_scaling_factor = None
        if not self.is_sliding and rope.get("rope_type") == "yarn":
            factor = rope["factor"]
            attention_factor = rope.get("attention_factor", 1.0 + 0.1 * math.log(factor))
            config.yarn_rotary_scaling_factor = factor
            config.yarn_mscale = (attention_factor - 1.0) / (0.1 * math.log(factor)) if factor > 1 else 1.0
            config.yarn_mscale_all_dim = 0.0
        super().__init__(config, submodules, layer_number, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Choose the correct angles before executing Megatron self-attention."""
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb[0 if self.is_sliding else 1]
        return super().forward(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep rotation arithmetic in FP32 as in the original OLMo 3 checkpoint."""
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states, **kwargs)
        # Megatron computes cos/sin in the input dtype. FP32 Q/K preserve HF's
        # FP32 rotary multiplication; cast once after rotating, in core_attention.
        return query.float(), key.float(), value


class Olmo3DotProductAttention(TEDotProductAttention):
    """Restore Q/K activation dtype after FP32 rotary, before TE attention."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run TE attention with matched Q/K/V dtypes."""
        return super().forward(query.to(value.dtype), key.to(value.dtype), value, *args, **kwargs)


@dataclass
class Olmo3TransformerConfig(BridgeTransformerConfig):
    """Declare the per-layer rotary and window fields missing from stock GPT."""

    olmo3_layer_types: tuple[str, ...] = ()
    olmo3_sliding_window: int | None = None
    olmo3_rope_parameters: dict = field(default_factory=dict)
    olmo3_layerwise_rope_parameters: bool = False


@dataclass(kw_only=True)
class Olmo3ModelConfig(BridgeGPTModelConfig):
    """Serializable GPT config for OLMo 3's mixed rotary construction."""

    builder: ClassVar[str] = "megatron.bridge.models.olmo3.modeling_olmo3.Olmo3ModelBuilder"
    transformer_config_class: ClassVar[type[BridgeTransformerConfig]] = Olmo3TransformerConfig


class Olmo3ModelBuilder(GPTModelBuilder):
    """Reuse the stock GPT builder and install per-layer rotary embeddings."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one GPT stage with OLMo 3's local/default and global/YaRN RoPE."""
        config = self._model_config
        if not isinstance(config, Olmo3ModelConfig):
            raise TypeError(f"Expected Olmo3ModelConfig, got {type(config).__name__}.")
        if len(config.olmo3_layer_types) != config.num_layers:
            raise ValueError("OLMo 3 requires exactly one attention type per layer")
        if config.num_query_groups % config.tensor_model_parallel_size:
            raise ValueError("OLMo 3 currently requires KV heads divisible by TP size")
        if config.flash_decode or config.fused_single_qkv_rope or config.apply_rope_fusion:
            raise ValueError("OLMo 3 does not support fused inference RoPE")
        if config.qk_layernorm:
            raise ValueError(
                "OLMo 3 full-width sharded Q/K norms require the global qk_layernorm flag to stay disabled"
            )
        if config.position_embedding_type != "rope" or config.yarn_rotary_scaling_factor is not None:
            raise ValueError("OLMo 3 rotary scaling is configured per layer, not by the global GPT rotary settings")
        model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        model.rotary_pos_emb = Olmo3RotaryEmbedding(config)
        return model
