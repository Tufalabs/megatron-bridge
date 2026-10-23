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

"""Bidirectional Hugging Face / Megatron conversion for dense OLMo 3."""

import copy
from typing import Any

from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.olmo3.modeling_olmo3 import Olmo3ModelConfig, Olmo3PostNormLinear, olmo3_layer_spec
from megatron.bridge.models.olmo3.olmo3_provider import Olmo3ModelProvider


AutoMapping.register_module_type(Olmo3PostNormLinear.__name__, "row")


@MegatronModelBridge.register_bridge(
    source="Olmo3ForCausalLM",
    target=GPTModel,
    provider=Olmo3ModelProvider,
    model_type="olmo3",
)
class Olmo3Bridge(MegatronModelBridge):
    """Convert OLMo 3 configs and all dense checkpoint parameters."""

    MODEL_CONFIG_CLASS = Olmo3ModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, config: PretrainedConfig) -> dict[str, Any]:
        """Translate architecture and preserve the checkpoint's mixed RoPE semantics."""
        kwargs = super().hf_config_to_model_config_kwargs(config)
        layer_types = tuple(config.layer_types)
        if len(layer_types) != config.num_hidden_layers or set(layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError("Unsupported OLMo 3 layer_types")
        if "sliding_attention" in layer_types and (config.sliding_window is None or config.sliding_window <= 0):
            raise ValueError("Sliding attention requires a positive sliding_window")
        if config.hidden_act != "silu" or getattr(config, "attention_bias", False):
            raise ValueError("Only bias-free SiLU OLMo 3 is supported")
        rope = copy.deepcopy(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
        layerwise_rope = "full_attention" in rope or "sliding_attention" in rope
        if layerwise_rope:
            local_rope = rope.get("sliding_attention") or {}
            rope = rope.get("full_attention") or {}
            if local_rope.get("rope_type", local_rope.get("type", "default")) != "default":
                raise ValueError("OLMo 3 sliding attention requires unscaled default RoPE")
            if local_rope.get("rope_theta", kwargs["rotary_base"]) != rope.get("rope_theta", kwargs["rotary_base"]):
                raise ValueError("OLMo 3 currently requires matching local and full RoPE bases")
        rope["rope_type"] = rope.pop("type", rope.get("rope_type", "default"))
        if rope["rope_type"] not in {"default", "yarn"}:
            raise ValueError("Only default and YaRN OLMo 3 RoPE are supported")
        if rope["rope_type"] == "yarn":
            rope.setdefault("original_max_position_embeddings", config.max_position_embeddings)
            if rope.get("factor", 0) < 1:
                raise ValueError("YaRN factor must be at least one")
        kwargs.update(
            olmo3_rope_parameters=rope,
            olmo3_layer_types=layer_types,
            olmo3_sliding_window=config.sliding_window,
            olmo3_layerwise_rope_parameters=layerwise_rope,
            transformer_layer_spec=olmo3_layer_spec,
            position_embedding_type="rope",
            yarn_rotary_scaling_factor=None,
            kv_channels=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            normalization="RMSNorm",
            # Full-width Q/K weights are TP-sharded, unlike the replicated
            # per-head norms selected by Megatron's global qk_layernorm flag.
            qk_layernorm=False,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            hidden_dropout=0.0,
            attention_dropout=config.attention_dropout,
            apply_rope_fusion=False,
            autocast_dtype=kwargs["params_dtype"],
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
        )
        return kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Keep the deprecated verl provider adapter on the canonical mapping."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    @classmethod
    def megatron_to_hf_config(cls, provider: Olmo3ModelProvider | Olmo3ModelConfig) -> dict:
        """Restore layer types, sliding window, and the original RoPE settings on export."""
        config = super().megatron_to_hf_config(provider)
        rope = copy.deepcopy(provider.olmo3_rope_parameters)
        rope.pop("rope_theta", None)
        config.update(
            layer_types=list(provider.olmo3_layer_types),
            sliding_window=provider.olmo3_sliding_window,
            rope_theta=provider.rotary_base,
            rope_scaling=rope if rope.get("rope_type") != "default" else None,
            attention_bias=False,
            attention_dropout=provider.attention_dropout,
        )
        if provider.olmo3_layerwise_rope_parameters:
            config["rope_parameters"] = {
                "sliding_attention": {"rope_type": "default", "rope_theta": provider.rotary_base},
                "full_attention": {**rope, "rope_theta": provider.rotary_base},
            }
        return config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map post-norm blocks, fused projections, and tensor-sharded full-width norms."""
        ordinary = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": "model.layers.*.post_feedforward_layernorm.weight",
        }
        mappings = [AutoMapping(m, h) for m, h in ordinary.items()]
        for projection in ("q", "k"):
            mappings.append(
                ColumnParallelMapping(
                    f"decoder.layers.*.self_attention.{projection}_layernorm.weight",
                    f"model.layers.*.self_attn.{projection}_norm.weight",
                )
            )
        mappings.extend(
            [
                QKVMapping(
                    "decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    "decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)
