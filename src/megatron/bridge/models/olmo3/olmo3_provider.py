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

"""Legacy OLMo 3 provider adapter for callers such as verl."""

from dataclasses import dataclass, field
from typing import Callable

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding, olmo3_layer_spec


@dataclass
class Olmo3ModelProvider(GPTModelProvider):
    """Build OLMo 3 with post-norm residual blocks and layer-specific RoPE.

    Architecture dimensions are populated by AutoBridge from the HF config.
    The builder-backed API is canonical. This adapter retains mixed rotary
    construction for existing provider callers without Megatron-Core changes.
    """

    olmo3_layer_types: tuple[str, ...] = ()
    olmo3_sliding_window: int | None = None
    olmo3_rope_parameters: dict = field(default_factory=dict)
    olmo3_layerwise_rope_parameters: bool = False
    transformer_layer_spec: ModuleSpec | Callable = olmo3_layer_spec

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Instantiate GPTModel and install local/default plus global/YaRN RoPE."""
        if len(self.olmo3_layer_types) != self.num_layers:
            raise ValueError("OLMo 3 requires exactly one attention type per layer")
        if self.num_query_groups % self.tensor_model_parallel_size:
            raise ValueError("OLMo 3 currently requires KV heads divisible by TP size")
        if self.flash_decode or self.fused_single_qkv_rope:
            raise ValueError("OLMo 3 does not support fused inference RoPE")
        # GPTModel consumes a stack of angles through its standard rope path.
        # Each attention layer selects its own angles and concentration factor.
        self.qk_layernorm = False
        self.position_embedding_type = "rope"
        self.apply_rope_fusion = False
        self.yarn_rotary_scaling_factor = None
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        model.rotary_pos_emb = Olmo3RotaryEmbedding(self)
        return model
