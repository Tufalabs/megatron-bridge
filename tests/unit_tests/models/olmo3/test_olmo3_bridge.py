"""OLMo 3 registration, config, mapping, and normalization regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Olmo3Config

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping
from megatron.bridge.models.olmo3 import Olmo3Bridge, Olmo3ModelBuilder, Olmo3ModelConfig, Olmo3ModelProvider
from megatron.bridge.models.olmo3.modeling_olmo3 import Olmo3FullDimRMSNorm, Olmo3RotaryEmbedding


pytestmark = pytest.mark.unit


@pytest.fixture
def config():
    return Olmo3Config(
        architectures=["Olmo3ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=65536,
        sliding_window=16,
        rms_norm_eps=1e-6,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_theta=500000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "attention_factor": 1.2079441541679836,
        },
        tie_word_embeddings=False,
    )


def test_autobridge_and_config_roundtrip(config):
    provider = AutoBridge.from_hf_config(config).get_model_config()
    assert isinstance(provider, Olmo3ModelConfig)
    assert not provider.qk_layernorm  # Sharded full-width norms manage their own TP reductions.
    assert provider.gated_linear_unit
    assert provider.normalization == "RMSNorm"
    assert not provider.share_embeddings_and_output_weights
    assert provider.kv_channels == 32
    assert provider.rotary_base == 500000
    exported = Olmo3Bridge.megatron_to_hf_config(provider)
    restored = Olmo3Config(**exported)
    assert restored.layer_types == config.layer_types
    assert restored.sliding_window == config.sliding_window
    assert restored.model_type == "olmo3"
    assert exported["rope_scaling"]["attention_factor"] == 1.2079441541679836
    assert exported["rope_scaling"]["original_max_position_embeddings"] == 8192


def test_mapping_covers_every_hf_parameter(config):
    from transformers import Olmo3ForCausalLM

    with torch.device("meta"):
        model = Olmo3ForCausalLM(config)
    registry = Olmo3Bridge().mapping_registry()
    for name in model.state_dict():
        assert registry.hf_to_megatron_lookup(name) is not None, name
    for name in ("q", "k"):
        mapping = registry.megatron_to_hf_lookup(f"decoder.layers.0.self_attention.{name}_layernorm.weight")
        assert isinstance(mapping, ColumnParallelMapping)
    assert registry.megatron_to_hf_lookup("decoder.layers.0.input_layernorm.weight") is None


@pytest.mark.parametrize(
    "override",
    [{"hidden_act": "relu"}, {"attention_bias": True}, {"sliding_window": 0}, {"layer_types": ["full_attention"]}],
)
def test_rejects_incompatible_config(config, override):
    for name, value in override.items():
        assert hasattr(config, name)
        setattr(config, name, value)
    with pytest.raises(ValueError):
        Olmo3Bridge().hf_config_to_model_config(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_width_norm_forward_and_backward(dtype):
    config = SimpleNamespace(params_dtype=dtype, tensor_model_parallel_size=1)
    norm = Olmo3FullDimRMSNorm(32, config, 1e-6)
    torch.manual_seed(79)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 32, dtype=dtype))
    reference_weight = norm.weight.detach().clone().requires_grad_()
    x = torch.randn(5, 2, 4, 8, dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    expected = reference.float().reshape(5, 2, 32)
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
    expected = (reference_weight * expected).to(dtype).reshape_as(reference)
    actual = norm(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    upstream = torch.randn_like(x)
    actual.backward(upstream)
    expected.backward(upstream)
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(x.grad, reference.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad, rtol=tolerance, atol=tolerance)


def test_local_rope_is_unscaled_even_when_global_uses_yarn(config):
    provider = AutoBridge.from_hf_config(config).get_model_config()
    provider.use_cpu_initialization = True
    # MCore eagerly populates GPU cos/sin caches even with CPU initialization.
    # Cache contents are covered by the GPU parity test; only frequencies are needed here.
    with patch(
        "megatron.core.models.common.embeddings.yarn_rotary_pos_embedding.YarnRotaryEmbedding._set_cos_sin_cache"
    ):
        rope = Olmo3RotaryEmbedding(provider)
    inv = 1.0 / 500000 ** (torch.arange(0, 32, 2).float() / 32)
    torch.testing.assert_close(rope.inv_freq, inv)
    assert rope.full_rope.scaling_factor == 8.0
    torch.testing.assert_close(rope.full_rope.inv_freq_inter, inv / 8.0)


def test_layerwise_rope_config_preserves_yarn_and_roundtrip(config):
    assert hasattr(config, "rope_parameters")
    config.rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "attention_factor": 1.2079441541679836,
        },
    }
    provider = AutoBridge.from_hf_config(config).get_model_config()
    assert provider.olmo3_layerwise_rope_parameters
    assert provider.olmo3_rope_parameters["rope_type"] == "yarn"
    assert provider.olmo3_rope_parameters["factor"] == 8.0
    exported = Olmo3Bridge.megatron_to_hf_config(provider)
    assert exported["rope_parameters"] == config.rope_parameters


@pytest.mark.parametrize(
    "local_rope",
    [
        {"rope_type": "yarn", "rope_theta": 500000, "factor": 8.0},
        {"rope_type": "default", "rope_theta": 1000000},
    ],
)
def test_rejects_unsupported_layerwise_rope(config, local_rope):
    assert hasattr(config, "rope_parameters")
    config.rope_parameters = {
        "sliding_attention": local_rope,
        "full_attention": {"rope_type": "default", "rope_theta": 500000},
    }
    with pytest.raises(ValueError):
        Olmo3Bridge().hf_config_to_model_config(config)


def test_builder_mapping_does_not_use_legacy_provider(config):
    bridge = Olmo3Bridge()
    assert "provider_bridge" not in Olmo3Bridge.__dict__
    assert Olmo3Bridge.USE_MODEL_CONFIG_FOR_CONVERSION
    with (
        patch.object(bridge, "provider_bridge", side_effect=AssertionError("legacy provider used")),
        patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("legacy kwargs used")),
    ):
        model_config = bridge.hf_config_to_model_config(config)
    assert model_config.get_builder_cls() is Olmo3ModelBuilder
    assert model_config.hidden_size == config.hidden_size
    assert model_config.num_layers == config.num_hidden_layers
    assert model_config.num_query_groups == config.num_key_value_heads
    assert model_config.olmo3_layer_types == tuple(config.layer_types)


def test_builder_config_serialization_preserves_architecture(config):
    model_config = AutoBridge.from_hf_config(config).get_model_config()
    restored = Olmo3ModelConfig.from_dict(model_config.as_dict())
    assert isinstance(restored, Olmo3ModelConfig)
    assert restored.get_builder_cls() is Olmo3ModelBuilder
    assert restored.transformer_layer_spec is model_config.transformer_layer_spec
    assert tuple(restored.olmo3_layer_types) == model_config.olmo3_layer_types
    assert restored.olmo3_rope_parameters == model_config.olmo3_rope_parameters
    assert restored.olmo3_sliding_window == model_config.olmo3_sliding_window
    assert restored.activation_func is model_config.activation_func
    with pytest.raises(AttributeError):
        restored.unknown_olmo3_setting = True


def test_legacy_provider_uses_canonical_architecture(config):
    auto = AutoBridge.from_hf_config(config)
    model_config = auto.get_model_config()
    with pytest.warns(FutureWarning):
        provider = auto.to_megatron_provider(load_weights=False)
    assert isinstance(provider, Olmo3ModelProvider)
    for name in (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "rotary_base",
        "normalization",
        "qk_layernorm",
        "gated_linear_unit",
        "add_bias_linear",
        "add_qkv_bias",
        "layernorm_epsilon",
        "olmo3_layer_types",
        "olmo3_sliding_window",
        "olmo3_rope_parameters",
        "olmo3_layerwise_rope_parameters",
    ):
        assert getattr(provider, name) == getattr(model_config, name), name


@pytest.mark.parametrize(
    "overrides",
    [
        {"qk_layernorm": True},
        {"position_embedding_type": "yarn"},
        {"apply_rope_fusion": True},
        {"flash_decode": True},
        {"num_query_groups": 1, "tensor_model_parallel_size": 2},
    ],
)
def test_builder_rejects_incompatible_runtime_overrides(config, overrides):
    model_config = AutoBridge.from_hf_config(config).get_model_config()
    for name, value in overrides.items():
        assert hasattr(model_config, name)
        setattr(model_config, name, value)
    with pytest.raises(ValueError):
        Olmo3ModelBuilder(model_config).build_model(SimpleNamespace())
