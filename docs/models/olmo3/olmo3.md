# OLMo 3

AutoBridge supports `Olmo3ForCausalLM` / `model_type=olmo3`, including
[allenai/Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT).
Architecture dimensions and the attention schedule come from the Hugging Face
configuration. No Megatron-Core source modifications or new dependencies are
needed.

## Architecture and conversion

The implementation preserves OLMo 3's bias-free SiLU MLP, untied output
weights, full-projection Q/K RMSNorm, and post-attention/post-MLP RMSNorm
before residual addition. Sliding layers use default RoPE; full layers use
checkpoint-configured YaRN, including its attention concentration factor.
Legacy flat RoPE settings and newer Transformers per-layer settings are
preserved on import/export. Sliding layers require unscaled RoPE and the same
base frequency as full layers. Rotary arithmetic uses FP32 before casting
back to the attention dtype.

Post-attention, post-MLP, and final norms use Transformer Engine with the
checkpoint's epsilon. Full-width Q/K normalization needs TP-wide statistics
and TP-sharded affine weights, which the stock TENorm adapter does not
provide. Its autograd-aware TP sum preserves both forward statistics and
cross-rank denominator gradients. Learned weights are applied in FP32 before
a single cast back to the input dtype. The global `qk_layernorm` flag remains
false because Megatron uses it to reduce replicated per-head norm gradients;
the custom attention enables full-width norms on its private layer config.

HF import/export maps every learned tensor, including fused QKV and gated
MLP projections. Native distributed checkpoints describe the Q/K norm shards.

## Model API

The canonical construction path uses a serializable `Olmo3ModelConfig` and
`Olmo3ModelBuilder`. The builder reuses the stock GPT builder and installs
OLMo 3's mixed local/full rotary embeddings. Both construction APIs share
one HF configuration mapping.

```python
import torch

from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "allenai/Olmo-3-7B-Think-SFT", trust_remote_code=False,
)
config = bridge.get_model_config()
config.tensor_model_parallel_size = 1
config.pipeline_model_parallel_size = 1
config.params_dtype = torch.bfloat16
config.bf16 = True
models = bridge.get_model(config, wrap_with_ddp=False)
```

The deprecated `bridge.to_megatron_provider()` adapter remains available for
existing callers such as verl. Architecture fields are populated from the HF
config rather than size-specific provider classes.

See the [model examples](../../../examples/models/olmo3/README.md) for shared
conversion CLI commands and a published-checkpoint inference example.

## Use this checkout in an existing image

Put the checkout's `src` directory first on `PYTHONPATH` in every worker:

```bash
export BRIDGE_CHECKOUT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$BRIDGE_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
```

For Slurm/Pyxis, mount the checkout at the same absolute path on every node
and propagate `PYTHONPATH` to worker processes. Check the resolved
`megatron.bridge.__file__` inside a worker to verify the selected source.
With an image's virtual environment activated, `uv run --active --no-sync`
reuses its installed dependencies without resolving a new environment.

## Validation

```bash
uv run python -m pytest tests/unit_tests/models/olmo3/test_olmo3_bridge.py -q

# Three collected cases: builder TP1, builder TP2+SP, and legacy provider TP1.
# Each uses at most two GPUs and reloads a native checkpoint in a fresh process.
uv run python -m pytest tests/functional_tests/test_groups/models/olmo3/ -v

# Optional real-checkpoint parity check; the caller chooses the output directory.
export HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think-SFT}"
export OLMO3_WORK_DIR="${OLMO3_WORK_DIR:-$(mktemp -d)}"
uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  tests/functional_tests/test_groups/models/olmo3/test_olmo3_conversion.py \
  --work-dir "$OLMO3_WORK_DIR" --hf-model-path "$HF_MODEL" --tp 1
```

The H100 and GB200 L2 launchers run the collected cases using two GPUs:
`tests/functional_tests/launch_scripts/{h100,gb200}/active/L2_Launch_models_olmo3.sh`.
The test module follows the `test_*.py` discovery convention and also provides
the optional manual command-line entry point. `--legacy-provider` selects
the compatibility adapter; `--reload-checkpoint` checks a saved native checkpoint.

The functional fixture is a four-layer model with non-unit norm weights and a
short sliding window. Tests require exact tensor round-tripping, compare
plain and packed logits, check every parameter's gradient, and run three SGD
steps with decreasing loss. Logit acceptance requires cosine similarity of
at least 0.99 and identical last-token predictions; absolute errors are
reported separately.

The reference evaluates HF blocks with explicit checkpoint-native local and
full rotary embeddings. This also works around the Transformers 5.9 regression
that applied YaRN to sliding layers; it does not monkeypatch Transformers.
Use `--diagnose-norm-rounding` with TP1 for component comparisons and a
controlled swap of reference post/final norms to Transformer Engine.

## Measured checks (2026-10-23)

All 20 focused CPU tests and three collected GPU cases passed on GB300 with
PyTorch 2.14.1+cu130, Transformers 5.19.0, and pinned Megatron-Core revision
`ede2c586cf70dd452f93b9726b31cbedc8df0dd0`. Ruff lint and formatting checks pass.

| Case | Exact HF tensors | Plain / packed logit cosine | Minimum parameter-gradient cosine | Three SGD losses |
|---|---:|---|---:|---|
| Builder, four-layer toy, TP1 | 47 | 1.0 / 1.0 | 0.999903 | 5.5549, 5.4504, 5.3872 |
| Builder, same toy, TP2 with sequence parallelism | 47 | 0.999846 / 0.999853 | 0.998873 | 5.5543, 5.4511, 5.3870 |
| Legacy provider, same toy, TP1 | 47 | 1.0 / 1.0 | 0.999903 | 5.5549, 5.4504, 5.3872 |
| Builder, real 7B checkpoint, TP1 | 355 | 0.999886 / 0.999931 | Not measured | Not measured |

All listed forward checks match the reference's last-token prediction.
Fresh-process native-checkpoint reloads repeat exact weight and plain/packed
logit checks for all three toy cases. Real-checkpoint plain maximum/mean
absolute logit errors are 0.25/0.031958; packed errors are 0.216797/0.026632.

## Supported scope

Supported configurations require KV head count divisible by TP size,
bias-free projections, SiLU, and default or YaRN RoPE. Fused RoPE and flash
decode are disabled. Mixed rotary settings are configured per layer rather
than through global GPT rotary overrides.

FP8/FP4, pipeline/context parallel execution, optimizer-state resume, and
end-to-end verl training on this builder-backed base have not been validated
by these checks.
