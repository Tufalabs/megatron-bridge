# OLMo 3

This integration retains the provider API used by verl. Its original GPU
validation used Bridge 0.5.2 and Megatron-Core 0.18.0. It adds
`Olmo3ForCausalLM` / `model_type=olmo3` to AutoBridge. The initial checkpoint is
[allenai/Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT).
No Megatron-Core source modifications or new dependencies are needed.

## Architecture and conversion

The implementation preserves OLMo 3's bias-free SiLU MLP, untied output
weights, full-projection Q/K RMSNorm, and post-attention/post-MLP RMSNorm
before residual addition. Sliding layers use default RoPE; full layers use
checkpoint-configured YaRN, including its attention concentration factor.
Both legacy flat RoPE settings and newer Transformers per-layer settings are
preserved on import/export. Sliding layers require unscaled RoPE and the same
base frequency as full layers. Post norms explicitly use the checkpoint's
RMSNorm epsilon. Rotation arithmetic stays in FP32 and is cast back before TE
attention.
Full-width Q/K normalization also applies learned weights in FP32 and rounds
only once, matching the [original HF implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/olmo3/modeling_olmo3.py).

Q/K norm weights are TP-sharded. Their statistics use an autograd-aware TP
sum, so both forward values and cross-rank denominator gradients are correct.
The provider's standard `qk_layernorm` flag is intentionally false: Megatron
uses that flag to all-reduce replicated per-head norm gradients. The custom
attention enables its full-width norms on a private per-layer config.

HF import/export maps every learned tensor, including fused QKV and gated
MLP projections. Native distributed checkpoints describe Q/K norm shards.

```python
import torch

from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "allenai/Olmo-3-7B-Think-SFT", trust_remote_code=False,
)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.params_dtype = torch.bfloat16
provider.bf16 = True
provider.finalize()
# Run under torch.distributed.run (or let the caller initialize distributed).
provider.initialize_model_parallel(seed=1234)
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

## Use this checkout in an existing image

Bind-mount the checkout at `/opt/Megatron-Bridge`, then put its `src` directory
first on `PYTHONPATH` in **every actor/worker process on every node**:

```bash
export PYTHONPATH="/opt/Megatron-Bridge/src${PYTHONPATH:+:$PYTHONPATH}"
```

For Slurm/Pyxis, add `--container-mounts="$BRIDGE_CHECKOUT:/opt/Megatron-Bridge"`
to the existing launch options. Preserve other mounts. `BRIDGE_CHECKOUT`
should be an absolute shared-filesystem path visible from all compute nodes.
For Ray, propagate the mount and `PYTHONPATH` through the worker environment
and restart the worker processes before using the fork. Check the resolved
`megatron.bridge.__file__` in a worker; it should be under the mounted checkout.

With the image's virtual environment activated, replace `uv run` in the
validation commands below with `uv run --active --no-sync` to reuse that
environment without resolving or installing the checkout's dependencies.

This source overlay uses the image's existing PyTorch, TE, and Megatron-Core;
it does not rebuild the image or replace its dependencies. No verl
`external_lib` registration is necessary because the fork registers OLMo 3
when `megatron.bridge.models` is imported. Use verl's native Megatron Bridge
path (`use_mbridge=True`, `vanilla_mbridge=False` in its Megatron engine config).

## Validation

```bash
uv run python -m pytest tests/unit_tests/models/olmo3/test_olmo3_bridge.py -q

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  tests/functional_tests/test_groups/models/olmo3/validate_olmo3.py \
  --work-dir /tmp/olmo3-tp1 --tp 1

uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/functional_tests/test_groups/models/olmo3/validate_olmo3.py \
  --work-dir /tmp/olmo3-tp2 --tp 2 --sequence-parallel

# Fresh process: reload the native checkpoint and repeat weight/logit checks.
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/functional_tests/test_groups/models/olmo3/validate_olmo3.py \
  --work-dir /tmp/olmo3-tp2 --tp 2 --sequence-parallel --reload-checkpoint
```

The functional test creates a four-layer model with non-unit norm weights, verifies exact tensor
round-tripping, checks ordinary and packed logits against HF blocks with
explicit checkpoint-native rotary embeddings, checks every parameter's
gradient, and runs three optimizer steps. A short window exercises attention
masking even with small test sequences. Pass `--hf-model-path` with a local
checkpoint to test the real model's weights and logits instead.

Transformers 5.9.0's ordinary OLMo 3 forward applies YaRN to sliding layers
as well. The test constructs explicit local/global embeddings instead of
using that incorrect reference path; it does not monkeypatch Transformers.
See [the upstream RoPE fix](https://github.com/huggingface/transformers/pull/46911).

## Measured checks (2026-10-22)

Validated on GB300 with PyTorch 2.11.0+cu130, TE 2.16.1, Megatron-Core
0.18.0, Bridge 0.5.2, and Transformers 5.9.0. Nine focused unit tests and
Ruff checks pass. GPU checks use at most two GPUs.

| Case | Exact HF tensor round-trip | Plain / packed logit cosine | Minimum parameter-gradient cosine | Three SGD losses |
|---|---:|---|---:|---|
| Four-layer toy, learned norm weights, epsilon 1e-6, TP=1 | 47 | 1.0 / 1.0 | 0.999903 | 5.5549, 5.4504, 5.3872 |
| Same toy, TP=2 with sequence parallelism | 47 | 0.999846 / 0.999853 | 0.998873 | 5.5543, 5.4511, 5.3870 |
| Real 7B checkpoint, TP=1 | 355 | 0.999886 / 0.999931 | Not measured | Not measured |

A fresh process reloads the toy's TP=2 native distributed checkpoint and
repeats exact tensor and ordinary/packed logit checks successfully. All
listed forward checks match the reference's last-token prediction.

For the real 7B model, swapping **only the HF reference's post/final norm
kernels** to TE, with the same weights and epsilon, makes the tested plain
forward logits exactly identical (maximum error 0). This isolates the
remaining native-HF BF16 discrepancy to norm-kernel arithmetic; the
production model retains TE norms. With native HF norms, the measured plain
maximum/mean logit errors are 0.25/0.031958, and packed errors are
0.216797/0.026632. Use `--diagnose-norm-rounding` with TP=1 to repeat the
component comparison and controlled reference swap.

Supported configurations currently require KV head count divisible by TP
size, bias-free projections, SiLU, and default or YaRN RoPE. Fused RoPE and
flash decode are disabled. FP8/FP4, pipeline/context parallel execution,
and optimizer-state resume need separate validation before being considered
covered by these checks.

On the original Bridge 0.5.2 base, the real 7B checkpoint also completed 20
verl GRPO updates with SGLang on eight GB300 GPUs (TP1, 128 prompts with eight
responses each). All metrics were finite, gradients were nonzero, and response
aborts and trajectory staleness were zero. Minimum actor/rollout correlation
was 0.999718 and maximum absolute KL was 0.0004093. These measurements describe
the original base; GPU parity and training have not been rerun on this fork's
newer base.

## Fork port validation (2026-10-23)

All 12 focused CPU tests and the full repository pre-commit hooks pass with
PyTorch 2.14.1+cu130, Transformers 5.19.0, and the fork's pinned Megatron-Core
revision `ede2c586cf70dd452f93b9726b31cbedc8df0dd0`. The new cases cover nested
local/full RoPE import/export and rejection of incompatible local settings.

This family retains the provider construction path used by verl; the newer
builder-backed `get_model_config()` API is not implemented for OLMo 3 here.
GPU parity and training on the newer fork base remain unverified.
