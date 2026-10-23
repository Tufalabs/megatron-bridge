"""Focused OLMo 3 conversion, forward, packing, gradient, and checkpoint validation.

Pytest collects the TP1 and TP2 cases and allocates their work directories.
For manual validation, launch with ``uv run python -m torch.distributed.run
--nproc_per_node=2 test_olmo3_conversion.py --work-dir <output-dir> --tp 2``.
Pass --hf-model-path for
an existing checkpoint; otherwise a tiny four-layer mixed-attention model is
created. --reload-checkpoint validates the saved DCP in a fresh process.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from transformers import Olmo3Config, Olmo3ForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf


logger = logging.getLogger(__name__)


def reference_logits(model, ids):
    """Evaluate HF blocks with checkpoint-native local/default and global/YaRN RoPE.

    Explicit embeddings avoid the Transformers 5.9 regression that applies
    YaRN to sliding layers too. No global HF monkeypatch is installed.
    """
    config = model.config
    hidden = model.model.embed_tokens(ids)
    size = ids.shape[1]
    position = torch.arange(size, device=ids.device)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    layerwise_rope = "full_attention" in rope
    if layerwise_rope:
        rope = rope["full_attention"]
    theta = rope_theta_from_hf(config)
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=ids.device).float() / head_dim)
    frequencies = {"sliding_attention": (inv, 1.0)}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type == "default":
        frequencies["full_attention"] = (inv, 1.0)
    else:
        rope_kwargs = {"layer_type": "full_attention"} if layerwise_rope else {}
        frequencies["full_attention"] = ROPE_INIT_FUNCTIONS[rope_type](config, device=ids.device, **rope_kwargs)
    embeddings = {}
    for name, (freq, scale) in frequencies.items():
        angles = position.float()[:, None] * freq.float()[None, :]
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(0)
        embeddings[name] = (angles.cos() * scale, angles.sin() * scale)
    distance = position[:, None] - position[None, :]
    for layer, kind in zip(model.model.layers, config.layer_types):
        allowed = distance >= 0
        if kind == "sliding_attention":
            allowed = allowed & (distance < config.sliding_window)
        mask = torch.zeros(size, size, device=ids.device, dtype=hidden.dtype).masked_fill(~allowed, -torch.inf)
        hidden = layer(hidden, attention_mask=mask[None, None], position_embeddings=embeddings[kind])
        if isinstance(hidden, tuple):
            hidden = hidden[0]
    return model.lm_head(model.model.norm(hidden))


def check_logits(actual, expected, name):
    """Assert meaningful numerical agreement and report error magnitudes."""
    a, b = actual.float().flatten(), expected.float().flatten()
    metrics = {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "max_abs": (a - b).abs().max().item(),
        "mean_abs": (a - b).abs().mean().item(),
        "last_token_match": bool(torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))),
    }
    logger.info("%s: %s", name, metrics)
    assert metrics["cosine"] >= 0.99, (name, metrics)
    assert metrics["last_token_match"], (name, metrics)
    return metrics


def check_weights(bridge, models, reference):
    """Require every HF parameter to survive conversion bit for bit."""
    expected = reference.state_dict()
    seen = set()
    for name, weight in bridge.export_hf_weights(models, cpu=True, show_progress=False):
        assert name not in seen, name
        seen.add(name)
        assert torch.equal(weight, expected[name].cpu()), name
    assert seen == set(expected), (set(expected) - seen, seen - set(expected))
    return len(seen)


@pytest.mark.integration
@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "tp,sequence_parallel,legacy_provider",
    [(1, False, False), (2, True, False), (1, False, True)],
    ids=["builder-tp1", "builder-tp2-sp", "legacy-tp1"],
)
def test_olmo3_conversion(tmp_path: Path, tp: int, sequence_parallel: bool, legacy_provider: bool) -> None:
    """Check exact weights, packed/plain logits, gradients, training, and fresh reload."""
    if torch.cuda.device_count() < tp:
        pytest.skip(f"Requires {tp} CUDA devices")
    env = os.environ.copy()
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    devices = visible_devices.split(",") if visible_devices else [str(index) for index in range(tp)]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:tp])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={tp}",
        str(Path(__file__).resolve()),
        "--work-dir",
        str(tmp_path),
        "--tp",
        str(tp),
    ]
    if sequence_parallel:
        command.append("--sequence-parallel")
    if legacy_provider:
        command.append("--legacy-provider")
    for reload_checkpoint in (False, True):
        args = [*command, "--reload-checkpoint"] if reload_checkpoint else command
        subprocess.run(args, check=True, env=env, timeout=300)
        suffix = "reload" if reload_checkpoint else "validate"
        report_path = tmp_path / f"{suffix}-tp{tp}-sp{int(sequence_parallel)}.json"
        report = json.loads(report_path.read_text())
        assert report["exact_roundtrip_tensors"] > 0
        assert report["plain"]["last_token_match"]
        assert report["packed"]["last_token_match"]


def main() -> None:
    """Validate a toy model or a real local HF checkpoint using at most two GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--hf-model-path", type=Path)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--reload-checkpoint", action="store_true")
    parser.add_argument("--diagnose-norm-rounding", action="store_true")
    parser.add_argument("--legacy-provider", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == args.tp
    torch.manual_seed(472)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    hf_path = args.hf_model_path or args.work_dir / "hf"
    if args.hf_model_path is None and dist.get_rank() == 0 and not (hf_path / "config.json").exists():
        config = Olmo3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=65536,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            sliding_window=16,
            rms_norm_eps=1e-6,
            rope_theta=500000,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
                "attention_factor": 1.2079441541679836,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False,
        )
        toy = Olmo3ForCausalLM(config).bfloat16()
        # Learned norm weights expose cast-order bugs hidden by unit weights.
        with torch.no_grad():
            for name, parameter in toy.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.copy_(torch.linspace(0.5, 1.5, parameter.numel(), dtype=parameter.dtype))
        toy.save_pretrained(hf_path)
    dist.barrier()
    bridge = AutoBridge.from_hf_pretrained(str(hf_path), trust_remote_code=False, local_files_only=True)
    provider = bridge.to_megatron_provider() if args.legacy_provider else bridge.get_model_config()
    provider.tensor_model_parallel_size = args.tp
    provider.sequence_parallel = args.sequence_parallel
    provider.pipeline_model_parallel_size = 1
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.parallel_output = False
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    if args.legacy_provider:
        provider.initialize_model_parallel(seed=472)
    else:
        bridge._get_or_initialize_pg_collection(provider.transformer, seed=472)
    if args.reload_checkpoint:
        models = bridge.load_megatron_model(
            str(args.work_dir / "dcp"),
            wrap_with_ddp=False,
            mp_overrides={
                "tensor_model_parallel_size": args.tp,
                "pipeline_model_parallel_size": 1,
                "sequence_parallel": args.sequence_parallel,
            },
        )
    elif args.legacy_provider:
        models = provider.provide_distributed_model(wrap_with_ddp=False)
    else:
        models = bridge.get_model(provider, wrap_with_ddp=False)
    model = models[0].cuda().eval()
    reference = (
        Olmo3ForCausalLM.from_pretrained(
            hf_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    report = {
        "tp": args.tp,
        "sequence_parallel": args.sequence_parallel,
        "reload": args.reload_checkpoint,
        "api": "provider" if args.legacy_provider else "builder",
    }
    report["exact_roundtrip_tensors"] = check_weights(bridge, models, reference)
    probes = {}
    handles = []
    if args.diagnose_norm_rounding:
        native = model.module if hasattr(model, "module") else model
        mg_layer = native.decoder.layers[0]
        hf_layer = reference.model.layers[0]

        def capture(name):
            def hook(module, inputs, output):
                if name not in probes:
                    value = output[0] if isinstance(output, tuple) else output
                    probes[name] = value.detach().clone()
                    probes[name + ".input"] = inputs[0].detach().clone()

            return hook

        for name, module in {
            "mg_qkv": mg_layer.self_attention.linear_qkv,
            "hf_q": hf_layer.self_attn.q_proj,
            "hf_k": hf_layer.self_attn.k_proj,
            "mg_qnorm": mg_layer.self_attention.q_layernorm,
            "hf_qnorm": hf_layer.self_attn.q_norm,
            "mg_post_attn": mg_layer.self_attention.linear_proj.post_layernorm,
            "hf_post_attn": hf_layer.post_attention_layernorm,
            "mg_post_mlp": mg_layer.mlp.linear_fc2.post_layernorm,
            "hf_post_mlp": hf_layer.post_feedforward_layernorm,
        }.items():
            handles.append(module.register_forward_hook(capture(name)))
    ids = (torch.arange(64, device="cuda").unsqueeze(0) * 7 + 3) % provider.vocab_size
    positions = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        expected = reference_logits(reference, ids)
        actual = model(ids, positions, attention_mask=None)
        report["plain"] = check_logits(actual, expected, "plain")
        lengths = [32, 48]
        packed_ids = torch.cat((ids[:, : lengths[0]], ids[:, : lengths[1]]), dim=1)
        packed_positions = torch.cat([positions[:, :n] for n in lengths], dim=1)
        cumulative = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cumulative,
            cu_seqlens_kv=cumulative,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        packed_actual = model(packed_ids, packed_positions, attention_mask=None, packed_seq_params=packed)
        packed_expected = torch.cat([reference_logits(reference, ids[:, :n]) for n in lengths], dim=1)
        report["packed"] = check_logits(packed_actual, packed_expected, "packed")
    if args.diagnose_norm_rounding:
        for handle in handles:
            handle.remove()

        def component_metrics(a, b):
            a, b = a.float().flatten(), b.float().flatten()
            return {
                "cosine": F.cosine_similarity(a, b, dim=0).item(),
                "max_abs": (a - b).abs().max().item(),
                "mean_abs": (a - b).abs().mean().item(),
                "exact_fraction": (a == b).float().mean().item(),
            }

        components = {}
        for part in ("qnorm", "post_attn", "post_mlp"):
            for point in ("", ".input"):
                a = probes["mg_" + part + point]
                a = a.transpose(0, 1).reshape(1, 64, -1)
                b = probes["hf_" + part + point]
                components[part + point] = component_metrics(a, b)
        mg_qkv = probes["mg_qkv"]
        with torch.no_grad():
            torch_qkv = F.linear(probes["mg_qkv.input"], mg_layer.self_attention.linear_qkv.weight)
        components["TE_vs_torch_fused_qkv"] = component_metrics(mg_qkv, torch_qkv)
        q = torch_qkv.view(64, 1, provider.num_attention_heads, 3, provider.kv_channels)[:, :, :, 0]
        q = q.transpose(0, 1).reshape_as(probes["hf_q"])
        components["torch_fused_vs_separate_q"] = component_metrics(q, probes["hf_q"])
        report["first_layer_components"] = components
        logger.info("First layer component diagnosis: %s", json.dumps(components))
    if args.diagnose_norm_rounding:
        assert args.tp == 1, "The controlled reference swap requires TP=1"

        # Isolate TE versus HF RMSNorm reduction/arithmetic rounding with
        # the same formula and epsilon. Swap ONLY reference post/final norms;
        # keep checkpoint, projections, Q/K norms, RoPE, and attention fixed.
        def replacement(old):
            norm = TENorm(provider, provider.hidden_size, eps=provider.layernorm_epsilon).cuda().bfloat16()
            norm.weight.data.copy_(old.weight)
            return norm

        for layer in reference.model.layers:
            layer.post_attention_layernorm = replacement(layer.post_attention_layernorm)
            layer.post_feedforward_layernorm = replacement(layer.post_feedforward_layernorm)
        reference.model.norm = replacement(reference.model.norm)
        with torch.no_grad():
            swapped = reference_logits(reference, ids)
        report["hf_with_te_post_norms"] = check_logits(actual, swapped, "HF with TE post/final norms")
    if args.hf_model_path is None and not args.reload_checkpoint:
        bridge.save_megatron_model(models, str(args.work_dir / "dcp"))
        model.train()
        reference.train()
        model.zero_grad()
        reference.zero_grad()
        target = ids[:, 1:].flatten()
        expected = reference_logits(reference, ids)
        reference_loss = F.cross_entropy(expected[:, :-1].float().reshape(-1, provider.vocab_size), target)
        reference_loss.backward()
        actual = model(ids, positions, attention_mask=None)
        loss = F.cross_entropy(actual[:, :-1].float().reshape(-1, provider.vocab_size), target)
        loss.backward()
        gradient_wrapper = SimpleNamespace(
            ddp_config=SimpleNamespace(use_megatron_fsdp=False), named_parameters=model.named_parameters
        )
        _allreduce_non_tensor_model_parallel_grads([gradient_wrapper], model.config)
        originals = {name: p.detach().clone() for name, p in model.named_parameters()}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            parameter.data.copy_(parameter.grad)
        gradients = dict(bridge.export_hf_weights(models, cpu=True, show_progress=False))
        gradient_metrics = {}
        for name, parameter in reference.named_parameters():
            a, b = gradients[name].float().flatten(), parameter.grad.cpu().float().flatten()
            cosine = F.cosine_similarity(a, b, dim=0).item()
            assert torch.isfinite(a).all(), name
            assert cosine > 0.98, (name, cosine)
            gradient_metrics[name] = cosine
        report["min_gradient_cosine"] = min(gradient_metrics.values())
        for name, parameter in model.named_parameters():
            parameter.data.copy_(originals[name])
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(ids, positions, attention_mask=None)
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, provider.vocab_size), target)
            losses.append(loss.item())
            loss.backward()
            _allreduce_non_tensor_model_parallel_grads([gradient_wrapper], model.config)
            optimizer.step()
        assert all(torch.isfinite(torch.tensor(losses))), losses
        assert losses[-1] < losses[0], losses
        report["training_losses"] = losses
    if dist.get_rank() == 0:
        suffix = "reload" if args.reload_checkpoint else "validate"
        report_path = args.work_dir / f"{suffix}-tp{args.tp}-sp{int(args.sequence_parallel)}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("OLMO3_VALIDATION_PASS %s", json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
