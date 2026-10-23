# OLMo 3

OLMo 3 support is selected from the Hugging Face config by `AutoBridge`.
The model dimensions and local/full attention schedule come from that config;
there are no size-specific provider classes.

## Import and export a published checkpoint

Run from the repository root with a prepared Bridge environment and one GPU.
Choose a persistent output directory in `MODEL_WORK_DIR`.

```bash
export HF_MODEL="${HF_MODEL:-allenai/Olmo-3-7B-Think-SFT}"
export MODEL_WORK_DIR="${MODEL_WORK_DIR:?Set an output directory}"

./scripts/conversion/convert.sh import \
  --hf-model "$HF_MODEL" \
  --megatron-path "$MODEL_WORK_DIR/megatron" \
  --device gpu --gpus-per-node 1 --tp 1 --torch-dtype bfloat16

./scripts/conversion/convert.sh export \
  --hf-model "$HF_MODEL" \
  --megatron-path "$MODEL_WORK_DIR/megatron/iter_0000000" \
  --hf-path "$MODEL_WORK_DIR/hf-export" \
  --device gpu --gpus-per-node 1 --tp 1 --torch-dtype bfloat16
```

## Inference

```bash
HF_MODEL=allenai/Olmo-3-7B-Think-SFT \
  bash examples/models/olmo3/inference.sh
```

The wrapper uses the shared generation example and its full-prefix path.
Fused inference RoPE and flash decode are not supported by this integration.

## Builder API

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("allenai/Olmo-3-7B-Think-SFT")
config = bridge.get_model_config()
config.tensor_model_parallel_size = 1
config.pipeline_model_parallel_size = 1
models = bridge.get_model(config, wrap_with_ddp=False)
```

The legacy provider adapter remains available to existing verl callers.
See [architecture, validation, and limitations](../../../docs/models/olmo3/olmo3.md).
