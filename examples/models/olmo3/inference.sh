#!/usr/bin/env bash
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

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path "${HF_MODEL:-allenai/Olmo-3-7B-Think-SFT}" \
  --prompt "${PROMPT:-What is 2 + 2?}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --apply-chat-template --legacy-full-prefix "$@"
