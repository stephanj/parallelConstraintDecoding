#!/usr/bin/env bash
# Regenerates the FFM bindings for libllama (src/main/java/llama) from the Homebrew llama.cpp headers.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Homebrew ships llama.cpp and ggml as separate formulas since llama.cpp 0.4.x.
INC="${LLAMA_INCLUDE:-/opt/homebrew/opt/llama.cpp/include}"
GGML_INC="${GGML_INCLUDE:-/opt/homebrew/opt/ggml/include}"
OUT="$DIR/../src/main/java"
"${JEXTRACT:-$DIR/jextract-22/bin/jextract}" \
  --output "$OUT" \
  --target-package llama \
  --header-class-name Llama \
  -I "$INC" -I "$GGML_INC" \
  --include-function llama_backend_init \
  --include-function llama_backend_free \
  --include-function ggml_backend_load_all \
  --include-function llama_log_set \
  --include-function llama_model_default_params \
  --include-function llama_model_load_from_file \
  --include-function llama_model_free \
  --include-function llama_model_get_vocab \
  --include-function llama_vocab_n_tokens \
  --include-function llama_vocab_is_eog \
  --include-function llama_model_chat_template \
  --include-function llama_chat_apply_template \
  --include-function llama_model_meta_val_str \
  --include-struct llama_chat_message \
  --include-function llama_context_default_params \
  --include-function llama_init_from_model \
  --include-function llama_free \
  --include-function llama_get_memory \
  --include-function llama_memory_clear \
  --include-function llama_memory_seq_cp \
  --include-function llama_memory_seq_rm \
  --include-function llama_tokenize \
  --include-function llama_token_to_piece \
  --include-function llama_batch_init \
  --include-function llama_batch_free \
  --include-function llama_decode \
  --include-function llama_get_logits_ith \
  --include-function llama_synchronize \
  --include-function llama_sampler_chain_default_params \
  --include-function llama_sampler_chain_init \
  --include-function llama_sampler_chain_add \
  --include-function llama_sampler_init_grammar \
  --include-function llama_sampler_init_greedy \
  --include-function llama_sampler_sample \
  --include-function llama_sampler_free \
  --include-struct llama_sampler_chain_params \
  --include-struct llama_model_params \
  --include-struct llama_context_params \
  --include-struct llama_batch \
  --include-constant LLAMA_FLASH_ATTN_TYPE_ENABLED \
  --include-constant LLAMA_FLASH_ATTN_TYPE_DISABLED \
  "$INC/llama.h" "$GGML_INC/ggml-backend.h"
echo "bindings written to $OUT/llama"
