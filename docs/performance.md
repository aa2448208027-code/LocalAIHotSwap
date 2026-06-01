# Performance Review

This document records the current bottlenecks, implemented mitigations, and
future branches that should stay separate to avoid large, tangled changes.

## Current Bottlenecks

## VRAM Peak During Switch

The largest switch-time risk is having two GGUF models resident at once. The
project keeps `router.models_max = 1` and uses explicit unload/load operations
so peak VRAM tracks one active model plus allocator and runtime overhead.

Further reductions come from:

- smaller quantized models;
- lower `ctx_size`;
- `parallel = 1`;
- quantized KV cache via `cache_type_k` and `cache_type_v`;
- keeping model autoload disabled so the proxy controls lifecycle.

## Prefill Cost After Switch

Cross-model KV cache reuse is not a safe general mechanism. The proxy therefore
preserves logical context and replays messages. That means long conversations
increase first-token latency after a switch.

Implemented controls:

- `max_session_messages` limits persisted history.
- `max_prompt_chars` limits the request prompt while preserving the system
  prompt and current incoming messages.
- `max_prompt_tokens` can use the active `llama-server` `/apply-template` and
  `/tokenize` endpoints to fit the prompt against the target model's chat
  template. `token_budget_mode = "auto"` starts with a fast estimate, only calls
  the tokenizer endpoints when trimming appears necessary, and falls back to the
  estimate when those endpoints are unavailable.
- deterministic preset prompts keep same-model prefix reuse opportunities open.

Future work should cache repeated token counts for unchanged session prefixes.
Keep that separate because it needs invalidation rules tied to model identity,
prompt template version, and preset prompt changes.

## Request Drain During Switch

Unloading a model while a generation is active can fail or corrupt the user
experience. The orchestrator now closes the gate for new chats, waits for
in-flight chats to finish, then unloads the current model.

The long-running unload/load work happens outside the main condition lock, so
new requests can quickly fail with a clear `model switch in progress` error
instead of blocking until the target model finishes loading.

Streaming requests count as in-flight until the SSE iterator completes or the
client disconnects. Completed streams are written to session history after the
last chunk. Cancelled streams release the switch drain without storing partial
assistant output.

## Subprocess Output Backpressure

`llama-server` can produce enough output to fill an unread pipe. A filled pipe
can stall the child process. Managed subprocesses now send stdout and stderr to
`DEVNULL` by default.

If operational diagnostics are needed, add explicit log-file support with
rotation. Do not reintroduce unread pipes.

## Compatibility Notes

The router status parser accepts several response shapes:

- `{"status": {"value": "LOADED"}}`;
- `{"status": "UNLOADED"}`;
- `{"loaded": true}`.

This makes the code more tolerant of small `llama.cpp` response shape changes
while keeping failures explicit.

## Future Branches

Keep these as separate PRs:

- prefix token-count caching for tokenizer-aware budgets;
- richer streaming metadata and usage accounting;
- optional waiting queue for switch requests;
- log-file rotation and process diagnostics;
- real `llama-server` smoke tests gated by environment variables;
- GPU telemetry snapshots before unload, after unload, and after load;
- model compatibility profiles for Qwen, Gemma, Llama, and multimodal variants.

## Sources

- llama.cpp server README:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- ggml-org model-management article:
  <https://huggingface.co/blog/ggml-org/model-management-in-llamacpp>
