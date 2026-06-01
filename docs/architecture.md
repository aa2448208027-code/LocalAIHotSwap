# Architecture

## Design boundary

`llama-server` owns inference and model memory. In the default mode it runs as a
router process that can load and unload models dynamically. HotModelReplacement
owns conversation state, preset prompts, model lifecycle decisions, and request
routing.

This split is deliberate. It avoids relying on internal KV cache state as an
interchange format. The proxy preserves logical context by replaying the same
system prompt and session messages after a switch.

## Switch lifecycle

The default `zero_overlap` switch flow is:

1. Acquire the switch lock and mark the runtime as switching.
2. Reject new chat requests with HTTP 503 while the switch is active.
3. Wait for in-flight chat requests to finish, up to
   `switch_drain_timeout_seconds`.
4. Ask the router to unload the active model.
5. Optionally poll `nvidia-smi` until used memory is below a configured settle
   threshold or the timeout expires.
6. Ask the router to load the target model.
7. Poll router model status until the target is loaded.
8. Mark the target as active and resume chat traffic.

The flow optimizes peak VRAM by setting router `--models-max 1` and avoiding two
loaded GGUF models at the same time. The tradeoff is that the target model pays
full load time.

## Context preservation

The proxy stores sessions as message history:

- A configured preset system prompt is always prepended.
- Incoming client messages are appended to the session.
- The assistant response is appended after the backend returns.
- When the active model changes, the same stored messages are sent to the new
  backend on the next request.

This keeps the user-visible conversation and preset stable. The target model
still needs to prefill the replayed prompt.

Two budget controls limit the replay cost:

- `max_session_messages` trims persisted history to recent messages.
- `max_prompt_chars` trims old history when building a request while preserving
  the system prompt and the current incoming messages.
- `max_prompt_tokens` trims old history against the active model's tokenizer
  budget. In `auto` mode the proxy starts with a configurable
  character-per-token estimate, then uses `llama-server` template/tokenize
  endpoints only when the estimate indicates trimming is needed.

## Latency and memory notes

- Cross-model KV cache reuse is unsafe as a general guarantee.
- Same-model prefix reuse can still help inside `llama-server` when requests hit
  compatible slots and the prompt prefix is stable.
- Smaller quantized GGUF models reduce both load time and VRAM pressure.
- `--models-max 1`, `--parallel 1`, smaller `--ctx-size`, and quantized
  `--cache-type-k/v` are the highest-impact settings for peak VRAM.
- Unified KV plus RAM prompt cache can reduce repeated prefill work inside the
  same model instance, subject to llama.cpp model support.
- Keep the system prompt deterministic. Dynamic timestamps, tool inventories, or
  random request metadata can break prefix reuse.
- For long conversations, add summarization or trimming before the configured
  context window is exceeded.

## Failure behavior

If the target model fails to load, the switch reports an error. When a previous
model existed, the orchestrator attempts to load it again. Because
`zero_overlap` keeps only one model loaded, rollback also pays load latency.

## Compatibility mode

The code also supports `backend_mode = "process"` for older `llama-server`
builds. In that mode each model has its own server port and a switch stops the
old process before starting the new process. Router mode has lower orchestration
cost and tracks current `llama.cpp` model-management APIs.

Switch-time load and unload work is intentionally performed outside the primary
condition lock. New chat requests can then fail quickly with a switching error
while the backend is loading a model, instead of blocking behind a long critical
section.
