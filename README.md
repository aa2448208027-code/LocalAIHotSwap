# HotModelReplacement

HotModelReplacement is a small control plane for `llama.cpp` deployments. It keeps
conversation state and preset prompts outside `llama-server`, then switches the
active local GGUF model through the `llama-server` router.

The default switch policy is `zero_overlap`: run one `llama-server` router,
unload the current model, optionally wait for GPU memory to settle, then load
the target model. That keeps switch-time VRAM peaks low at the cost of reloading
weights and replaying the saved prompt/messages into the new model.

## What this project guarantees

- Preset prompt and session messages are preserved across model switches.
- Only one model is loaded in the managed `llama-server` router during a default
  switch.
- Active generations are drained before a switch unloads the current model.
- Long sessions can be bounded by message count and prompt character budget.
- Clients can keep using an OpenAI-compatible `/v1/chat/completions` endpoint.
- Streaming chat completions are forwarded as server-sent events when clients set
  `stream = true`.
- Switches are serialized with a lock so chat requests cannot race the process
  lifecycle.

## What it does not claim

- Cross-model KV cache reuse is not treated as a valid optimization. KV entries
  depend on model weights, tokenizer behavior, attention layout, and runtime
  state.
- Switching models still has weight load latency and prompt prefill latency.
- The proxy can minimize orchestration overhead, but it cannot remove the
  compute cost of evaluating a long preserved context on the target model.

## Quick start

1. Build or install a recent `llama.cpp` and make sure `llama-server` is on
   `PATH`. Router mode must support `/models/load` and `/models/unload`.
2. Copy the example config and point the model paths at local GGUF files:

   ```powershell
   Copy-Item configs\models.example.toml configs\models.toml
   ```

3. Start the proxy:

   ```powershell
   hotmodel serve --config configs\models.toml
   ```

4. Switch models:

   ```powershell
   hotmodel switch qwen3-small --config configs\models.toml
   ```

5. Send chat requests to the proxy:

   ```powershell
   curl http://127.0.0.1:18080/v1/chat/completions `
     -H "Content-Type: application/json" `
     -d "{\"model\":\"active\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
   ```

## Configuration

See [configs/models.example.toml](configs/models.example.toml). The TOML file
defines the proxy and router settings. The INI file defines the model presets
consumed by `llama-server` router mode.

Small Qwen-family GGUF models are a practical starting point because they keep
load latency and VRAM pressure low during iteration. Use `models_max = 1` for
the lowest peak VRAM. Raising it reduces switch latency by allowing multiple
loaded models, with a direct VRAM cost.

The most important performance knobs are:

- `router.models_max = 1`: prevents two model weights from being resident during
  a switch.
- `router.parallel = 1`: minimizes KV cache allocation.
- `router.ctx_size`: caps unified KV cache size and max prompt length.
- `router.cache_type_k` / `router.cache_type_v`: `q8_0` cuts KV memory compared
  with `f16`, with quality and speed tradeoffs to validate on your workload.
- `session.max_session_messages` and `session.max_prompt_chars`: bound replayed
  context so model switches do not turn into long prefill stalls.
- `session.max_prompt_tokens`: asks the active `llama-server` to apply the
  model chat template and tokenize the prompt, then drops older history until the
  token budget fits. In `auto` mode the proxy starts with a fast estimate and
  calls tokenizer endpoints only when the estimate indicates trimming is needed.
- `server.switch_drain_timeout_seconds`: waits for active generations to finish
  before unloading the current model.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the lifecycle model,
latency tradeoffs, and operational notes.

See [docs/performance.md](docs/performance.md) for the current performance
review, compatibility notes, and future branch plan.
