# Ollama Generator

Based on [LM Studio’s OpenAI Adapter](https://lmstudio.ai/lmstudio/openai-compat-endpoint).

[View in LM Studio](https://lmstudio.ai/miu-3/ollama-generator)

Use your Ollama models in LM Studio.

- Streaming, thinking, and tool calling.
- Optional sampling settings and response limits. Server defaults when unset.
- Native generation stats and optional context status.

Limitations:

- Image attachments are currently unavailable with the adaptor alone.
- Native context counters are not synchronized with Ollama. The SDK provides no update API.
- Ollama's generic `stop` is shown as EOS; it does not distinguish stop strings.
- The SDK cannot hide Prediction Loop warnings (`loader.info.activePredictionLoopHandlerWarning`).
