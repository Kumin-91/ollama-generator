# Ollama Generator

Use your Ollama models in LM Studio.

Based on [LM Studio’s OpenAI Adapter](https://lmstudio.ai/lmstudio/openai-compat-endpoint).

[View in LM Studio](https://lmstudio.ai/miu-3/ollama-generator)

Limitations:

- Image attachments are currently unavailable with the adaptor alone.
- The optional Alpha context setting shows a context usage line below responses; the native context gauge is unavailable.
    - The SDK does not expose an API to update the native context gauge.
- Ollama's generic `stop` is shown as EOS; it does not distinguish EOS from stop strings.
- LM Studio shows a Prediction Loop warning, sometimes as `loader.info.activePredictionLoopHandlerWarning`; the SDK cannot hide it.
