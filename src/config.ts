import { createConfigSchematics } from "@lmstudio/sdk";

export const configSchematics = createConfigSchematics()
  .field(
    "model",
    "string",
    {
      displayName: "Model",
      subtitle: "Enter the exact model name, including its tag.",
      placeholder: "LiquidAI/lfm2.5-1.2b-instruct:q8_0",
    },
    "",
  )
  .build();

export const globalConfigSchematics = createConfigSchematics()
  .field(
    "serverUrl",
    "string",
    {
      displayName: "Ollama Server URL",
      subtitle: "Ollama server address accessible from this device.",
      placeholder: "http://server:11434",
    },
    "",
  )
  .field(
    "apiKey",
    "string",
    {
      displayName: "API Key (Optional)",
      subtitle: "Only needed if your server requires authentication.",
      isProtected: true,
    },
    "",
  )
  .build();
