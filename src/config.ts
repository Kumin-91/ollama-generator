import { createConfigSchematics } from "@lmstudio/sdk";

// This file contains the definition of configuration schematics for your plugin.

export const configSchematics = createConfigSchematics()
  .field(
    "model",
    "select",
    {
      displayName: "Model",
      subtitle: "Select the model to use for generation.",
      options: [
        { value: "claude-sonnet-4-20250514", displayName: "claude-sonnet-4-20250514" },
        { value: "claude-opus-4-20250514", displayName: "claude-opus-4-20250514" },
        { value: "gpt-4.1-2025-04-14", displayName: "gpt-4.1-2025-04-14" },
        { value: "gpt-4.1-mini-2025-04-14", displayName: "gpt-4.1-mini-2025-04-14" },
      ],
    },
    "claude-sonnet-4-20250514",
  )
  .build();

export const globalConfigSchematics = createConfigSchematics()
  .field(
    "openaiApiKey",
    "string",
    {
      displayName: "OpenAI API Key",
      isProtected: true,
      placeholder: "sk-...",
    },
    "",
  )
  .field(
    "anthropicApiKey",
    "string",
    {
      displayName: "Anthropic API Key",
      isProtected: true,
      placeholder: "ant-...",
    },
    "",
  )
    .field(
    "overrideBaseUrl",
    "string",
    {
      displayName: "(Optional) Override Base URL",
      subtitle: "Override the base URL for API calls. Leave empty to let the plugin guess the base URL.",
      placeholder: "https://api.example.com/v1",
    },
    "",
  )
  .build();
