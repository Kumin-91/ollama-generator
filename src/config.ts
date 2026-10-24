import {
  createConfigSchematics,
  type ConfigSchematicsBuilder,
  type GeneratorController,
  type GlobalKVFieldValueTypeLibraryMap,
  type VirtualConfigSchematics,
} from "@lmstudio/sdk";
import { resolveGenerationOptions, resolveOllamaSettings } from "./settings";
import { DEFAULT_REQUEST_TIMEOUT_SECONDS } from "./ollama-client";

export type CheckboxNumericValue = GlobalKVFieldValueTypeLibraryMap["checkboxNumeric"]["value"];

type SettingsFieldType = Parameters<ConfigSchematicsBuilder<{}>["field"]>[1] | "checkboxNumeric";
type SettingsBuilder<T extends VirtualConfigSchematics> = Omit<ConfigSchematicsBuilder<T>, "field"> & {
  field<TKey extends string, TType extends SettingsFieldType>(
    key: TKey,
    type: TType,
    params: GlobalKVFieldValueTypeLibraryMap[TType]["param"],
    defaultValue: GlobalKVFieldValueTypeLibraryMap[TType]["value"],
  ): SettingsBuilder<T & {
    [K in TKey]: { key: K; type: GlobalKVFieldValueTypeLibraryMap[TType]["value"]; valueTypeKey: TType };
  }>;
};

export const configSchematics = createConfigSchematics()
  .field("model", "string", {
    displayName: "Model",
    subtitle: "Enter the exact model name, including its tag.",
    placeholder: "LiquidAI/lfm2.5-2.6b:q4_k_m",
  }, "LiquidAI/lfm2.5-2.6b:q4_k_m")
  .build();

// The SDK runtime supports checkboxNumeric, but the public builder type omits it.
export const globalConfigSchematics = (createConfigSchematics() as SettingsBuilder<{}>)
  .field("serverUrl", "string", {
    displayName: "Ollama Server URL",
    subtitle: "Ollama server address. Use HTTPS for remote connections.",
    placeholder: "http://server:11434",
  }, "")
  .field("apiKey", "string", {
    displayName: "API Key",
    placeholder: "(Optional) Enter API key",
    subtitle: "Only needed if your server requires authentication.",
    isProtected: true,
  }, "")
  .field("thinking", "select", {
    displayName: "Thinking",
    subtitle: "Supported modes depend on the model.",
    options: [
      { value: "default", displayName: "Server default" },
      { value: "off", displayName: "Off" },
      { value: "on", displayName: "On" },
      { value: "low", displayName: "Low" },
      { value: "medium", displayName: "Medium" },
      { value: "high", displayName: "High" },
    ],
  }, "default")
  .field("temperatureOption", "checkboxNumeric", {
    displayName: "(Optional) Temperature",
    min: 0, max: 2, step: 0.01, precision: 2,
    slider: { min: 0, max: 2, step: 0.01 },
  }, { checked: false, value: 0.8 })
  .field("maxTokensOption", "checkboxNumeric", {
    displayName: "(Optional) Limit Response Length",
    hint: "Maximum output tokens. Positive integer, or -1 for unlimited.",
    min: -1, max: 2147483647, int: true,
  }, { checked: false, value: 512 })
  .field("topKOption", "checkboxNumeric", {
    displayName: "(Optional) Top K Sampling",
    hint: "Use 0 to disable filtering.",
    min: 0, max: 2147483647, int: true,
  }, { checked: false, value: 40 })
  .field("repeatPenaltyOption", "checkboxNumeric", {
    displayName: "(Optional) Repeat Penalty",
    hint: "Use 1 for no penalty.",
    min: 0, step: 0.01, precision: 2,
  }, { checked: false, value: 1 })
  .field("topPOption", "checkboxNumeric", {
    displayName: "(Optional) Top P Sampling",
    min: 0, max: 1, step: 0.01, precision: 2,
    slider: { min: 0, max: 1, step: 0.01 },
  }, { checked: false, value: 0.9 })
  .field("minPOption", "checkboxNumeric", {
    displayName: "(Optional) Min P Sampling",
    min: 0, max: 1, step: 0.01, precision: 2,
    slider: { min: 0, max: 1, step: 0.01 },
  }, { checked: false, value: 0 })
  .field("seedOption", "checkboxNumeric", {
    displayName: "(Optional) Seed",
    hint: "Use -1 for a random seed.",
    min: -1, max: 2147483647, int: true,
  }, { checked: false, value: 0 })
  .field("requestTimeoutSeconds", "numeric", {
    displayName: "Request Timeout (seconds)",
    subtitle: "Maximum time per model request, including loading and streaming.",
    min: 1, max: 86400, int: true,
  }, DEFAULT_REQUEST_TIMEOUT_SECONDS)
  .field("maxToolRounds", "numeric", {
    displayName: "Maximum Tool Rounds",
    subtitle: "Maximum rounds of tool calls per response.",
    min: 1, max: 128, int: true,
  }, 16)
  .field("contextUsage", "boolean", {
    displayName: "(Beta) Context Usage",
    subtitle: "Show context usage for the last request in a status row.",
  }, false)
  .build();

export function resolveGenerationConfig(
  ctl: Pick<GeneratorController, "getPluginConfig" | "getGlobalPluginConfig">,
) {
  const config = ctl.getPluginConfig(configSchematics);
  const globalConfig = ctl.getGlobalPluginConfig(globalConfigSchematics);
  return {
    settings: resolveOllamaSettings({
      serverUrl: globalConfig.get("serverUrl"),
      apiKey: globalConfig.get("apiKey"),
      model: config.get("model"),
    }),
    options: resolveGenerationOptions(globalConfig),
    contextUsage: globalConfig.get("contextUsage"),
    timeoutMs: limitedInteger(globalConfig.get("requestTimeoutSeconds"), "Request Timeout", 86400) * 1000,
    maxToolRounds: limitedInteger(globalConfig.get("maxToolRounds"), "Maximum Tool Rounds", 128),
  };
}

function limitedInteger(value: number, name: string, max: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > max) {
    throw new Error(`${name} must be an integer between 1 and ${max}.`);
  }
  return value;
}
