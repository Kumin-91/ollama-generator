import {
  createConfigSchematics,
  type ConfigSchematicsBuilder,
  type GlobalKVFieldValueTypeLibraryMap,
  type VirtualConfigSchematics,
} from "@lmstudio/sdk";

export type CheckboxNumericValue = GlobalKVFieldValueTypeLibraryMap["checkboxNumeric"]["value"];

type SettingsFieldType = "string" | "select" | "checkboxNumeric";
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

// The SDK runtime supports checkboxNumeric, but the public builder type omits it.
function createSettingsBuilder(): SettingsBuilder<{}> {
  return createConfigSchematics() as SettingsBuilder<{}>;
}

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

export const globalConfigSchematics = createSettingsBuilder()
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
      displayName: "API Key",
      placeholder: "(Optional) Enter API key",
      subtitle: "Only needed if your server requires authentication.",
      isProtected: true,
    },
    "",
  )
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
  .build();
