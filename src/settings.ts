import { type InferParsedConfig } from "@lmstudio/sdk";
import { type CheckboxNumericValue, type globalConfigSchematics } from "./config";

export interface GenerationOptions {
  think?: boolean | "low" | "medium" | "high";
  options?: {
    temperature?: number;
    top_p?: number;
    num_predict?: number;
    seed?: number;
    top_k?: number;
    repeat_penalty?: number;
    min_p?: number;
  };
}

function optionalNumber(
  input: CheckboxNumericValue, name: string, min: number, max = Infinity, integer = false,
): number | undefined {
  if (!input.checked) return undefined;
  const value = input.value;
  if (!Number.isFinite(value) || value < min || value > max
    || (integer && !Number.isSafeInteger(value))) {
    throw new Error(`Enter a valid ${name}${integer ? " integer" : " value"}.`);
  }
  return value;
}

export function resolveGenerationOptions(
  config: InferParsedConfig<typeof globalConfigSchematics>,
): GenerationOptions {
  const result: GenerationOptions = {};
  const thinking = config.get("thinking");
  switch (thinking) {
    case "default": break;
    case "on": result.think = true; break;
    case "off": result.think = false; break;
    case "low": case "medium": case "high": result.think = thinking; break;
    default: throw new Error("Select a valid Thinking mode.");
  }

  const options: NonNullable<GenerationOptions["options"]> = {};
  options.temperature = optionalNumber(config.get("temperatureOption"), "Temperature", 0, 2);
  options.num_predict = optionalNumber(config.get("maxTokensOption"), "Limit Response Length", -1, 2147483647, true);
  if (options.num_predict === 0) {
    throw new Error("Limit Response Length must be a positive integer or -1.");
  }
  options.top_k = optionalNumber(config.get("topKOption"), "Top K Sampling", 0, 2147483647, true);
  options.repeat_penalty = optionalNumber(config.get("repeatPenaltyOption"), "Repeat Penalty", 0);
  options.top_p = optionalNumber(config.get("topPOption"), "Top P Sampling", 0, 1);
  options.min_p = optionalNumber(config.get("minPOption"), "Min P Sampling", 0, 1);
  options.seed = optionalNumber(config.get("seedOption"), "Seed", -1, 2147483647, true);
  if (Object.values(options).some(value => value !== undefined)) result.options = options;
  return result;
}

export interface OllamaSettingsInput {
  serverUrl: string;
  apiKey: string;
  model: string;
}

export interface OllamaSettings {
  serverUrl: string;
  apiKey: string | undefined;
  model: string;
}

export function resolveOllamaSettings(input: OllamaSettingsInput): OllamaSettings {
  const inputUrl = input.serverUrl.trim();
  if (!inputUrl) {
    throw new Error("Enter an Ollama Server URL.");
  }
  if (!/^https?:\/\//i.test(inputUrl)) {
    throw new Error("Ollama Server URL must use HTTP or HTTPS.");
  }
  if (/[\x00-\x1f\x7f]/.test(inputUrl)) {
    throw new Error("Ollama Server URL must not contain control characters.");
  }

  let url: URL;
  try {
    url = new URL(inputUrl);
  } catch {
    throw new Error("Enter a valid HTTP or HTTPS Ollama Server URL.");
  }

  if (url.username || url.password || /[?#]/.test(inputUrl)) {
    throw new Error("Ollama Server URL must not include credentials, a query, or a fragment.");
  }

  const path = url.pathname.replace(/\/+$/, "");
  if (/\/(?:api\/(?:chat|generate|tags|ps|show)|v1\/(?:chat\/completions|completions|models|responses|embeddings))$/.test(path)) {
    throw new Error("Use the Ollama Server URL without an endpoint path.");
  }

  // Strip API suffixes while preserving proxy paths.
  url.pathname = path.replace(/\/(?:api|v1)$/, "");
  const serverUrl = url.toString().replace(/\/+$/, "");
  const model = input.model.trim();
  if (!model || /[\s\x00-\x1f\x7f]/.test(model)) {
    throw new Error("Enter a Model name without whitespace or control characters.");
  }

  const apiKey = input.apiKey.trim();
  if (/[\r\n]/.test(apiKey)) {
    throw new Error("API Key must not contain line breaks.");
  }
  if (/[^\x20-\x7e]/.test(apiKey)) {
    throw new Error("API Key must contain only printable ASCII characters.");
  }

  return {
    serverUrl,
    apiKey: apiKey || undefined,
    model,
  };
}
