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
