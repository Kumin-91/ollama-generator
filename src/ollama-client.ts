import { type LLMTool, type ToolCallRequest } from "@lmstudio/sdk";
import { randomUUID } from "node:crypto";
import { type GenerationOptions } from "./generation-options";
import { type OllamaSettings } from "./ollama-settings";

export type OllamaToolCall = ToolCallRequest & {
  id: string;
  arguments: Record<string, unknown>;
};

export interface OllamaMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  images?: string[];
  tool_calls?: { id: string; function: { name: string; arguments: Record<string, unknown> } }[];
  tool_name?: string;
  tool_call_id?: string;
}

export interface OllamaUsage {
  promptTokens?: number;
  outputTokens?: number;
  outputSeconds?: number;
}

interface OllamaChunk {
  done: boolean;
  doneReason?: string;
  content: string;
  thinking: string;
  toolCalls: OllamaToolCall[];
  usage?: OllamaUsage;
}

export interface OllamaGenerationRequest {
  settings: OllamaSettings;
  messages: OllamaMessage[];
  tools: LLMTool[];
  options: GenerationOptions;
  signal: AbortSignal;
  onThinking: (text: string) => void;
  onContent: (text: string) => void;
}

export interface OllamaGenerationResult {
  toolCalls: OllamaToolCall[];
  doneReason?: string;
  usage?: OllamaUsage;
  firstSeconds?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonnegativeInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function durationSeconds(value: unknown): number | undefined {
  const duration = nonnegativeInteger(value);
  return duration === undefined ? undefined : duration / 1e9;
}

function parseToolCall(value: unknown): OllamaToolCall {
  if (!isRecord(value) || !isRecord(value.function)
    || typeof value.function.name !== "string" || !value.function.name
    || !isRecord(value.function.arguments)
    || (value.id !== undefined && typeof value.id !== "string")) {
    throw new Error("Ollama returned an invalid tool call.");
  }
  return {
    type: "function",
    id: value.id || randomUUID(),
    name: value.function.name,
    arguments: value.function.arguments,
  };
}

function parseChunk(line: string): OllamaChunk {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch {
    throw new Error("Ollama returned invalid streaming JSON.");
  }
  if (!isRecord(value)) throw new Error("Ollama returned an invalid stream packet.");
  if (typeof value.error === "string") throw new Error(`Ollama: ${value.error}`);
  if (typeof value.done !== "boolean" || (value.message === undefined && !value.done)) {
    throw new Error("Ollama returned an invalid stream packet.");
  }
  const message = value.message === undefined ? {} : value.message;
  if (!isRecord(message)
    || (message.content !== undefined && typeof message.content !== "string")
    || (message.thinking !== undefined && typeof message.thinking !== "string")
    || (message.tool_calls !== undefined && !Array.isArray(message.tool_calls))) {
    throw new Error("Ollama returned an invalid chat message.");
  }
  return {
    done: value.done,
    ...(value.done && typeof value.done_reason === "string" ? { doneReason: value.done_reason } : {}),
    content: message.content ?? "",
    thinking: message.thinking ?? "",
    toolCalls: (message.tool_calls ?? []).map(parseToolCall),
    ...(value.done ? { usage: {
      promptTokens: nonnegativeInteger(value.prompt_eval_count),
      outputTokens: nonnegativeInteger(value.eval_count),
      outputSeconds: durationSeconds(value.eval_duration),
    } } : {}),
  };
}

async function requestOllama(
  settings: OllamaSettings,
  endpoint: "show" | "chat" | "ps",
  body: Record<string, unknown> | undefined,
  signal: AbortSignal,
): Promise<Response> {
  signal.throwIfAborted();
  const response = await fetch(`${settings.serverUrl}/api/${endpoint}`, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "Content-Type": "application/json",
      ...(settings.apiKey ? { Authorization: `Bearer ${settings.apiKey}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify({ model: settings.model, ...body }),
    signal,
  });
  if (!response.ok) {
    const error: unknown = await response.json().catch(() => null);
    throw new Error(isRecord(error) && typeof error.error === "string"
      ? `Ollama (${response.status}): ${error.error}`
      : `Ollama request failed (HTTP ${response.status}).`);
  }
  return response;
}

function taggedModel(model: string): string {
  return model.lastIndexOf(":") > model.lastIndexOf("/") ? model : `${model}:latest`;
}

export async function getOllamaContextLength(
  settings: OllamaSettings, signal: AbortSignal,
): Promise<number | undefined> {
  try {
    // Optional statistics must not leave a completed answer waiting indefinitely.
    const lookupSignal = AbortSignal.any([signal, AbortSignal.timeout(3000)]);
    const response = await requestOllama(settings, "ps", undefined, lookupSignal);
    const data: unknown = await response.json();
    if (!isRecord(data) || !Array.isArray(data.models)) return undefined;
    const limits = new Set<number>();
    for (const model of data.models) {
      if (!isRecord(model) || ![model.name, model.model].some(name =>
        typeof name === "string" && taggedModel(name) === taggedModel(settings.model))) continue;
      const limit = nonnegativeInteger(model.context_length);
      if (limit === undefined || limit === 0) return undefined;
      limits.add(limit);
    }
    return limits.size === 1 ? limits.values().next().value : undefined;
  } catch {
    return undefined;
  }
}

export async function generateOllama({
  settings, messages, tools, options, signal, onThinking, onContent,
}: OllamaGenerationRequest): Promise<OllamaGenerationResult> {
  if (messages.some(message => message.images?.length)) {
    const response = await requestOllama(settings, "show", {}, signal);
    const model: unknown = await response.json();
    if (!isRecord(model) || !Array.isArray(model.capabilities)
      || !model.capabilities.every(capability => typeof capability === "string")) {
      throw new Error("Ollama did not return valid model capabilities for image input.");
    }
    if (!model.capabilities.includes("vision")) {
      throw new Error(`Model "${settings.model}" does not support image input. Select a vision model.`);
    }
  }
  const startedAt = performance.now();
  const response = await requestOllama(settings, "chat", {
    messages,
    ...(tools.length ? { tools } : {}),
    stream: true,
    ...options,
  }, signal);
  if (!response.body) throw new Error("Ollama returned an empty response body.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let firstSeconds: number | undefined;
  const toolCalls: OllamaToolCall[] = [];
  const callIds = new Set<string>();
  try {
    while (true) {
      signal.throwIfAborted();
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      if (done && buffer.trim()) buffer += "\n";
      let newline: number;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        signal.throwIfAborted();
        const chunk = parseChunk(line);
        if (firstSeconds === undefined && (chunk.content || chunk.thinking || chunk.toolCalls.length)) {
          firstSeconds = (performance.now() - startedAt) / 1000;
        }
        if (chunk.thinking) onThinking(chunk.thinking);
        if (chunk.content) onContent(chunk.content);
        for (const call of chunk.toolCalls) {
          if (callIds.has(call.id)) throw new Error("Ollama returned a duplicate tool call ID.");
          callIds.add(call.id);
          toolCalls.push(call);
        }
        signal.throwIfAborted();
        // Tool calls are available only after a successful, complete stream.
        if (chunk.done) return { toolCalls, usage: chunk.usage, firstSeconds, doneReason: chunk.doneReason };
      }
      if (done) break;
    }
    throw new Error("Ollama stream ended before completion.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
