import { type LLMTool } from "@lmstudio/sdk";
import { randomUUID } from "node:crypto";
import { type GenerationOptions, type OllamaSettings } from "./settings";

export const DEFAULT_REQUEST_TIMEOUT_SECONDS = 600;
const MAX_STREAM_LINE_BYTES = 1024 * 1024;
const MAX_JSON_BYTES = 8 * 1024 * 1024;
const MAX_TOOL_CALLS = 128;

export interface OllamaToolCall {
  id: string;
  function: { name: string; arguments: Record<string, unknown> };
}

export interface OllamaMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  images?: string[];
  tool_calls?: OllamaToolCall[];
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
  timeoutMs?: number;
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
    id: value.id || randomUUID(),
    function: { name: value.function.name, arguments: value.function.arguments },
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
    redirect: "manual",
    headers: {
      "Content-Type": "application/json",
      ...(settings.apiKey ? { Authorization: `Bearer ${settings.apiKey}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify({ model: settings.model, ...body }),
    signal,
  });
  if (response.status >= 300 && response.status < 400) {
    void response.body?.cancel().catch(() => {});
    throw new Error("Ollama redirected the request. Enter the final server URL.");
  }
  if (!response.ok) {
    const error: unknown = await readJson(response, signal).catch(() => null);
    signal.throwIfAborted();
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
    const data = await readJson(response, lookupSignal);
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

async function* readBody(response: Response, signal: AbortSignal): AsyncGenerator<Uint8Array> {
  if (!response.body) throw new Error("Ollama returned an empty response body.");
  const reader = response.body.getReader();
  const onAbort = () => { void reader.cancel(signal.reason).catch(() => {}); };
  signal.addEventListener("abort", onAbort, { once: true });
  try {
    while (true) {
      signal.throwIfAborted();
      const { value, done } = await reader.read();
      signal.throwIfAborted();
      if (done) return;
      yield value;
    }
  } finally {
    signal.removeEventListener("abort", onAbort);
    // Start cancellation without waiting for an unresponsive source to acknowledge it.
    void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function readJson(response: Response, signal: AbortSignal): Promise<unknown> {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let text = "";
  let bytes = 0;
  for await (const value of readBody(response, signal)) {
    bytes += value.byteLength;
    if (bytes > MAX_JSON_BYTES) throw new Error("Ollama JSON response exceeds the 8 MiB limit.");
    text += decoder.decode(value, { stream: true });
  }
  return JSON.parse(text + decoder.decode());
}

async function* readStreamLines(response: Response, signal: AbortSignal): AsyncGenerator<string> {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let lineBytes = 0;
  for await (const value of readBody(response, signal)) {
    let start = 0;
    while (start < value.length) {
      const newline = value.indexOf(10, start);
      const end = newline === -1 ? value.length : newline;
      lineBytes += end - start;
      if (lineBytes > MAX_STREAM_LINE_BYTES) {
        throw new Error("Ollama streaming JSON line exceeds the 1 MiB limit.");
      }
      buffer += decoder.decode(value.subarray(start, end), { stream: newline === -1 });
      if (newline === -1) break;
      yield buffer;
      buffer = "";
      lineBytes = 0;
      start = newline + 1;
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) yield buffer;
}

export async function generateOllama({
  settings, messages, tools, options, signal: parentSignal,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_SECONDS * 1000, onThinking, onContent,
}: OllamaGenerationRequest): Promise<OllamaGenerationResult> {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 86_400_000) {
    throw new Error("Ollama request timeout must be between 1 ms and 24 hours.");
  }
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(new DOMException(
    `Ollama request timed out after ${timeoutMs / 1000} seconds.`, "TimeoutError",
  )), timeoutMs);
  timer.unref();
  const signal = AbortSignal.any([parentSignal, timeout.signal]);
  try {
    if (messages.some(message => message.images?.length)) {
      const response = await requestOllama(settings, "show", {}, signal);
      const model = await readJson(response, signal);
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
    let firstSeconds: number | undefined;
    const toolCalls: OllamaToolCall[] = [];
    const callIds = new Set<string>();
    for await (const rawLine of readStreamLines(response, signal)) {
      const line = rawLine.trim();
      if (!line) continue;
      signal.throwIfAborted();
      const chunk = parseChunk(line);
      if (firstSeconds === undefined && (chunk.content || chunk.thinking || chunk.toolCalls.length)) {
        firstSeconds = (performance.now() - startedAt) / 1000;
      }
      if (chunk.thinking) onThinking(chunk.thinking);
      signal.throwIfAborted();
      if (chunk.content) onContent(chunk.content);
      for (const call of chunk.toolCalls) {
        if (toolCalls.length >= MAX_TOOL_CALLS) throw new Error("Ollama exceeded the limit of 128 tool calls per request.");
        if (callIds.has(call.id)) throw new Error("Ollama returned a duplicate tool call ID.");
        callIds.add(call.id);
        toolCalls.push(call);
      }
      signal.throwIfAborted();
      // Tool calls are available only after a successful, complete stream.
      if (chunk.done) return { toolCalls, usage: chunk.usage, firstSeconds, doneReason: chunk.doneReason };
    }
    throw new Error("Ollama stream ended before completion.");
  } catch (error) {
    signal.throwIfAborted();
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
