import { type LLMTool, type ToolCallRequest } from "@lmstudio/sdk";
import { randomUUID } from "node:crypto";
import { type GenerationOptions } from "./generation-options";
import { type OllamaSettings } from "./ollama-settings";

export interface OllamaMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: { id: string; function: { name: string; arguments: Record<string, unknown> } }[];
  tool_name?: string;
  tool_call_id?: string;
}

interface OllamaChunk {
  done: boolean;
  content: string;
  thinking: string;
  toolCalls: ToolCallRequest[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseToolCall(value: unknown): ToolCallRequest {
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
    content: message.content ?? "",
    thinking: message.thinking ?? "",
    toolCalls: (message.tool_calls ?? []).map(parseToolCall),
  };
}

export async function* streamOllamaChat(
  settings: OllamaSettings,
  messages: OllamaMessage[],
  tools: LLMTool[],
  overrides: GenerationOptions,
  signal: AbortSignal,
): AsyncGenerator<OllamaChunk> {
  signal.throwIfAborted();
  const response = await fetch(`${settings.serverUrl}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(settings.apiKey ? { Authorization: `Bearer ${settings.apiKey}` } : {}),
    },
    body: JSON.stringify({
      model: settings.model,
      messages,
      ...(tools.length ? { tools } : {}),
      stream: true,
      ...overrides,
    }),
    signal,
  });
  if (!response.ok) {
    const error: unknown = await response.json().catch(() => null);
    throw new Error(isRecord(error) && typeof error.error === "string"
      ? `Ollama (${response.status}): ${error.error}`
      : `Ollama request failed (HTTP ${response.status}).`);
  }
  if (!response.body) throw new Error("Ollama returned an empty response body.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
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
        yield chunk;
        if (chunk.done) return;
      }
      if (done) break;
    }
    throw new Error("Ollama stream ended before completion.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
