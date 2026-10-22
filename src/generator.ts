import { randomUUID } from "node:crypto";
import { type Chat, type GeneratorController, type ToolCallRequest } from "@lmstudio/sdk";
import OpenAI from "openai";
import {
  type ChatCompletionChunk,
  type ChatCompletionMessageParam,
  type ChatCompletionMessageToolCall,
  type ChatCompletionTool,
} from "openai/resources/chat/completions";
import { configSchematics, globalConfigSchematics } from "./config";
import { resolveOllamaSettings } from "./ollama-settings";

type ToolCallState = {
  id?: string;
  name?: string;
  arguments: string;
};

function requireToolCallId(id: string | undefined): string {
  if (!id) throw new Error("Tool call history is missing a call ID.");
  return id;
}

function toOpenAIMessages(history: Chat): ChatCompletionMessageParam[] {
  const messages: ChatCompletionMessageParam[] = [];

  for (const message of history) {
    const role = message.getRole();
    switch (role) {
      case "system":
      case "user":
        messages.push({ role, content: message.getText() });
        break;
      case "assistant": {
        const toolCalls = message.getToolCallRequests().map<ChatCompletionMessageToolCall>(call => ({
          id: requireToolCallId(call.id),
          type: "function",
          function: { name: call.name, arguments: JSON.stringify(call.arguments ?? {}) },
        }));
        messages.push({
          role,
          content: message.getText(),
          ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        });
        break;
      }
      case "tool":
        for (const result of message.getToolCallResults()) {
          messages.push({ role, tool_call_id: requireToolCallId(result.toolCallId), content: result.content });
        }
        break;
    }
  }
  return messages;
}

function parseToolCall(call: ToolCallState): ToolCallRequest {
  if (!call.name) throw new Error("Tool call is missing a function name.");

  let args: unknown;
  try {
    args = JSON.parse(call.arguments || "{}");
  } catch {
    throw new Error("Tool call arguments must be valid JSON.");
  }
  if (typeof args !== "object" || args === null || Array.isArray(args)) {
    throw new Error("Tool call arguments must be a JSON object.");
  }
  return {
    type: "function",
    id: call.id || randomUUID(),
    name: call.name,
    arguments: args as Record<string, unknown>,
  };
}

async function consumeStream(stream: AsyncIterable<ChatCompletionChunk>, ctl: GeneratorController) {
  const pending = new Map<number, ToolCallState>();

  for await (const chunk of stream) {
    ctl.abortSignal.throwIfAborted();
    const choice = chunk.choices[0];
    if (!choice) continue;

    if (choice.delta.content) ctl.fragmentGenerated(choice.delta.content);

    for (const delta of choice.delta.tool_calls ?? []) {
      const call = pending.get(delta.index) ?? { arguments: "" };
      if (delta.id) call.id = delta.id;
      if (delta.function?.name) call.name = delta.function.name;
      if (delta.function?.arguments) call.arguments += delta.function.arguments;
      pending.set(delta.index, call);
    }

    if (choice.finish_reason === "tool_calls") {
      if (!pending.size) throw new Error("Server finished tool generation without a tool call.");
      // Validate every call before reporting them sequentially to LM Studio.
      const calls = [...pending].sort(([a], [b]) => a - b).map(([, call]) => parseToolCall(call));
      for (const call of calls) {
        ctl.abortSignal.throwIfAborted();
        ctl.toolCallGenerationStarted({ toolCallId: call.id });
        ctl.toolCallGenerationNameReceived(call.name);
        ctl.toolCallGenerationArgumentFragmentGenerated(JSON.stringify(call.arguments));
        ctl.toolCallGenerationEnded(call);
      }
      pending.clear();
    }
  }

  ctl.abortSignal.throwIfAborted();
  if (pending.size) throw new Error("Tool call stream ended before completion.");
}

export async function generate(ctl: GeneratorController, history: Chat) {
  const config = ctl.getPluginConfig(configSchematics);
  const globalConfig = ctl.getGlobalPluginConfig(globalConfigSchematics);
  const settings = resolveOllamaSettings({
    serverUrl: globalConfig.get("serverUrl"),
    apiKey: globalConfig.get("apiKey"),
    model: config.get("model"),
  });
  const client = new OpenAI({
    baseURL: `${settings.serverUrl}/v1`,
    apiKey: settings.apiKey ?? "ollama",
    defaultHeaders: settings.apiKey ? undefined : { Authorization: null },
    maxRetries: 0,
  });
  const tools: ChatCompletionTool[] = ctl.getToolDefinitions().map(tool => ({
    type: "function",
    function: {
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters ?? {},
    },
  }));
  const stream = await client.chat.completions.create({
    model: settings.model,
    messages: toOpenAIMessages(history),
    tools: tools.length ? tools : undefined,
    stream: true,
  }, { signal: ctl.abortSignal });

  await consumeStream(stream, ctl);
}
