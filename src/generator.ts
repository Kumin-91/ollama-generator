import { type Chat, type ChatMessage, type GeneratorController, type ToolCallRequest } from "@lmstudio/sdk";
import { readFile } from "node:fs/promises";
import { configSchematics, globalConfigSchematics } from "./config";
import { resolveGenerationOptions } from "./generation-options";
import { type OllamaMessage, streamOllamaChat } from "./ollama-client";
import { resolveOllamaSettings } from "./ollama-settings";

function requireToolCallId(id: string | undefined): string {
  if (!id) throw new Error("Tool call history is missing a call ID.");
  return id;
}

async function readImages(message: ChatMessage, ctl: GeneratorController): Promise<string[]> {
  if (!message.hasFiles()) return [];
  const files = message.getFiles(ctl.client);
  if (files.some(file => !file.isImage())) {
    throw new Error("Only image attachments are supported by this generator.");
  }
  if (message.getRole() !== "user") {
    throw new Error("Image attachments are only supported in user messages.");
  }
  const images: string[] = [];
  for (const file of files) {
    ctl.abortSignal.throwIfAborted();
    // SDK 1.4 cannot resolve a file path for inline Base64 handles.
    const image = file.identifier.startsWith("base64:")
      ? file.identifier.slice("base64:".length)
      : (await readFile(await file.getFilePath(), { signal: ctl.abortSignal })).toString("base64");
    ctl.abortSignal.throwIfAborted();
    if (!image) throw new Error(`Image attachment "${file.name}" is empty.`);
    images.push(image);
  }
  return images;
}

async function toOllamaMessages(history: Chat, ctl: GeneratorController): Promise<OllamaMessage[]> {
  const messages: OllamaMessage[] = [];
  const toolNames = new Map<string, string>();
  for (const message of history) {
    ctl.abortSignal.throwIfAborted();
    const images = await readImages(message, ctl);
    const role = message.getRole();
    switch (role) {
      case "system":
      case "user":
        messages.push({ role, content: message.getText(), ...(images.length ? { images } : {}) });
        break;
      case "assistant": {
        const toolCalls = message.getToolCallRequests().map(call => {
          const id = requireToolCallId(call.id);
          if (toolNames.has(id)) throw new Error("Tool call history contains a duplicate call ID.");
          toolNames.set(id, call.name);
          return { id, function: { name: call.name, arguments: call.arguments ?? {} } };
        });
        messages.push({
          role,
          content: message.getText(),
          ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        });
        break;
      }
      case "tool":
        for (const result of message.getToolCallResults()) {
          const id = requireToolCallId(result.toolCallId);
          const name = toolNames.get(id);
          if (!name) throw new Error("Tool result history is missing its matching tool call.");
          messages.push({
            role, content: result.content, tool_call_id: id, tool_name: name,
          });
        }
        break;
    }
  }
  return messages;
}

export async function generate(ctl: GeneratorController, history: Chat) {
  if (ctl.abortSignal.aborted) return;
  try {
    const config = ctl.getPluginConfig(configSchematics);
    const globalConfig = ctl.getGlobalPluginConfig(globalConfigSchematics);
    const settings = resolveOllamaSettings({
      serverUrl: globalConfig.get("serverUrl"),
      apiKey: globalConfig.get("apiKey"),
      model: config.get("model"),
    });
    const stream = streamOllamaChat(settings, await toOllamaMessages(history, ctl), ctl.getToolDefinitions(),
      resolveGenerationOptions(globalConfig), ctl.abortSignal);
    const calls: ToolCallRequest[] = [];
    const callIds = new Set<string>();
    for await (const chunk of stream) {
      ctl.abortSignal.throwIfAborted();
      if (chunk.thinking) ctl.fragmentGenerated(chunk.thinking, { reasoningType: "reasoning" });
      if (chunk.content) ctl.fragmentGenerated(chunk.content);
      for (const call of chunk.toolCalls) {
        const id = requireToolCallId(call.id);
        if (callIds.has(id)) throw new Error("Ollama returned a duplicate tool call ID.");
        callIds.add(id);
        calls.push(call);
      }
    }
    ctl.abortSignal.throwIfAborted();
    // Report validated tool calls only after the stream completes successfully.
    for (const call of calls) {
      ctl.abortSignal.throwIfAborted();
      ctl.toolCallGenerationStarted({ toolCallId: call.id });
      ctl.toolCallGenerationNameReceived(call.name);
      ctl.toolCallGenerationArgumentFragmentGenerated(JSON.stringify(call.arguments));
      ctl.toolCallGenerationEnded(call);
    }
  } catch (error) {
    // Finish normally on user cancellation so partial output stays without a failure banner.
    if (!ctl.abortSignal.aborted) throw error;
  }
}
