import { type Chat, type ChatMessage, type LMStudioClient } from "@lmstudio/sdk";
import { readFile } from "node:fs/promises";
import { stripStatsFooter } from "./history-stats";
import { type OllamaMessage } from "./ollama-client";

function requireToolCallId(id: string | undefined): string {
  if (!id) throw new Error("Tool call history is missing a call ID.");
  return id;
}

async function readImages(
  message: ChatMessage, client: LMStudioClient, signal: AbortSignal,
): Promise<string[]> {
  if (!message.hasFiles()) return [];
  const files = message.getFiles(client);
  if (files.some(file => !file.isImage())) {
    throw new Error("Only image attachments are supported by this generator.");
  }
  if (message.getRole() !== "user") {
    throw new Error("Image attachments are only supported in user messages.");
  }
  const images: string[] = [];
  for (const file of files) {
    signal.throwIfAborted();
    // SDK 1.4 cannot resolve a file path for inline Base64 handles.
    const image = file.identifier.startsWith("base64:")
      ? file.identifier.slice("base64:".length)
      : (await readFile(await file.getFilePath(), { signal })).toString("base64");
    signal.throwIfAborted();
    if (!image) throw new Error(`Image attachment "${file.name}" is empty.`);
    images.push(image);
  }
  return images;
}

export async function toOllamaMessages(
  history: Chat, client: LMStudioClient, signal: AbortSignal,
): Promise<OllamaMessage[]> {
  const messages: OllamaMessage[] = [];
  const toolNames = new Map<string, string>();
  for (const message of history) {
    signal.throwIfAborted();
    const images = await readImages(message, client, signal);
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
          content: stripStatsFooter(message.getText()),
          ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        });
        break;
      }
      case "tool":
        for (const result of message.getToolCallResults()) {
          const id = requireToolCallId(result.toolCallId);
          const name = toolNames.get(id);
          if (!name) throw new Error("Tool result history is missing its matching tool call.");
          messages.push({ role, content: result.content, tool_call_id: id, tool_name: name });
        }
        break;
    }
  }
  return messages;
}
