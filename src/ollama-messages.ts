import { type Chat, type ChatMessage, type LMStudioClient } from "@lmstudio/sdk";
import { readFile } from "node:fs/promises";
import { type OllamaMessage } from "./ollama-client";
import { withAbort } from "./abort";

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
      : (await readFile(await withAbort(file.getFilePath(), signal), { signal })).toString("base64");
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
  const resultIds = new Set<string>();
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
          if (resultIds.has(id)) throw new Error("Tool call history contains a duplicate result.");
          resultIds.add(id);
          messages.push({ role, content: result.content, tool_call_id: id, tool_name: name });
        }
        break;
    }
  }
  return messages;
}

const legacyFooterPattern = /\n\n<!-- ollama-generator:stats:v1:start -->\n\n---\n\n(?:Input|Output|Context):[^\n]*(?:  \n(?:Input|Output|Context):[^\n]*)*\n\n<!-- ollama-generator:stats:v1:end -->\s*$/;
const timeValue = "(?:N/A|\\d+\\.\\d{2}s)";
const footerSeparator = "\n\n---\n";
const footerLines = [
  /^\*\*Token Stats:\*\* (?:N\/A|\d+(?:\.\d+)?(?:e\+\d+)?) tok\/sec \| (?:N\/A|\d[\d,]*) tokens$/,
  new RegExp(`^\\*\\*Time:\\*\\* Processing:? ${timeValue} \\| First:? ${timeValue} \\| Total:? ${timeValue}$`),
  /^\*\*Context:\*\* (?:N\/A|\d[\d,]*) \/ (?:N\/A|\d[\d,]*) tokens \((?:N\/A|\d+\.\d{2}%)\)$/,
];

function hasOpenCodeFence(content: string): boolean {
  let fence = "";
  for (const line of content.split("\n")) {
    const match = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (!match) continue;
    if (!fence) fence = match[1];
    else if (match[1][0] === fence[0] && match[1].length >= fence.length && !match[2].trim()) fence = "";
  }
  return !!fence;
}

// Remove statistics from older conversations without stripping code samples or ordinary prose.
export function stripStatsFooter(content: string): string {
  const legacyStart = content.search(legacyFooterPattern);
  const text = legacyStart !== -1 && !hasOpenCodeFence(content.slice(0, legacyStart))
    ? content.slice(0, legacyStart) : content;
  const start = text.lastIndexOf(footerSeparator);
  if (start === -1 || hasOpenCodeFence(text.slice(0, start))) return text;
  const lines = text.slice(start + footerSeparator.length).trimEnd().split(/\r?\n/).map(line => line.trimEnd());
  if (!lines.length || lines.length > footerLines.length) return text;
  let previous = -1;
  for (const line of lines) {
    const index = footerLines.findIndex(pattern => pattern.test(line));
    if (index <= previous) return text;
    previous = index;
  }
  return text.slice(0, start);
}
