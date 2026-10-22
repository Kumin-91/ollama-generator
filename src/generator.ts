import { type Chat, type GeneratorController, type ToolCallRequest } from "@lmstudio/sdk";
import { configSchematics, globalConfigSchematics } from "./config";
import { resolveGenerationOptions } from "./generation-options";
import { type OllamaMessage, streamOllamaChat } from "./ollama-client";
import { resolveOllamaSettings } from "./ollama-settings";

function requireToolCallId(id: string | undefined): string {
  if (!id) throw new Error("Tool call history is missing a call ID.");
  return id;
}

function toOllamaMessages(history: Chat): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  const toolNames = new Map<string, string>();
  for (const message of history) {
    if (message.hasFiles()) {
      throw new Error("File attachments are not supported by this generator yet.");
    }
    const role = message.getRole();
    switch (role) {
      case "system":
      case "user":
        messages.push({ role, content: message.getText() });
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
    const stream = streamOllamaChat(settings, toOllamaMessages(history), ctl.getToolDefinitions(),
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
