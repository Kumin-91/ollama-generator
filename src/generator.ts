import { type Chat, type GeneratorController } from "@lmstudio/sdk";
import { resolveGenerationConfig } from "./config";
import { generateOllama } from "./ollama-client";
import { toOllamaMessages } from "./ollama-messages";

export async function generate(ctl: GeneratorController, history: Chat) {
  if (ctl.abortSignal.aborted) return;
  try {
    const { settings, options } = resolveGenerationConfig(ctl);
    const { toolCalls } = await generateOllama({
      settings,
      messages: await toOllamaMessages(history, ctl.client, ctl.abortSignal),
      tools: ctl.getToolDefinitions(),
      options,
      signal: ctl.abortSignal,
      onThinking: text => ctl.fragmentGenerated(text, { reasoningType: "reasoning" }),
      onContent: text => ctl.fragmentGenerated(text),
    });
    // Report validated tool calls only after the stream completes successfully.
    for (const call of toolCalls) {
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
