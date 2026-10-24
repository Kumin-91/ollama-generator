import { type Chat, type GeneratorController, type PredictionLoopHandlerController } from "@lmstudio/sdk";
import { randomUUID } from "node:crypto";
import { owner, name } from "../manifest.json";
import { resolveGenerationConfig } from "./config";
import { createLoopRenderer } from "./loop-renderer";
import { executeLoopTool, openToolSession, prepareLoopTools } from "./loop-tools";
import { withAbort } from "./abort";
import { generateOllama, getOllamaContextLength, type OllamaGenerationResult } from "./ollama-client";
import { toOllamaMessages } from "./ollama-messages";

export async function generate(ctl: GeneratorController, history: Chat) {
  if (ctl.abortSignal.aborted) return;
  try {
    const { settings, options, timeoutMs } = resolveGenerationConfig(ctl);
    const { toolCalls } = await generateOllama({
      settings,
      messages: await toOllamaMessages(history, ctl.client, ctl.abortSignal),
      tools: ctl.getToolDefinitions(),
      options, timeoutMs,
      signal: ctl.abortSignal,
      onThinking: text => ctl.fragmentGenerated(text, { reasoningType: "reasoning" }),
      onContent: text => ctl.fragmentGenerated(text),
    });
    // Report validated tool calls only after the stream completes successfully.
    for (const call of toolCalls) {
      ctl.abortSignal.throwIfAborted();
      ctl.toolCallGenerationStarted({ toolCallId: call.id });
      const { name, arguments: args } = call.function;
      ctl.toolCallGenerationNameReceived(name);
      ctl.toolCallGenerationArgumentFragmentGenerated(JSON.stringify(args));
      ctl.toolCallGenerationEnded({ type: "function", id: call.id, name, arguments: args });
    }
  } catch (error) {
    // Finish normally on user cancellation so partial output stays without a failure banner.
    if (!ctl.abortSignal.aborted) throw error;
  }
}

const identifier = `${owner}/${name}`;
const generatorIdentifiers = new Set([identifier, `dev/${identifier}`]);

export async function predict(ctl: PredictionLoopHandlerController) {
  if (ctl.abortSignal.aborted) return;
  let session: Awaited<ReturnType<typeof openToolSession>> | undefined;
  try {
    const source = await withAbort(ctl.tokenSource(), ctl.abortSignal);
    if ("modelKey" in source || !generatorIdentifiers.has(source.identifier)) {
      throw new Error("The selected model does not match Ollama Generator. Select Ollama Generator again.");
    }
    const { settings, options, contextUsage, timeoutMs, maxToolRounds } = resolveGenerationConfig(ctl);
    const history = await withAbort(ctl.pullHistory(), ctl.abortSignal);
    const messages = await toOllamaMessages(history, ctl.client, ctl.abortSignal);
    const callIds = new Set(messages.flatMap(message => message.tool_calls?.map(call => call.id) ?? []));
    session = await openToolSession(ctl);
    const tools = prepareLoopTools(session.tools);
    let callId = 0;
    let toolRounds = 0;
    await ctl.setSenderName(settings.model);
    while (true) {
      ctl.abortSignal.throwIfAborted();
      const renderer = createLoopRenderer(ctl, settings.model, contextUsage);
      let result: OllamaGenerationResult;
      try {
        result = await generateOllama({
          settings, messages, tools: tools.definitions, options, timeoutMs, signal: ctl.abortSignal,
          onThinking: renderer.onThinking, onContent: renderer.onContent,
        });
      } catch (error) {
        renderer.attachGenInfo({ toolCalls: [] }, undefined, ctl.abortSignal.aborted ? "userStopped" : "failed");
        throw error;
      } finally {
        renderer.finish();
      }
      if (!result.toolCalls.length) {
        const contextLength = contextUsage
          ? await getOllamaContextLength(settings, ctl.abortSignal) : undefined;
        ctl.abortSignal.throwIfAborted();
        renderer.attachGenInfo(result, contextLength);
        break;
      }
      if (toolRounds >= maxToolRounds) {
        renderer.attachGenInfo(result, undefined, "failed");
        throw new Error(`Ollama exceeded the limit of ${maxToolRounds} tool rounds. Start a new response to continue.`);
      }
      toolRounds++;
      const toolBlock = renderer.toolBlock();
      renderer.attachGenInfo(result);
      messages.push({ role: "assistant", ...renderer.message(), tool_calls: result.toolCalls });
      for (const call of result.toolCalls) {
        // Preserve unambiguous saved history when a server reuses IDs between rounds.
        if (callIds.has(call.id)) call.id = randomUUID();
        callIds.add(call.id);
        messages.push(await executeLoopTool(ctl, toolBlock, call, tools.byName.get(call.function.name), callId++));
      }
    }
    // A title suggestion must not turn a completed response into a failure.
    try {
      const namingSignal = AbortSignal.any([ctl.abortSignal, AbortSignal.timeout(3000)]);
      if (await withAbort(ctl.needsNaming(), namingSignal)) {
        const title = history.getMessagesArray().find(message => message.getRole() === "user")
          ?.getText().trim().replace(/\s+/g, " ").slice(0, 60);
        namingSignal.throwIfAborted();
        if (title) await withAbort(ctl.suggestName(title), namingSignal);
      }
    } catch {}
  } catch (error) {
    if (!ctl.abortSignal.aborted) throw error;
  } finally {
    session?.[Symbol.dispose]();
  }
}
