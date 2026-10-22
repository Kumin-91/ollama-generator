import { type PredictionLoopHandlerController } from "@lmstudio/sdk";
import { randomUUID } from "node:crypto";
import { owner, name } from "../manifest.json";
import { resolveGenerationConfig } from "./config";
import { createLoopRenderer } from "./loop-renderer";
import { executeLoopTool, openToolSession, prepareLoopTools, withAbort } from "./loop-tools";
import { generateOllama, getOllamaContextLength, type OllamaGenerationResult, type OllamaMessage } from "./ollama-client";
import { toOllamaMessages } from "./ollama-messages";

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
    const { settings, options, contextUsage } = resolveGenerationConfig(ctl);
    const history = await withAbort(ctl.pullHistory(), ctl.abortSignal);
    const messages = await toOllamaMessages(history, ctl.client, ctl.abortSignal);
    const callIds = new Set(messages.flatMap(message => message.tool_calls?.map(call => call.id) ?? []));
    session = await openToolSession(ctl);
    const tools = prepareLoopTools(session.tools);
    let callId = 0;
    await ctl.setSenderName(settings.model);
    while (true) {
      ctl.abortSignal.throwIfAborted();
      const renderer = createLoopRenderer(ctl, settings.model, contextUsage);
      let result: OllamaGenerationResult;
      try {
        result = await generateOllama({
          settings, messages, tools: tools.definitions, options, signal: ctl.abortSignal,
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
      const results: OllamaMessage[] = [];
      const toolBlock = renderer.toolBlock();
      renderer.attachGenInfo(result);
      for (const call of result.toolCalls) {
        // Preserve unambiguous saved history when a server reuses IDs between rounds.
        if (callIds.has(call.id)) call.id = randomUUID();
        callIds.add(call.id);
        results.push(await executeLoopTool(ctl, toolBlock, call, tools.byName.get(call.name), callId++));
      }
      messages.push({
        role: "assistant", ...renderer.message(),
        tool_calls: result.toolCalls.map(call => ({
          id: call.id, function: { name: call.name, arguments: call.arguments },
        })),
      }, ...results);
    }
    // A title suggestion must not turn a completed response into a failure.
    try {
      if (await withAbort(ctl.needsNaming(), ctl.abortSignal)) {
        const title = history.getMessagesArray().find(message => message.getRole() === "user")
          ?.getText().trim().replace(/\s+/g, " ").slice(0, 60);
        if (title) await withAbort(ctl.suggestName(title), ctl.abortSignal);
      }
    } catch {}
  } catch (error) {
    if (!ctl.abortSignal.aborted) throw error;
  } finally {
    session?.[Symbol.dispose]();
  }
}
