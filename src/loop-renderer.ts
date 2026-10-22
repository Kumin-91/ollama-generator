import {
  type LLMGenInfo, type LLMPredictionStopReason,
  type PredictionLoopHandlerController, type PredictionProcessContentBlockController,
} from "@lmstudio/sdk";
import { type OllamaGenerationResult } from "./ollama-client";

export function createLoopRenderer(ctl: PredictionLoopHandlerController, model: string, contextUsage: boolean) {
  let block: PredictionProcessContentBlockController | undefined;
  let lastBlock: PredictionProcessContentBlockController | undefined;
  let thinking = false;
  let content = "";
  let reasoning = "";
  const finishThinking = () => {
    if (thinking) {
      block?.setStyle({ type: "thinking", ended: true });
      block = undefined;
    }
    thinking = false;
  };
  const answerBlock = () => {
    if (!block || thinking) {
      finishThinking();
      block = ctl.createContentBlock();
      lastBlock = block;
    }
    return block;
  };
  return {
    onThinking(text: string) {
      if (!thinking || !block) {
        block = ctl.createContentBlock({
          includeInContext: false, style: { type: "thinking", ended: false },
        });
        lastBlock = block;
        thinking = true;
      }
      block.appendText(text);
      reasoning += text;
    },
    onContent(text: string) {
      answerBlock().appendText(text);
      content += text;
    },
    finish: finishThinking,
    toolBlock: answerBlock,
    attachGenInfo(result: OllamaGenerationResult, contextLength?: number, reason?: LLMPredictionStopReason) {
      const info = nativeGenerationInfo(model, result, contextUsage, contextLength, reason);
      lastBlock?.attachGenInfo(info);
      if (contextUsage && !result.toolCalls.length && !reason) {
        const used = info.stats.totalTokensCount;
        const count = (value: number | undefined) => value?.toLocaleString("en-US") ?? "N/A";
        const percentage = used !== undefined && contextLength !== undefined
          ? `${(used / contextLength * 100).toFixed(2)}%` : "N/A";
        ctl.createContentBlock({ includeInContext: false }).appendText(
          `---\n\n**Context:** ${count(used)} / ${count(contextLength)} tokens (${percentage})`,
          { tokensCount: 0 },
        );
      }
    },
    message() {
      return { content, ...(reasoning ? { thinking: reasoning } : {}) };
    },
  };
}

function stopReason(result: OllamaGenerationResult): LLMPredictionStopReason {
  if (result.toolCalls.length) return "toolCalls";
  if (result.doneReason === "length") return "maxPredictedTokensReached";
  // Ollama's normal `stop` does not distinguish EOS from a stop string.
  // The SDK requires a reason, so completed responses use its EOS category.
  return "eosFound";
}

export function nativeGenerationInfo(
  model: string, result: OllamaGenerationResult, contextUsage: boolean,
  contextLength?: number, reason?: LLMPredictionStopReason,
): LLMGenInfo {
  const { usage } = result;
  const totalTokens = usage?.promptTokens !== undefined && usage.outputTokens !== undefined
    && Number.isSafeInteger(usage.promptTokens + usage.outputTokens)
    ? usage.promptTokens + usage.outputTokens : undefined;
  return {
    indexedModelIdentifier: model,
    identifier: model,
    loadModelConfig: { fields: !contextUsage || contextLength === undefined ? [] : [
      { key: "llm.load.contextLength", value: contextLength },
    ] },
    predictionConfig: { fields: [] },
    stats: {
      stopReason: reason ?? stopReason(result),
      tokensPerSecond: usage?.outputTokens !== undefined && usage.outputSeconds !== undefined
        && usage.outputSeconds > 0 ? usage.outputTokens / usage.outputSeconds : undefined,
      predictedTokensCount: usage?.outputTokens,
      timeToFirstTokenSec: result.firstSeconds,
      ...(contextUsage ? {
        promptTokensCount: usage?.promptTokens,
        totalTokensCount: totalTokens,
      } : {}),
    },
  };
}
