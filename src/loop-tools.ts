import { type LLMTool, type PredictionLoopHandlerController, type PredictionProcessContentBlockController } from "@lmstudio/sdk";
import { type OllamaMessage, type OllamaToolCall } from "./ollama-client";
import { withAbort } from "./abort";

type ToolSession = Awaited<ReturnType<PredictionLoopHandlerController["startToolUseSession"]>>;
type LoopTool = ToolSession["tools"][number];

export async function openToolSession(ctl: PredictionLoopHandlerController): Promise<ToolSession> {
  ctl.abortSignal.throwIfAborted();
  const pending = ctl.startToolUseSession().then(session => {
    if (ctl.abortSignal.aborted) {
      session[Symbol.dispose]();
      ctl.abortSignal.throwIfAborted();
    }
    return session;
  });
  return await withAbort(pending, ctl.abortSignal);
}

export function prepareLoopTools(tools: ToolSession["tools"]) {
  const byName = new Map<string, LoopTool>();
  const definitions: LLMTool[] = [];
  for (const tool of tools) {
    if (byName.has(tool.name)) throw new Error(`Enabled tools have a duplicate name: "${tool.name}".`);
    byName.set(tool.name, tool);
    definitions.push({ type: "function", function: {
      name: tool.name, description: tool.description, parameters: tool.parametersJsonSchema,
    } });
  }
  return { byName, definitions };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function executeLoopTool(
  ctl: PredictionLoopHandlerController,
  block: PredictionProcessContentBlockController,
  call: OllamaToolCall,
  tool: LoopTool | undefined,
  callId: number,
): Promise<OllamaMessage> {
  ctl.abortSignal.throwIfAborted();
  const fn = call.function;
  const request = () => ({
    callId, toolCallRequestId: call.id, name: fn.name,
    parameters: fn.arguments, pluginIdentifier: tool?.pluginIdentifier,
  });
  block.appendToolRequest(request());
  const status = ctl.createToolStatus(callId, { type: "toolCallQueued" });
  const result = (content: string): OllamaMessage => {
    ctl.createContentBlock({ roleOverride: "tool" }).appendToolResult({
      callId, toolCallRequestId: call.id, content,
    });
    return { role: "tool", content, tool_call_id: call.id, tool_name: fn.name };
  };
  const invalidArguments = (error: unknown) => {
    const message = errorMessage(error);
    status.setStatus({ type: "toolCallGenerationFailed", error: message });
    return result(JSON.stringify({ error: message }));
  };
  try {
    try {
      if (!tool) throw new Error(`Unknown tool: "${fn.name}".`);
      tool.checkParameters(fn.arguments);
    } catch (error) {
      return invalidArguments(error);
    }
    status.setStatus({ type: "confirmingToolCall" });
    const confirmation = await ctl.requestConfirmToolCall(request());
    ctl.abortSignal.throwIfAborted();
    if (confirmation.type === "deny") {
      status.setStatus({ type: "toolCallDenied", denyReason: confirmation.denyReason });
      return result(JSON.stringify({ error: confirmation.denyReason ?? "Tool call denied by the user." }));
    }
    if (confirmation.toolArgsOverride !== undefined) {
      try {
        tool.checkParameters(confirmation.toolArgsOverride);
      } catch (error) {
        return invalidArguments(error);
      }
      fn.arguments = confirmation.toolArgsOverride;
      block.replaceToolRequest(request());
    }
    status.setStatus({ type: "callingTool" });
    ctl.abortSignal.throwIfAborted();
    const started = performance.now();
    const value: unknown = await withAbort(Promise.resolve(tool.implementation(fn.arguments, {
      callId, signal: ctl.abortSignal,
      status: text => status.setCustomStatusText(text), warn: text => status.addWarning(text),
    })), ctl.abortSignal);
    ctl.abortSignal.throwIfAborted();
    const content = typeof value === "string" ? value : value === undefined ? "undefined" : JSON.stringify(value);
    if (content === undefined) throw new Error(`Tool "${fn.name}" returned a value that cannot be serialized.`);
    status.setStatus({ type: "toolCallSucceeded", timeMs: Math.round(performance.now() - started) });
    return result(content);
  } catch (error) {
    status.setStatus({ type: "toolCallFailed", error: ctl.abortSignal.aborted ? "Tool call stopped." : errorMessage(error) });
    throw error;
  }
}
