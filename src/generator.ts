import { type Chat, type GeneratorController, type InferParsedConfig } from "@lmstudio/sdk";
import OpenAI from "openai";
import {
  type ChatCompletionMessageParam,
  type ChatCompletionMessageToolCall,
  type ChatCompletionTool,
  type ChatCompletionToolMessageParam,
} from "openai/resources/index";
import { configSchematics, globalConfigSchematics } from "./config";

/* -------------------------------------------------------------------------- */
/*                                   Types                                    */
/* -------------------------------------------------------------------------- */

type ToolCallState = {
  id: string;
  name: string | null;
  index: number;
  arguments: string;
};

/* -------------------------------------------------------------------------- */
/*                               Build helpers                                */
/* -------------------------------------------------------------------------- */

/** Build a pre-configured OpenAI client. */
function createOpenAI(
  globalConfig: InferParsedConfig<typeof globalConfigSchematics>,
  isOpenAIModel: boolean,
) {
  const overrideBaseUrl = globalConfig.get("overrideBaseUrl");

  // Use override URL if provided, otherwise auto-detect based on model type
  const baseURL =
    overrideBaseUrl ||
    (isOpenAIModel ? "https://api.openai.com/v1" : "https://api.anthropic.com/v1");

  const apiKey = isOpenAIModel
    ? globalConfig.get("openaiApiKey")
    : globalConfig.get("anthropicApiKey");

  return new OpenAI({
    apiKey,
    baseURL,
  });
}

/** Convert internal chat history to the format expected by OpenAI. */
function toOpenAIMessages(history: Chat): ChatCompletionMessageParam[] {
  const messages: ChatCompletionMessageParam[] = [];

  for (const message of history) {
    switch (message.getRole()) {
      case "system":
        messages.push({ role: "system", content: message.getText() });
        break;

      case "user":
        messages.push({ role: "user", content: message.getText() });
        break;

      case "assistant": {
        const toolCalls: ChatCompletionMessageToolCall[] = message
          .getToolCallRequests()
          .map(toolCall => ({
            id: toolCall.id ?? "",
            type: "function",
            function: {
              name: toolCall.name,
              arguments: JSON.stringify(toolCall.arguments ?? {}),
            },
          }));

        messages.push({
          role: "assistant",
          content: message.getText(),
          ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        });
        break;
      }

      case "tool": {
        message.getToolCallResults().forEach(toolCallResult => {
          messages.push({
            role: "tool",
            tool_call_id: toolCallResult.toolCallId ?? "",
            content: toolCallResult.content,
          } as ChatCompletionToolMessageParam);
        });
        break;
      }
    }
  }

  return messages;
}

/** Convert LM Studio tool definitions to OpenAI function-tool descriptors. */
function toOpenAITools(ctl: GeneratorController): ChatCompletionTool[] | undefined {
  const tools = ctl.getToolDefinitions().map<ChatCompletionTool>(t => ({
    type: "function",
    function: {
      name: t.function.name,
      description: t.function.description,
      parameters: t.function.parameters ?? {},
    },
  }));
  return tools.length ? tools : undefined;
}

/* -------------------------------------------------------------------------- */
/*                            Stream-handling utils                           */
/* -------------------------------------------------------------------------- */

function wireAbort(ctl: GeneratorController, stream: { controller: AbortController }) {
  ctl.onAborted(() => {
    console.info("Generation aborted by user.");
    stream.controller.abort();
  });
}

async function consumeStream(stream: AsyncIterable<any>, ctl: GeneratorController) {
  let current: ToolCallState | null = null;

  function maybeFlushCurrentToolCall() {
    if (current === null || current.name === null) {
      return;
    }
    ctl.toolCallGenerationEnded({
      type: "function",
      name: current.name,
      arguments: JSON.parse(current.arguments),
      id: current.id,
    });
    current = null;
  }

  for await (const chunk of stream) {
    console.info("Received chunk:", JSON.stringify(chunk));
    const delta = chunk.choices?.[0]?.delta as
      | {
          content?: string;
          tool_calls?: Array<{
            index: number;
            id?: string;
            function?: { name?: string; arguments?: string };
          }>;
        }
      | undefined;

    if (!delta) continue;

    /* Text streaming */
    if (delta.content) {
      ctl.fragmentGenerated(delta.content);
    }

    /* Tool-call streaming */
    for (const toolCall of delta.tool_calls ?? []) {
      if (toolCall.id !== undefined) {
        maybeFlushCurrentToolCall();
        current = { id: toolCall.id, name: null, index: toolCall.index, arguments: "" };
        ctl.toolCallGenerationStarted();
      }

      if (toolCall.function?.name && current) {
        current.name = toolCall.function.name;
        ctl.toolCallGenerationNameReceived(toolCall.function.name);
      }

      if (toolCall.function?.arguments && current) {
        current.arguments += toolCall.function.arguments;
        ctl.toolCallGenerationArgumentFragmentGenerated(toolCall.function.arguments);
      }
    }

    /* Finalize tool call */
    if (chunk.choices?.[0]?.finish_reason === "tool_calls" && current?.name) {
      maybeFlushCurrentToolCall();
    }
  }

  console.info("Generation completed.");
}

/* -------------------------------------------------------------------------- */
/*                                     API                                    */
/* -------------------------------------------------------------------------- */

export async function generate(ctl: GeneratorController, history: Chat) {
  const config = ctl.getPluginConfig(configSchematics);
  const model = config.get("model");

  // Choose the appropriate config fields based on the model
  const isOpenAIModel = model.startsWith("gpt-");
  const globalConfig = ctl.getGlobalPluginConfig(globalConfigSchematics);

  /* 1. Setup client & payload */
  const openai = createOpenAI(globalConfig, isOpenAIModel);
  const messages = toOpenAIMessages(history);
  const tools = toOpenAITools(ctl);

  /* 2. Kick off streaming completion */
  const stream = await openai.chat.completions.create({
    model: model,
    messages,
    tools,
    stream: true,
  });

  /* 3. Abort wiring & stream processing */
  wireAbort(ctl, stream as any);
  await consumeStream(stream as any, ctl);
}
