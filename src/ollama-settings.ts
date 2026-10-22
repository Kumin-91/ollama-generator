export interface OllamaSettingsInput {
  serverUrl: string;
  apiKey: string;
  model: string;
}

export interface OllamaSettings {
  serverUrl: string;
  apiKey: string | undefined;
  model: string;
}

export function resolveOllamaSettings(input: OllamaSettingsInput): OllamaSettings {
  const inputUrl = input.serverUrl.trim();
  if (!inputUrl) {
    throw new Error("Enter an Ollama Server URL.");
  }
  if (!/^https?:\/\//i.test(inputUrl)) {
    throw new Error("Ollama Server URL must use HTTP or HTTPS.");
  }

  let url: URL;
  try {
    url = new URL(inputUrl);
  } catch {
    throw new Error("Enter a valid HTTP or HTTPS Ollama Server URL.");
  }

  if (url.username || url.password || url.search || url.hash) {
    throw new Error("Ollama Server URL must not include credentials, a query, or a fragment.");
  }

  const path = url.pathname.replace(/\/+$/, "");
  if (/\/(?:api\/(?:chat|generate|tags|ps|show)|v1\/(?:chat\/completions|completions|models|responses|embeddings))$/.test(path)) {
    throw new Error("Use the Ollama Server URL without an endpoint path.");
  }

  // Strip API suffixes while preserving proxy paths.
  url.pathname = path.replace(/\/(?:api|v1)$/, "");
  const serverUrl = url.toString().replace(/\/+$/, "");
  const model = input.model.trim();
  if (!model || /\s/.test(model)) {
    throw new Error("Enter a Model name without whitespace.");
  }

  const apiKey = input.apiKey.trim();
  if (/[\r\n]/.test(apiKey)) {
    throw new Error("API Key must not contain line breaks.");
  }

  return {
    serverUrl,
    apiKey: apiKey || undefined,
    model,
  };
}
