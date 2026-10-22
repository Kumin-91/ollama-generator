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
