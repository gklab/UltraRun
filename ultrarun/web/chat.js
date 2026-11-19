"use strict";

const markdown = window.markdownit({ html: false, maxNesting: 20 });
const defaultLink = markdown.renderer.rules.link_open;
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  tokens[index].attrSet("target", "_blank");
  tokens[index].attrSet("rel", "noopener noreferrer");
  return defaultLink ? defaultLink(tokens, index, options, env, renderer)
    : renderer.renderToken(tokens, index, options);
};

const transcript = document.querySelector("#transcript");
const welcome = document.querySelector("#welcome");
const turns = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const prompt = document.querySelector("#prompt");
const limit = document.querySelector("#limit");
const send = document.querySelector("#send");
const sendLabel = document.querySelector("#send-label");
const reset = document.querySelector("#reset");
const notice = document.querySelector("#notice");
const connection = document.querySelector("#connection");
const count = new Intl.NumberFormat("en-US");
let sessionId = crypto.randomUUID();
let history = [];
let busy = false;

async function postChat(body) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
  }
  return response;
}

async function streamChat(messages, maxTokens, onDelta) {
  const response = await postChat({ messages, max_tokens: maxTokens, stream: true });
  if (!response.headers.get("Content-Type")?.startsWith("application/x-ndjson") || !response.body) {
    throw new Error("The server did not return a chat stream.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = done ? "" : lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.done) return event;
        if (event.delta) onDelta(event.delta);
      }
      if (done) throw new Error("The chat stream ended early.");
    }
  } finally {
    void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function scrollToLatest() {
  transcript.scrollTop = transcript.scrollHeight;
}

function makeTurn(role) {
  const container = document.createElement("article");
  container.className = `reply ${role}-turn`;
  const label = document.createElement("h2");
  label.className = "reply-label";
  label.textContent = role === "user" ? "You" : "Qwen";
  const body = document.createElement("div");
  body.className = "reply-body";
  container.append(label, body);
  turns.append(container);
  return { container, body };
}

function renderMarkdown(node, text) {
  const env = {};
  const tokens = markdown.parse(text, env);
  // Raw HTML is disabled in markdown-it. Only its generated Markdown markup is inserted here.
  node.innerHTML = markdown.renderer.render(tokens, markdown.options, env);
  return tokens;
}

function addHtmlPreview(container, text, tokens) {
  const htmlBlocks = tokens.filter((token) => token.type === "fence"
      && /^(html|htm)(\s|$)/i.test(token.info.trim()) && token.content.trim());
  const html = htmlBlocks.map((token) => token.content);
  if (!html.length && /^\s*<(?:!doctype\s+html|html|head|body|main|section|div)\b/i.test(text)) {
    html.push(text);
  }
  html.forEach((source, index) => {
    const preview = document.createElement("details");
    preview.className = "preview";
    const title = document.createElement("summary");
    title.textContent = html.length > 1 ? `HTML preview ${index + 1}` : "HTML preview";
    preview.append(title);
    preview.addEventListener("toggle", () => {
      if (!preview.open || preview.querySelector("iframe")) return;
      const frame = document.createElement("iframe");
      frame.title = title.textContent;
      frame.setAttribute("sandbox", "");
      frame.referrerPolicy = "no-referrer";
      frame.srcdoc = '<!doctype html><meta charset="utf-8">'
        + '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        + 'style-src \'unsafe-inline\'; img-src data:; font-src data:; base-uri \'none\'; form-action \'none\'">'
        + source;
      preview.append(frame);
    });
    container.append(preview);
  });
}

function showMetrics(container, metrics) {
  const summary = document.createElement("dl");
  summary.className = "metrics";
  summary.setAttribute("aria-label", "Reply performance");
  const entries = [
    ["Decode", `${metrics.decode_tok_s.toFixed(1)} tok/s`],
    ["First token", `${metrics.ttft_s.toFixed(2)} s`],
    ["Cached input", `${count.format(metrics.cached_prompt_tokens)} / ${count.format(metrics.input_tokens)}`],
    ["Output tokens", count.format(metrics.output_tokens)],
    ["Verification rounds", count.format(metrics.target_decode_passes)],
    ["Draft schedule", metrics.draft_schedule],
  ];
  for (const [label, value] of entries) {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    item.append(term, detail);
    summary.append(item);
  }
  container.append(summary);
}

function setBusy(activity = "") {
  busy = Boolean(activity);
  send.disabled = reset.disabled = busy;
  sendLabel.textContent = activity || "Send";
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = prompt.value.trim();
  if (busy || !input || !composer.reportValidity()) return;
  const maxTokens = limit.valueAsNumber;
  const user = { role: "user", content: input };
  setBusy("Generating…");
  notice.textContent = "";
  prompt.value = "";
  welcome.hidden = true;
  makeTurn("user").body.textContent = input;
  const reply = makeTurn("assistant");
  reply.body.textContent = "Generating…";
  reply.body.classList.add("pending");
  scrollToLatest();

  let draft = "";
  let timer;
  const paint = () => {
    const follow = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80;
    const tokens = renderMarkdown(reply.body, draft);
    reply.body.classList.remove("pending");
    if (follow) scrollToLatest();
    return tokens;
  };
  let result;
  try {
    result = await streamChat([...history, user], maxTokens, (delta) => {
      draft += delta;
      if (timer === undefined) {
        timer = setTimeout(() => {
          timer = undefined;
          paint();
        }, 80);
      }
    });
    clearTimeout(timer);
    draft = result.text;
    const tokens = paint();
    const follow = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80;
    addHtmlPreview(reply.container, draft, tokens);
    showMetrics(reply.container, result.metrics);
    history.push(user, { role: "assistant", content: draft });
    if (follow) scrollToLatest();
  } catch (error) {
    clearTimeout(timer);
    if (draft) paint();
    else reply.body.textContent = "This turn could not be completed.";
    reply.body.classList.remove("pending");
    if (!draft) reply.body.classList.add("problem");
    notice.textContent = error.message;
    if (!prompt.value && !result) prompt.value = input;
  } finally {
    setBusy();
  }
});

prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!busy) composer.requestSubmit();
  }
});

for (const button of document.querySelectorAll(".suggestion")) {
  button.addEventListener("click", () => {
    prompt.value = button.dataset.prompt;
    prompt.focus();
  });
}

reset.addEventListener("click", async () => {
  if (busy) return;
  setBusy("Resetting…");
  notice.textContent = "";
  try {
    await postChat({ reset: true });
    sessionId = crypto.randomUUID();
    history = [];
    turns.replaceChildren();
    welcome.hidden = false;
    prompt.value = "";
    prompt.focus();
  } catch (error) {
    notice.textContent = error.message;
  } finally {
    setBusy();
  }
});

async function checkConnection() {
  try {
    const response = await fetch("/healthz");
    if (!response.ok) throw new Error();
    const health = await response.json();
    if (!health.ready) throw new Error();
    document.querySelector("#model").textContent = health.model.split("/").pop();
    connection.textContent = `Local · ${health.dtype}`;
    connection.dataset.ready = "true";
  } catch {
    connection.textContent = "Server offline";
    connection.dataset.ready = "false";
  }
}

void checkConnection();
