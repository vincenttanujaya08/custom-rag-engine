const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const debugContent = document.getElementById("debug-content");
const statusBadge = document.getElementById("status-badge");

let activeBotDiv = null;
let abortController = null;

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `max-w-[80%] rounded-xl px-4 py-2 text-sm leading-relaxed whitespace-pre-wrap msg-${role}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function updateStatus(state) {
  const colors = { connecting: "bg-yellow-600", online: "bg-green-600", offline: "bg-red-600", streaming: "bg-blue-600" };
  statusBadge.textContent = state;
  statusBadge.className = `text-xs ${colors[state] || "bg-gray-600"} px-2 py-0.5 rounded-full`;
}

function setFormEnabled(enabled) {
  queryInput.disabled = !enabled;
  chatForm.querySelector("button").disabled = !enabled;
}

function renderDebug(chunks, tokenCount) {
  let html = `<div class="text-xs text-gray-400 mb-2">Token count: <span class="text-gray-100 font-mono">${tokenCount ?? "N/A"}</span></div>`;
  if (!chunks || chunks.length === 0) {
    html += `<p class="text-xs text-gray-500">No context chunks retrieved.</p>`;
  } else {
    html += `<div class="text-xs text-gray-400 mb-1">Sources (${chunks.length}):</div>`;
    chunks.forEach((c, i) => {
      const preview = c.text.length > 120 ? c.text.slice(0, 120) + "..." : c.text;
      html += `<div class="chunk-item pl-2 mb-2 text-xs">
        <div class="text-gray-400">#${i + 1} — score: <span class="text-emerald-400 font-mono">${c.score}</span></div>
        <div class="text-gray-300">${preview}</div>
      </div>`;
    });
  }
  debugContent.innerHTML = html;
}

async function checkHealth() {
  updateStatus("connecting");
  try {
    const res = await fetch("/api/v1/health");
    if (res.ok) {
      const data = await res.json();
      updateStatus("online");
    } else {
      updateStatus("offline");
    }
  } catch {
    updateStatus("offline");
  }
}

async function sendQuery(query) {
  if (abortController) abortController.abort();
  abortController = new AbortController();

  addMessage(query, "user");
  activeBotDiv = addMessage("", "bot");
  setFormEnabled(false);
  updateStatus("streaming");

  const payload = { query, top_k: 5, threshold: 0.3, max_tokens: 256 };

  try {
    const response = await fetch("/api/v1/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.is_end) {
            renderDebug(data.sources || [], null);
            updateStatus("online");
          } else {
            activeBotDiv.textContent += data.token;
            chatMessages.scrollTop = chatMessages.scrollHeight;
          }
        } catch {
          // skip malformed lines
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      activeBotDiv.textContent += "\n[Error: " + err.message + "]";
      updateStatus("offline");
    }
  } finally {
    setFormEnabled(true);
    queryInput.focus();
    abortController = null;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;
  queryInput.value = "";
  sendQuery(q);
});

checkHealth();
