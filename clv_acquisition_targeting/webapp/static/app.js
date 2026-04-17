const logEl = document.getElementById("calc-log");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result-panel");
const form = document.getElementById("csr-form");

function setStatus(state, text) {
  statusEl.textContent = text;
  statusEl.className = "status-pill " + state;
}

function clearLog() {
  logEl.innerHTML = "";
  resultEl.innerHTML = "";
}

function appendStep(step) {
  const card = document.createElement("article");
  card.className = "step-card";
  const title = step.title || "Step";
  const n = step.step != null ? `Step ${step.step}` : "";
  card.innerHTML = `
    <div class="title-row">
      <h3>${escapeHtml(title)}</h3>
      <span class="step-meta">${escapeHtml(n)}</span>
    </div>
    <p class="step-detail">${escapeHtml(step.detail || "")}</p>
    <pre class="data">${escapeHtml(JSON.stringify(step.data || {}, null, 2))}</pre>
  `;
  logEl.appendChild(card);
  logEl.scrollTop = logEl.scrollHeight;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function showResult(res) {
  const band = res.risk_band || "";
  const bandNice = band === "lower" ? "lower lapse risk" : band === "moderate" ? "moderate lapse risk" : "higher lapse risk";
  resultEl.innerHTML = `
    <div class="result-banner">
      <h3>Estimated customer lifetime value</h3>
      <div class="result-clv">${formatMoney(res.clv)}</div>
      <div class="result-sub">
        Churn / lapse probability (next window): <strong>${(res.churn_probability * 100).toFixed(2)}%</strong>
        — ${bandNice}.<br/>
        Historical part: ${formatMoney(res.historical_value)} · Future (discounted) part: ${formatMoney(res.future_value)}<br/>
        <span style="opacity:.85">${escapeHtml(res.currency_note || "")}</span>
      </div>
    </div>
  `;
}

function formatMoney(v) {
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return n.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

async function runStream() {
  clearLog();
  setStatus("running", "Running calculations…");
  form.querySelector("button.primary").disabled = true;

  const payload = {
    age: Number(document.getElementById("age").value),
    tenure_months: Number(document.getElementById("tenure_months").value),
    num_orders: Number(document.getElementById("num_orders").value),
    total_spend: Number(document.getElementById("total_spend").value),
    avg_order_value: Number(document.getElementById("avg_order_value").value),
    days_since_last_order: Number(document.getElementById("days_since_last_order").value),
    email_opens_30d: Number(document.getElementById("email_opens_30d").value),
    app_sessions_30d: Number(document.getElementById("app_sessions_30d").value),
    region: document.getElementById("region").value,
  };

  const loading = document.createElement("div");
  loading.className = "shimmer";
  loading.id = "stream-loading";
  logEl.appendChild(loading);

  try {
    const res = await fetch("/api/predict/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          continue;
        }
        if (msg.type === "step") {
          document.getElementById("stream-loading")?.remove();
          appendStep(msg.step);
        } else if (msg.type === "result") {
          document.getElementById("stream-loading")?.remove();
          showResult(msg.result);
          setStatus("done", "Complete");
        } else if (msg.type === "error") {
          document.getElementById("stream-loading")?.remove();
          setStatus("error", msg.message || "Error");
          appendStep({
            step: "!",
            title: "Could not complete",
            detail: msg.message || "Invalid request",
            data: {},
          });
        }
      }
    }
  } catch (e) {
    document.getElementById("stream-loading")?.remove();
    setStatus("error", "Request failed");
    appendStep({
      step: "!",
      title: "Error",
      detail: String(e.message || e),
      data: {},
    });
  } finally {
    form.querySelector("button.primary").disabled = false;
  }
}

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  runStream();
});

document.getElementById("btn-fill-demo").addEventListener("click", () => {
  document.getElementById("age").value = 48;
  document.getElementById("tenure_months").value = 36;
  document.getElementById("num_orders").value = 14;
  document.getElementById("total_spend").value = 12800;
  document.getElementById("avg_order_value").value = 420;
  document.getElementById("days_since_last_order").value = 18;
  document.getElementById("email_opens_30d").value = 5;
  document.getElementById("app_sessions_30d").value = 4;
  document.getElementById("region").value = "MW";
});
