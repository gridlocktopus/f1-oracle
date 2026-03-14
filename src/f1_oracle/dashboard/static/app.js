const TABLE_STORE = {};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderTable(targetId, rows) {
  const root = document.getElementById(targetId);
  TABLE_STORE[targetId] = Array.isArray(rows) ? rows : [];
  if (!rows || rows.length === 0) {
    root.innerHTML = "<p>No rows.</p>";
    return;
  }
  const cols = Object.keys(rows[0]);
  const head = `<tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
  const body = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${escapeHtml(r[c] ?? "")}</td>`).join("")}</tr>`)
    .join("");
  root.innerHTML = `<table>${head}${body}</table>`;
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

document.getElementById("pred-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const query = new URLSearchParams({
    season: fd.get("season"),
    rnd: fd.get("round"),
    kind: fd.get("kind"),
    mode: fd.get("mode"),
    limit: "100",
  });
  try {
    const out = await fetchJson(`/api/predictions?${query.toString()}`);
    document.getElementById("pred-path").textContent = out.path;
    renderTable("pred-table", out.rows);
  } catch (err) {
    document.getElementById("pred-path").textContent = String(err);
    renderTable("pred-table", []);
  }
});

document.getElementById("compare-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const query = new URLSearchParams({
    season: fd.get("season"),
    rnd: fd.get("round"),
    kind: fd.get("kind"),
    mode: fd.get("mode"),
    limit: "100",
  });
  try {
    const out = await fetchJson(`/api/compare?${query.toString()}`);
    document.getElementById("compare-summary").textContent = out.summary;
    renderTable("compare-table", out.rows);
  } catch (err) {
    document.getElementById("compare-summary").textContent = String(err);
    renderTable("compare-table", []);
  }
});

document.getElementById("eval-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const query = new URLSearchParams({
    season: fd.get("season"),
    start_round: fd.get("start_round"),
    end_round: fd.get("end_round"),
    kind: fd.get("kind"),
    mode: fd.get("mode"),
  });
  try {
    const out = await fetchJson(`/api/evaluate?${query.toString()}`);
    document.getElementById("eval-summary").textContent = out.summary;
    renderTable("eval-table", out.rows);
  } catch (err) {
    document.getElementById("eval-summary").textContent = String(err);
    renderTable("eval-table", []);
  }
});

function commandArgv(command, p) {
  if (command === "season-setup") {
    return [
      ["ingest", "ergast", "calendar", "--season", p.season],
      ["ingest", "ergast", "circuits", "--season", p.season],
      ["ingest", "ergast", "drivers", "--season", p.season],
      ["ingest", "ergast", "constructors", "--season", p.season],
      ["build", "canonical", "weekends", "--season", p.season],
      ["build", "canonical", "circuits", "--season", p.season],
      ["build", "canonical", "drivers", "--season", p.season],
      ["build", "canonical", "constructors", "--season", p.season],
      ["build", "canonical", "entries", "--season", p.season],
    ];
  }
  if (command === "train") return ["train"];
  if (command === "status") return ["status", "--season", p.season, "--round", p.round];
  if (command === "predict-race") return ["predict", "race", "--season", p.season, "--round", p.round, "--tags", p.tags, "--explain", "--print"];
  if (command === "predict-quali") return ["predict", "quali", "--season", p.season, "--round", p.round, "--tags", p.tags, "--explain", "--print"];
  if (command === "compare-race-top") return ["compare", "race", "--season", p.season, "--round", p.round, "--kind", "top"];
  if (command === "compare-quali-top") return ["compare", "quali", "--season", p.season, "--round", p.round, "--kind", "top"];
  if (command === "update-race") return ["update", "race", "--season", p.season, "--round", p.round];
  if (command === "update-quali") return ["update", "quali", "--season", p.season, "--round", p.round];
  if (command === "sync-sprint-results") {
    return [
      ["ingest", "ergast", "results", "sprint", "--season", p.season],
      ["build", "canonical", "results", "sprint", "--season", p.season],
    ];
  }
  if (command === "evaluate-race-top") return ["evaluate", "--season", p.season, "--start-round", p.startRound, "--end-round", p.endRound, "--kind", "race", "--mode", "top"];
  if (command === "evaluate-race-dist") return ["evaluate", "--season", p.season, "--start-round", p.startRound, "--end-round", p.endRound, "--kind", "race", "--mode", "dist"];
  if (command === "ingest-fastf1-practice") return ["ingest", "fastf1", "practice", "--season", p.season, "--round", p.round, "--sessions", p.sessions, "--only-missing"];

  return [];
}

async function refreshCoverage() {
  const summary = document.getElementById("coverage-summary");
  try {
    const out = await fetchJson("/api/training-coverage");
    const completed = out.completed_seasons.length > 0 ? out.completed_seasons.join(", ") : "none";
    summary.textContent = [
      `trained seasons range: ${out.trained_range}`,
      `completed seasons: ${completed}`,
      `current season (${out.current_season ?? "n/a"}) progress: ${out.current_progress}`,
    ].join("\n");
    renderTable("coverage-table", out.rows);
  } catch (err) {
    summary.textContent = String(err);
    renderTable("coverage-table", []);
  }
}

function setAllInputValues(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((el) => {
    el.value = String(value);
  });
}

function getOpsSeasonRound() {
  const form = document.getElementById("job-form");
  return {
    season: String(form.elements.season.value || ""),
    round: String(form.elements.round.value || ""),
  };
}

function renderStageGuide(info) {
  const root = document.getElementById("stage-guide");
  if (!root) return;

  const sprintBlock = info.sprint_weekend
    ? `
      <li><strong>After sprint official:</strong>
        <pre class="mono">Operations -> sync sprint results
Operations -> predict race</pre>
      </li>`
    : "";

  const sprintHint = info.sprint_weekend
    ? `<p class="hint sprint-note">Sprint weekend detected. Sync sprint results after the sprint so race predictions use that extra signal.</p>`
    : `<p class="hint">Standard weekend flow: qualifying feeds race prediction directly.</p>`;

  root.innerHTML = `
    <ol class="guide-list">
      <li><strong>New season setup:</strong>
        <pre class="mono">Operations -> season setup</pre>
      </li>
      <li><strong>After practice sessions:</strong>
        <pre class="mono">Operations -> ingest practice (fastf1)
Operations -> predict qualifying</pre>
      </li>
      <li><strong>After qualifying official:</strong>
        <pre class="mono">Operations -> update after qualifying
Operations -> compare quali (top)</pre>
      </li>
      ${sprintBlock}
      <li><strong>Before the grand prix:</strong>
        <pre class="mono">Operations -> predict race</pre>
      </li>
      <li><strong>After race official:</strong>
        <pre class="mono">Operations -> update after race
Operations -> compare race (top)</pre>
      </li>
    </ol>
    ${sprintHint}
  `;
}

async function refreshWeekendInfo() {
  const banner = document.getElementById("weekend-banner");
  const title = document.getElementById("weekend-title");
  const meta = document.getElementById("weekend-meta");
  const badge = document.getElementById("weekend-badge");
  const { season, round } = getOpsSeasonRound();

  try {
    const out = await fetchJson(`/api/weekend-info?${new URLSearchParams({ season, rnd: round }).toString()}`);
    banner.className = `weekend-banner ${out.sprint_weekend ? "weekend-banner-sprint" : "weekend-banner-standard"}`;
    badge.className = `weekend-badge ${out.sprint_weekend ? "weekend-badge-sprint" : "weekend-badge-standard"}`;
    badge.textContent = out.sprint_weekend ? "Sprint Weekend" : "Standard Weekend";
    title.textContent = `${out.race_name || "Race weekend"} - round ${out.round}`;
    const scheduleBits = [
      out.qualifying_date ? `Qualifying: ${out.qualifying_date}` : null,
      out.sprint_date ? `Sprint: ${out.sprint_date}` : null,
      out.race_date ? `Race: ${out.race_date}` : null,
    ].filter(Boolean);
    meta.textContent = scheduleBits.join(" | ") || "Weekend schedule available.";
    renderStageGuide(out);
  } catch (err) {
    banner.className = "weekend-banner weekend-banner-pending";
    badge.className = "weekend-badge";
    badge.textContent = "Unavailable";
    title.textContent = "Weekend context unavailable";
    meta.textContent = String(err);
    renderStageGuide({ sprint_weekend: false });
  }
}

async function loadDefaults() {
  try {
    const out = await fetchJson("/api/defaults");
    setAllInputValues("season", out.season);
    setAllInputValues("round", out.round);
    setAllInputValues("start_round", out.start_round);
    setAllInputValues("end_round", out.end_round);
  } catch (err) {
    // Keep static HTML defaults if API defaults are unavailable.
    console.warn("Failed to load defaults:", err);
  }
}

function openTableModal(sourceId, title) {
  const rows = TABLE_STORE[sourceId] || [];
  const modal = document.getElementById("table-modal");
  const modalTitle = document.getElementById("modal-title");
  modalTitle.textContent = title || "Full view";
  renderTable("modal-table", rows);
  modal.hidden = false;
}

function closeTableModal() {
  const modal = document.getElementById("table-modal");
  modal.hidden = true;
}

async function pollJob(jobId) {
  const status = document.getElementById("job-status");
  const output = document.getElementById("job-output");

  while (true) {
    const out = await fetchJson(`/api/jobs/${jobId}`);
    status.textContent = `status=${out.status} rc=${out.returncode ?? "..."}`;
    output.textContent = out.output || "";
    output.scrollTop = output.scrollHeight;
    if (out.status === "completed" || out.status === "failed") {
      refreshCoverage();
      refreshWeekendInfo();
      break;
    }
    await new Promise((r) => setTimeout(r, 1200));
  }
}

document.getElementById("job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const command = fd.get("command");
  const params = {
    season: String(fd.get("season")),
    round: String(fd.get("round")),
    startRound: String(fd.get("start_round")),
    endRound: String(fd.get("end_round")),
    sessions: String(fd.get("sessions") || "FP1,FP2,FP3"),
    tags: String(fd.get("tags") || "dist,top"),
  };
  const cmd = commandArgv(command, params);
  if (!cmd || cmd.length === 0) return;

  const body = Array.isArray(cmd[0]) ? { argv_batch: cmd } : { argv: cmd };

  const out = await fetchJson("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  pollJob(out.job_id);
});

document.getElementById("coverage-refresh").addEventListener("click", () => {
  refreshCoverage();
});

document.querySelectorAll(".open-full").forEach((btn) => {
  btn.addEventListener("click", () => {
    openTableModal(btn.dataset.source, btn.dataset.title);
  });
});

document.getElementById("modal-close-btn").addEventListener("click", closeTableModal);
document.getElementById("modal-close-bg").addEventListener("click", closeTableModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeTableModal();
});

async function bootstrapDashboard() {
  await loadDefaults();
  await refreshCoverage();
  await refreshWeekendInfo();
}

document.querySelectorAll('#job-form input[name="season"], #job-form input[name="round"]').forEach((el) => {
  el.addEventListener("change", refreshWeekendInfo);
  el.addEventListener("input", refreshWeekendInfo);
});

bootstrapDashboard();
