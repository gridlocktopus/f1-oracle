function renderTable(targetId, rows) {
  const root = document.getElementById(targetId);
  if (!rows || rows.length === 0) {
    root.innerHTML = "<p>No rows.</p>";
    return;
  }
  const cols = Object.keys(rows[0]);
  const head = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const body = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${String(r[c] ?? "")}</td>`).join("")}</tr>`)
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

refreshCoverage();
