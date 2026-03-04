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

function commandArgv(command, season, round) {
  if (command === "train") return ["train"];
  if (command === "run-round") return ["run-round", "--season", season, "--round", round, "--print", "--explain"];
  if (command === "predict-race") return ["predict", "race", "--season", season, "--round", round, "--tags", "dist,top", "--explain", "--print"];
  if (command === "predict-quali") return ["predict", "quali", "--season", season, "--round", round, "--tags", "dist,top", "--explain", "--print"];
  if (command === "update-race") return ["update", "race", "--season", season, "--round", round];
  if (command === "update-quali") return ["update", "quali", "--season", season, "--round", round];
  return [];
}

async function pollJob(jobId) {
  const status = document.getElementById("job-status");
  const output = document.getElementById("job-output");

  while (true) {
    const out = await fetchJson(`/api/jobs/${jobId}`);
    status.textContent = `status=${out.status} rc=${out.returncode ?? "..."}`;
    output.textContent = out.output || "";
    output.scrollTop = output.scrollHeight;
    if (out.status === "completed" || out.status === "failed") break;
    await new Promise((r) => setTimeout(r, 1200));
  }
}

document.getElementById("job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const command = fd.get("command");
  const season = String(fd.get("season"));
  const round = String(fd.get("round"));
  const argv = commandArgv(command, season, round);
  if (argv.length === 0) return;

  const out = await fetchJson("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ argv }),
  });
  pollJob(out.job_id);
});

