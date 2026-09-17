/* Parallel constrained decoding — local demo UI. Talks to the Java server in the same jar. */
(() => {
  const $ = (id) => document.getElementById(id);
  const api = async (path, opts = {}) => {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    return r.json();
  };
  const fmtMs = (ms) => ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";

  /** Reads a fetch() body as Server-Sent Events, calling on(event, data) for each. */
  async function sse(path, body, on) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        let ev = "message", data = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event: ")) ev = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        on(ev, data ? JSON.parse(data) : {});
      }
    }
  }

  // ---------------------------------------------------------------- state
  let presets = [];          // summaries
  let current = null;        // full preset shown in the race view
  let editing = null;        // full preset in the editor
  let sampleIdx = -1;        // index into current.samples shown in the textarea (-1: none)
  let lastRunIdx = -1;       // sample index used by the previous run, to rotate on the next one

  // ---------------------------------------------------------------- views
  function showView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("is-active", v.id === "view-" + name));
    document.querySelectorAll(".view-tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === name));
    if (name === "bench") loadRuns();
    if (name === "presets" && !editing && current) openEditor(current.id);
  }
  document.querySelectorAll(".view-tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
  document.querySelectorAll("[data-view-link]").forEach((b) => b.addEventListener("click", () => {
    if (current) openEditor(current.id);
    showView(b.dataset.viewLink);
  }));

  // ---------------------------------------------------------------- presets
  async function loadPresets(selectId) {
    presets = await api("/api/presets");
    // Demo-first ordering: the Devoxx routing scenario leads, then spam triage, then the rest by title.
    const lead = ["devoxx_cfp", "spam_email"];
    presets.sort((a, b) => {
      const ia = lead.indexOf(a.id), ib = lead.indexOf(b.id);
      if (ia !== ib) return (ia === -1 ? lead.length : ia) - (ib === -1 ? lead.length : ib);
      return a.title.localeCompare(b.title);
    });
    const sel = $("preset-select");
    sel.innerHTML = presets.map((p) => `<option value="${p.id}">${esc(p.title)}</option>`).join("");
    const preferred = selectId || new URLSearchParams(location.search).get("preset") || "devoxx_cfp";
    const id = presets.some((p) => p.id === preferred) ? preferred : (presets[0] && presets[0].id);
    if (id) { sel.value = id; await selectPreset(id); }
    renderPresetList();
  }

  function currentSample() {
    return current && current.samples && sampleIdx >= 0 ? current.samples[sampleIdx] : null;
  }

  /** Shows sample i of the current preset in the textarea (presets without samples show their context). */
  function showSample(i) {
    const n = current.samples ? current.samples.length : 0;
    sampleIdx = n ? ((i % n) + n) % n : -1;
    const s = currentSample();
    $("context").value = s ? s.context : current.context;
    $("context-meta").textContent = s
      ? `${current.description || ""} Talk ${sampleIdx + 1} of ${n}: ${s.label}`
      : (current.description || "");
    $("next-sample").hidden = n < 2;
    resetRace();
  }
  $("next-sample").addEventListener("click", () => showSample(sampleIdx + 1));

  async function selectPreset(id) {
    current = await api("/api/presets/" + id);
    lastRunIdx = -1;
    $("context-title").textContent = "Input";
    $("context-meta").textContent = current.description || "";
    $("schema-chips").innerHTML = Object.entries(current.schema).map(([name, f]) =>
      `<li title="${esc(f.description)}">${esc(name)} <span>${f.type === "boolean" ? "true | false" : f.choices.length + " values"}</span></li>`).join("");
    showSample(current.samples && current.samples.length ? Math.floor(Math.random() * current.samples.length) : -1);
  }
  $("preset-select").addEventListener("change", (e) => selectPreset(e.target.value));

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------------------------------------------------------------- race
  const RULER_MIN_MS = 3000;
  let rulerMax = RULER_MIN_MS;
  let raf = null;

  function drawTicks() {
    const ticks = $("ruler-ticks");
    ticks.innerHTML = "";
    for (let i = 0; i <= 10; i++) {
      const s = document.createElement("span");
      s.style.left = (i * 10) + "%";
      s.textContent = i === 0 ? "0" : (rulerMax * i / 10 / 1000).toFixed(1) + " s";
      ticks.appendChild(s);
    }
  }
  /** Rescale the ruler to the measured times once both are final (the live timer may have overshot). */
  function fitRuler(naiveMs, parMs) {
    rulerMax = Math.max(RULER_MIN_MS, Math.ceil(Math.max(naiveMs, parMs) / 500) * 500);
    drawTicks();
    setBar("naive", naiveMs); setBar("parallel", parMs);
  }
  function setBar(which, ms) {
    if (ms > rulerMax) { rulerMax = Math.ceil(ms / 1000) * 1000; drawTicks(); }
    $("bar-" + which).style.width = Math.min(100, ms / rulerMax * 100) + "%";
    $("time-" + which).textContent = fmtMs(ms);
  }

  function resetRace() {
    if (raf) cancelAnimationFrame(raf);
    rulerMax = RULER_MIN_MS; drawTicks();
    setBar("naive", 0); setBar("parallel", 0);
    $("naive-stream").textContent = ""; $("naive-issues").innerHTML = "";
    for (const id of ["naive-ms", "naive-passes", "naive-valid", "par-ms", "par-passes", "par-valid"]) { $(id).textContent = "–"; $(id).className = ""; }
    $("verdict").hidden = true;
    const fields = $("par-fields");
    fields.className = "fields pending";
    fields.innerHTML = current ? Object.keys(current.schema).map((n) =>
      `<li><span class="fname">${esc(n)}</span><span class="fval">…</span><span class="fbar-wrap"><span class="fbar"><i style="width:0"></i></span><span class="fprob"></span></span></li>`).join("") : "";
  }

  function renderParallel(res) {
    const fields = $("par-fields");
    const expected = (currentSample() && currentSample().expected) || {};
    fields.className = "fields";
    fields.innerHTML = res.fields.map((f) => {
      const alts = Object.entries(f.probs).map(([c, p]) => `${c}: ${(p * 100).toFixed(1)}%`).join("\n");
      const exp = expected[f.name];
      const expHtml = exp === undefined ? "" :
        `<span class="fexp ${String(f.value) === exp ? "match" : "miss"}">CFP: ${esc(exp)}</span>`;
      return `<li title="${esc(alts)}"><span class="fname">${esc(f.name)}</span><span class="fval">${esc(f.value)}${expHtml}</span>` +
        `<span class="fbar-wrap"><span class="fbar"><i style="width:${(f.prob * 100).toFixed(0)}%"></i></span><span class="fprob">${(f.prob * 100).toFixed(0)}%${f.levels > 1 ? " · " + f.levels + " lvl" : ""}</span></span></li>`;
    }).join("");
    $("par-ms").textContent = fmtMs(res.elapsedMs);
    $("par-passes").textContent = res.forwardPasses;
    $("par-valid").textContent = "valid"; $("par-valid").className = "ok";
    setBar("parallel", res.elapsedMs);
  }

  function renderNaive(res, parallel) {
    $("naive-ms").textContent = fmtMs(res.elapsedMs);
    $("naive-passes").textContent = res.forwardPasses;
    const ok = res.schemaMatch;
    $("naive-valid").textContent = ok ? "valid" : (res.validJson ? "off-schema" : "broken");
    $("naive-valid").className = ok ? "ok" : "bad";
    fitRuler(res.elapsedMs, parallel.elapsedMs);

    const issues = [];
    if (!res.validJson) issues.push("The output is not valid JSON.");
    for (const k of res.missingKeys) issues.push(`Missing field: ${k}`);
    for (const k of res.extraKeys) issues.push(`Invented field: ${k}`);
    for (const k of res.invalidEnums) issues.push(`Value outside the allowed set: ${k}`);
    $("naive-issues").innerHTML = issues.map((i) => `<li>${esc(i)}</li>`).join("");

    // Highlight offending keys in the streamed text.
    const bad = [...res.missingKeys.map(() => null), ...res.extraKeys, ...res.invalidEnums.map((s) => s.split("=")[0])].filter(Boolean);
    if (bad.length) {
      let html = esc($("naive-stream").textContent);
      for (const k of bad) html = html.replace(new RegExp(`"(${k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})"`, "g"), '<span class="key-bad">"$1"</span>');
      $("naive-stream").innerHTML = html;
    }

    const speed = res.elapsedMs / Math.max(parallel.elapsedMs, 1);
    const n = parallel.fields.length;
    let text = `<strong>${speed.toFixed(1)}× faster</strong> — ${n} fields decided in ${parallel.forwardPasses} forward passes instead of ${res.forwardPasses}, ` +
      (ok ? "and both outputs match the schema this time." : `and the parallel output is the only one that matches the schema.`);
    const expected = (currentSample() && currentSample().expected) || {};
    const keys = Object.keys(expected);
    if (keys.length) {
      const parHit = keys.filter((k) => parallel.fields.some((f) => f.name === k && String(f.value) === expected[k])).length;
      const naiveHit = keys.filter((k) => res.json && res.json[k] !== undefined && String(res.json[k]) === expected[k]).length;
      text += ` Against how the CFP actually filed this talk: parallel ${parHit}/${keys.length}, token by token ${naiveHit}/${keys.length}.`;
    }
    $("verdict-text").innerHTML = text;
    $("verdict").hidden = false;
  }

  async function runRace() {
    if (!current) return;
    // A repeated run on an unedited sample moves on to the next talk in the bank.
    const s = currentSample();
    if (s && lastRunIdx === sampleIdx && $("context").value === s.context) showSample(sampleIdx + 1);
    lastRunIdx = sampleIdx;
    const btn = $("run-race");
    btn.disabled = true; btn.textContent = "Running…";
    resetRace();
    const { samples, ...rest } = current;
    const body = { ...rest, context: $("context").value };
    const stream = $("naive-stream");
    let parallel = null, t0 = null, revealed = false;
    try {
      await sse("/api/run/race", body, (ev, data) => {
        if (ev === "parallel") { parallel = data; }
        else if (ev === "start") {
          t0 = performance.now();
          stream.innerHTML = '<span class="cursor"></span>';
          const tick = () => {
            const ms = performance.now() - t0;
            setBar("naive", ms);
            if (parallel && !revealed && ms >= parallel.elapsedMs) { revealed = true; renderParallel(parallel); }
            raf = requestAnimationFrame(tick);
          };
          raf = requestAnimationFrame(tick);
        } else if (ev === "token") {
          const cur = stream.querySelector(".cursor");
          stream.insertBefore(document.createTextNode(data.t), cur);
          stream.scrollTop = stream.scrollHeight;
        } else if (ev === "naive") {
          cancelAnimationFrame(raf);
          stream.querySelector(".cursor")?.remove();
          if (!revealed) renderParallel(parallel);
          renderNaive(data, parallel);
        }
      });
    } catch (e) {
      cancelAnimationFrame(raf);
      $("naive-issues").innerHTML = `<li>${esc(e.message)}</li>`;
    } finally {
      btn.disabled = false; btn.textContent = "Run the race";
    }
  }
  $("run-race").addEventListener("click", runRace);

  // ---------------------------------------------------------------- editor
  function renderPresetList() {
    $("preset-items").innerHTML = presets.map((p) =>
      `<li data-id="${p.id}" class="${editing && editing.id === p.id ? "is-active" : ""}"><div class="pt">${esc(p.title)}</div><div class="pm">${p.fields} fields</div></li>`).join("");
    $("preset-items").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => openEditor(li.dataset.id)));
  }

  async function openEditor(id) {
    editing = id ? await api("/api/presets/" + id) : { id: "", title: "", description: "", context: "", schema: {} };
    $("p-id").value = editing.id; $("p-id").disabled = !!id;
    $("p-title").value = editing.title; $("p-desc").value = editing.description || ""; $("p-context").value = editing.context;
    $("field-rows").innerHTML = "";
    for (const [name, f] of Object.entries(editing.schema)) addFieldRow(name, f);
    if (!id) addFieldRow("", { type: "enum", description: "", choices: [] });
    $("preset-delete").hidden = !id;
    $("editor-status").textContent = "";
    renderPresetList();
  }

  function addFieldRow(name, f) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><input name="name" value="${esc(name)}" placeholder="field_name" pattern="[A-Za-z_][A-Za-z0-9_]*" required></td>` +
      `<td><select name="type"><option value="enum">choice</option><option value="boolean">true / false</option></select></td>` +
      `<td><input name="description" value="${esc(f.description)}" placeholder="What the model should decide"></td>` +
      `<td><input name="choices" value="${esc((f.choices || []).join(", "))}" placeholder="A, B, C"></td>` +
      `<td><button type="button" class="remove" aria-label="Remove field">×</button></td>`;
    tr.querySelector('[name="type"]').value = f.type;
    const syncType = () => { tr.querySelector('[name="choices"]').disabled = tr.querySelector('[name="type"]').value === "boolean"; };
    tr.querySelector('[name="type"]').addEventListener("change", syncType); syncType();
    tr.querySelector(".remove").addEventListener("click", () => tr.remove());
    $("field-rows").appendChild(tr);
  }
  $("field-add").addEventListener("click", () => addFieldRow("", { type: "enum", description: "", choices: [] }));
  $("preset-new").addEventListener("click", () => openEditor(null));

  function readEditor() {
    const schema = {};
    for (const tr of $("field-rows").querySelectorAll("tr")) {
      const g = (n) => tr.querySelector(`[name="${n}"]`).value.trim();
      const name = g("name");
      if (!name) continue;
      const type = g("type");
      schema[name] = { type, description: g("description") };
      if (type === "enum") schema[name].choices = g("choices").split(",").map((s) => s.trim()).filter(Boolean);
    }
    return { id: $("p-id").value.trim(), title: $("p-title").value.trim(), description: $("p-desc").value.trim(), context: $("p-context").value, schema };
  }

  async function savePreset() {
    const p = readEditor();
    $("editor-status").textContent = "Saving…";
    try {
      const saved = await api("/api/presets/" + p.id, { method: "PUT", body: JSON.stringify(p) });
      $("editor-status").textContent = "Saved.";
      editing = saved; $("p-id").disabled = true; $("preset-delete").hidden = false;
      await loadPresets(saved.id);
      return saved;
    } catch (e) {
      $("editor-status").textContent = e.message;
      return null;
    }
  }
  $("preset-form").addEventListener("submit", (e) => { e.preventDefault(); savePreset(); });
  $("preset-run").addEventListener("click", async () => { if (await savePreset()) { showView("demo"); runRace(); } });
  $("preset-delete").addEventListener("click", async () => {
    if (!editing?.id || !confirm(`Delete "${editing.title}"?`)) return;
    await api("/api/presets/" + editing.id, { method: "DELETE" });
    editing = null;
    await loadPresets();
    openEditor(presets[0]?.id ?? null);
  });

  // ---------------------------------------------------------------- benchmarks
  let charts = {};
  let runsCache = [];

  async function loadRuns() {
    runsCache = await api("/api/results");
    const sel = $("run-select");
    sel.innerHTML = runsCache.length
      ? runsCache.map((r, i) => `<option value="${i}">${new Date(r.timestamp).toLocaleString()}</option>`).join("")
      : `<option value="">No runs yet</option>`;
    if (runsCache.length) renderRun(runsCache[0]);
    else { $("bench-rows").innerHTML = `<tr class="pending"><td colspan="7">Run all scenarios to get the first set of numbers.</td></tr>`; }
  }
  $("run-select").addEventListener("change", (e) => { if (e.target.value !== "") renderRun(runsCache[+e.target.value]); });

  function rowHtml(r) {
    const n = r.naive, p = r.parallel;
    const status = n.schemaMatch ? "valid" : (n.validJson ? [...n.missingKeys.map((k) => "missing " + k), ...n.invalidEnums.map((k) => "off-schema " + k)].join(", ") : "invalid JSON");
    return `<tr><td>${esc(r.title)}</td><td class="num">${r.fields}</td>` +
      `<td class="num naive">${fmtMs(n.elapsedMs)}</td><td class="num par">${fmtMs(p.elapsedMs)}</td><td class="num">${r.speedup}×</td>` +
      `<td class="${n.schemaMatch ? "ok" : "bad"}">${esc(status)}</td><td class="num">${n.forwardPasses} → ${p.forwardPasses}</td></tr>`;
  }

  function renderRun(run) {
    $("bench-rows").innerHTML = run.presets.map(rowHtml).join("");
    drawCharts(run.presets);
  }

  function drawCharts(rows) {
    for (const c of Object.values(charts)) c.destroy();
    const labels = rows.map((r) => { const t = r.title.replace(/\s*\(.*\)$/, ""); return t.length > 30 ? t.slice(0, 29) + "…" : t; });
    const par = getComputedStyle(document.documentElement).getPropertyValue("--par").trim();
    const naive = getComputedStyle(document.documentElement).getPropertyValue("--naive").trim();
    const font = { family: "IBM Plex Sans" };
    charts.latency = new Chart($("chart-latency"), {
      type: "bar",
      data: { labels, datasets: [
        { label: "Token by token", data: rows.map((r) => r.naive.elapsedMs), backgroundColor: naive },
        { label: "Parallel constrained", data: rows.map((r) => r.parallel.elapsedMs), backgroundColor: par },
      ] },
      options: { indexAxis: "y", responsive: true, plugins: { legend: { labels: { font } } },
        scales: { x: { title: { display: true, text: "milliseconds", font }, ticks: { font } }, y: { ticks: { font } } } },
    });
    const phases = ["prefill", "suffix", "tree", "tokenize", "broadcast"];
    const shades = ["#0f766e", "#2a9d8f", "#7cc7bd", "#b9c2bd", "#5e6b78"];
    charts.phases = new Chart($("chart-phases"), {
      type: "bar",
      data: { labels, datasets: phases.map((ph, i) => ({ label: ph, data: rows.map((r) => r.parallel.phases[ph]), backgroundColor: shades[i] })) },
      options: { indexAxis: "y", responsive: true, plugins: { legend: { labels: { font } } },
        scales: { x: { stacked: true, title: { display: true, text: "milliseconds", font }, ticks: { font } }, y: { stacked: true, ticks: { font } } } },
    });
  }

  $("run-bench").addEventListener("click", async () => {
    const btn = $("run-bench");
    btn.disabled = true; btn.textContent = "Running…";
    const rows = [];
    $("bench-rows").innerHTML = presets.map((p) => `<tr class="pending" data-id="${p.id}"><td>${esc(p.title)}</td><td colspan="6">waiting</td></tr>`).join("");
    try {
      await sse("/api/benchmark", {}, (ev, data) => {
        if (ev === "preset") {
          rows.push(data);
          const tr = $("bench-rows").querySelector(`tr[data-id="${data.id}"]`);
          if (tr) tr.outerHTML = rowHtml(data); else $("bench-rows").insertAdjacentHTML("beforeend", rowHtml(data));
          drawCharts(rows);
        } else if (ev === "done") {
          loadRuns();
        }
      });
    } catch (e) {
      $("bench-hint").textContent = e.message;
    } finally {
      btn.disabled = false; btn.textContent = "Run all scenarios";
    }
  });

  // ---------------------------------------------------------------- boot
  (async () => {
    try {
      const s = await api("/api/status");
      $("model-name").textContent = s.model;
    } catch (e) { /* status is cosmetic */ }
    await loadPresets();
    drawTicks();
    const params = new URLSearchParams(location.search);
    if (params.get("view")) showView(params.get("view"));
    if (params.has("autorun")) runRace();
  })();
})();
