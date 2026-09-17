/* Tetris played by the fan-out decoder: every spawned piece becomes one extraction call whose
 * single field, `placement`, ranges over the legal placements of that piece. */
(() => {
  const $ = (id) => document.getElementById(id);
  const W = 10, H = 20;
  const COLORS = { I: "#0f766e", O: "#b45309", T: "#6d28d9", S: "#15803d", Z: "#b42318", J: "#1d4ed8", L: "#c2410c" };
  // Rotation states as cell offsets (row, col), rotation 0 first; spawned at the top.
  const SHAPES = {
    I: [[[0,0],[0,1],[0,2],[0,3]], [[0,0],[1,0],[2,0],[3,0]]],
    O: [[[0,0],[0,1],[1,0],[1,1]]],
    T: [[[0,0],[0,1],[0,2],[1,1]], [[0,0],[1,0],[2,0],[1,1]], [[1,0],[1,1],[1,2],[0,1]], [[0,1],[1,1],[2,1],[1,0]]],
    S: [[[0,1],[0,2],[1,0],[1,1]], [[0,0],[1,0],[1,1],[2,1]]],
    Z: [[[0,0],[0,1],[1,1],[1,2]], [[0,1],[1,0],[1,1],[2,0]]],
    J: [[[0,0],[1,0],[1,1],[1,2]], [[0,0],[0,1],[1,0],[2,0]], [[0,0],[0,1],[0,2],[1,2]], [[0,1],[1,1],[2,1],[2,0]]],
    L: [[[0,2],[1,0],[1,1],[1,2]], [[0,0],[1,0],[2,0],[2,1]], [[0,0],[0,1],[0,2],[1,0]], [[0,0],[0,1],[1,1],[2,1]]],
  };
  const PIECE_NAMES = { I: "I (line)", O: "O (square)", T: "T", S: "S", Z: "Z", J: "J", L: "L" };

  // ---------------------------------------------------------------- game state
  let board, bag, piece, next, pieces, lines, score, running, gameOver, timer, lastDecision;
  let tps = { count: 0, since: performance.now() };

  function newBoard() { return Array.from({ length: H }, () => Array(W).fill(null)); }
  function drawBag() { if (!bag.length) bag = ["I","O","T","S","Z","J","L"].sort(() => Math.random() - 0.5); return bag.pop(); }
  function reset() {
    board = newBoard(); bag = []; piece = drawBag(); next = drawBag();
    pieces = 0; lines = 0; score = 0; gameOver = false; lastDecision = null;
    tps = { count: 0, since: performance.now() };
    render(); renderStats(); $("tetris-probs").innerHTML = ""; $("tetris-status").textContent = "";
  }

  // ---------------------------------------------------------------- placement mechanics
  function cells(type, rot, row, col) {
    return SHAPES[type][rot % SHAPES[type].length].map(([r, c]) => [row + r, col + c]);
  }
  function fits(b, cs) { return cs.every(([r, c]) => r >= 0 && r < H && c >= 0 && c < W && !b[r][c]); }
  function dropRow(b, type, rot, col) {
    // Hard-drop from above: lowest row at which the piece still fits. -1 if it doesn't fit at all.
    let row = -1;
    for (let r = 0; r < H; r++) { if (fits(b, cells(type, rot, r, col))) row = r; else if (row >= 0) break; }
    return row;
  }
  function place(b, type, rot, row, col) {
    const nb = b.map((r) => r.slice());
    for (const [r, c] of cells(type, rot, row, col)) nb[r][c] = type;
    let cleared = 0;
    const kept = nb.filter((r) => { const full = r.every(Boolean); if (full) cleared++; return !full; });
    while (kept.length < H) kept.unshift(Array(W).fill(null));
    return { board: kept, cleared };
  }
  function heights(b) { return Array.from({ length: W }, (_, c) => { for (let r = 0; r < H; r++) if (b[r][c]) return H - r; return 0; }); }
  function holes(b) { let n = 0; for (let c = 0; c < W; c++) { let seen = false; for (let r = 0; r < H; r++) { if (b[r][c]) seen = true; else if (seen) n++; } } return n; }
  function bumpiness(hs) { let s = 0; for (let i = 1; i < W; i++) s += Math.abs(hs[i] - hs[i - 1]); return s; }

  /** Every (rotation, column) the piece can be hard-dropped into, with the consequences of doing so. */
  function legalPlacements(b, type) {
    const out = [];
    const rots = SHAPES[type].length;
    const before = holes(b);
    for (let rot = 0; rot < rots; rot++) {
      const width = Math.max(...SHAPES[type][rot].map(([, c]) => c)) + 1;
      for (let col = 0; col <= W - width; col++) {
        const row = dropRow(b, type, rot, col);
        if (row < 0) continue;
        const res = place(b, type, rot, row, col);
        const hs = heights(res.board);
        const newHoles = holes(res.board) - before;
        const maxH = Math.max(...hs);
        // A plain heuristic (Dellacherie-flavoured) used only as a reference in the UI.
        const heur = res.cleared * 10 - Math.max(0, newHoles) * 8 - maxH * 1.5 - bumpiness(hs) * 0.5;
        out.push({ rot, col, row, label: `column ${col}, rotation ${rot}`, cleared: res.cleared, newHoles, maxH, bump: bumpiness(hs), heur });
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- the decision
  function boardText(b) {
    return b.map((r) => r.map((c) => (c ? "#" : ".")).join("")).join("\n");
  }
  function buildPreset(placements) {
    const hs = heights(board);
    const context =
      `TETRIS BOARD - 10 columns (0-9, left to right) x 20 rows, top row first. '#' = filled, '.' = empty.\n` +
      boardText(board) + "\n\n" +
      `Column heights: ${hs.join(" ")}\nHoles: ${holes(board)}\nLines cleared so far: ${lines}\n` +
      `Falling piece: ${PIECE_NAMES[piece]}. Next piece: ${PIECE_NAMES[next]}.\n\n` +
      `Legal placements (rotation 0 = spawn orientation, clockwise) with the result of each: lines cleared, new holes, stack height:\n` +
      placements.map((p) => `- ${p.label}: ${p.cleared} lines, ${p.newHoles > 0 ? p.newHoles : 0} holes, height ${p.maxH}`).join("\n") + "\n\n" +
      `Goal: keep the stack low and flat, avoid holes, clear lines.`;
    return {
      id: "tetris", title: "Tetris", description: "",
      context,
      schema: {
        placement: {
          type: "enum",
          description: "Where to drop the falling piece",
          choices: placements.map((p) => p.label),
        },
      },
    };
  }

  function sampleChoice(probs, temperature) {
    const entries = Object.entries(probs);
    if (temperature <= 0.01) return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    const weights = entries.map(([, p]) => Math.pow(Math.max(p, 1e-9), 1 / temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    for (let i = 0; i < entries.length; i++) { r -= weights[i]; if (r <= 0) return entries[i][0]; }
    return entries[entries.length - 1][0];
  }

  async function decide(placements) {
    const t0 = performance.now();
    const r = await fetch("/api/run/parallel", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildPreset(placements)),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const res = await r.json();
    const field = res.fields[0];
    const temperature = parseFloat($("tetris-temp").value);
    const chosenLabel = sampleChoice(field.probs, temperature);
    const chosen = placements.find((p) => p.label === chosenLabel) || placements.find((p) => p.label === field.value);
    return { chosen, probs: field.probs, argmax: field.value, engineMs: res.elapsedMs, roundTripMs: performance.now() - t0, passes: res.forwardPasses };
  }

  // ---------------------------------------------------------------- game loop
  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

  /** The placements offered to the model: all legal ones, or the heuristic's top N in column order. */
  function candidates(all) {
    const mode = $("tetris-candidates").value;
    if (mode === "all" || all.length <= 6) return all;
    return all.slice().sort((a, b) => b.heur - a.heur).slice(0, parseInt(mode, 10))
      .sort((a, b) => a.col - b.col || a.rot - b.rot);
  }

  async function turn() {
    const all = legalPlacements(board, piece);
    if (!all.length) { endGame(); return; }
    const placements = candidates(all);
    let d;
    try { d = await decide(placements); }
    catch (e) { $("tetris-status").textContent = e.message; running = false; renderControls(); return; }
    if (!running) return;
    lastDecision = { ...d, placements, all };
    renderProbs();

    // Animate: rotate, slide to the column at the top, then fall row by row.
    const p = d.chosen;
    const speed = parseInt($("tetris-speed").value, 10);
    for (let r = 0; r <= p.row; r++) {
      render({ type: piece, rot: p.rot, row: r, col: p.col });
      await sleep(speed);
      if (!running) return;
    }
    const res = place(board, piece, p.rot, p.row, p.col);
    board = res.board;
    pieces++; lines += res.cleared; score += [0, 100, 300, 500, 800][res.cleared] || 0;
    tps.count++;
    piece = next; next = drawBag();
    render(); renderStats();
    if (!fits(board, cells(piece, 0, 0, 3))) { endGame(); return; }
    timer = setTimeout(turn, 0);
  }

  function endGame() {
    running = false; gameOver = true;
    $("tetris-status").textContent = `Game over after ${pieces} pieces and ${lines} lines.`;
    renderControls();
  }

  function start() {
    if (gameOver) reset();
    running = true; renderControls();
    tps = { count: 0, since: performance.now() };
    turn();
  }
  function pause() { running = false; clearTimeout(timer); renderControls(); }

  // ---------------------------------------------------------------- rendering
  const CELL = 24;
  function render(falling) {
    const cv = $("tetris-canvas");
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.strokeStyle = "#e6ebe8"; ctx.lineWidth = 1;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      ctx.strokeRect(c * CELL + 0.5, r * CELL + 0.5, CELL, CELL);
      if (board[r][c]) { ctx.fillStyle = COLORS[board[r][c]]; ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 1, CELL - 1); }
    }
    if (falling) {
      ctx.fillStyle = COLORS[falling.type];
      for (const [r, c] of cells(falling.type, falling.rot, falling.row, falling.col)) ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 1, CELL - 1);
    }
    // next piece preview
    const nv = $("tetris-next"); const nctx = nv.getContext("2d");
    nctx.fillStyle = "#ffffff"; nctx.fillRect(0, 0, nv.width, nv.height);
    nctx.fillStyle = COLORS[next];
    for (const [r, c] of SHAPES[next][0]) nctx.fillRect(c * 18 + 4, r * 18 + 4, 17, 17);
  }

  function renderStats() {
    const secs = (performance.now() - tps.since) / 1000;
    $("tetris-pieces").textContent = pieces;
    $("tetris-lines").textContent = lines;
    $("tetris-score").textContent = score;
    $("tetris-rate").textContent = secs > 1 ? (tps.count / secs).toFixed(1) + " /s" : "–";
    if (lastDecision) {
      $("tetris-latency").textContent = Math.round(lastDecision.engineMs) + " ms";
      $("tetris-passes").textContent = lastDecision.passes;
    }
  }

  function renderProbs() {
    const d = lastDecision; if (!d) return;
    const best = d.all.reduce((a, b) => (b.heur > a.heur ? b : a));
    const rows = Object.entries(d.probs).slice(0, 8).map(([label, p]) => {
      const pl = d.placements.find((x) => x.label === label);
      const cls = label === d.chosen.label ? "chosen" : "";
      const note = pl ? `${pl.cleared ? pl.cleared + " line" + (pl.cleared > 1 ? "s" : "") + ", " : ""}${pl.newHoles > 0 ? pl.newHoles + " hole" + (pl.newHoles > 1 ? "s" : "") + ", " : ""}h ${pl.maxH}` : "";
      return `<li class="${cls}"><span class="plabel">${label}</span><span class="pbar"><i style="width:${(p * 100).toFixed(0)}%"></i></span><span class="pprob">${(p * 100).toFixed(0)}%</span><span class="pnote">${note}</span></li>`;
    }).join("");
    $("tetris-probs").innerHTML = rows;
    $("tetris-heur").textContent = `Heuristic would pick ${best.label}` + (best.label === d.chosen.label ? " — same." : ".");
    renderStats();
  }

  function renderControls() {
    $("tetris-start").textContent = running ? "Pause" : (gameOver ? "Play again" : (pieces ? "Resume" : "Start"));
  }

  // ---------------------------------------------------------------- wiring
  $("tetris-start").addEventListener("click", () => (running ? pause() : start()));
  $("tetris-reset").addEventListener("click", () => { pause(); reset(); });
  $("tetris-temp").addEventListener("input", () => { $("tetris-temp-val").textContent = parseFloat($("tetris-temp").value).toFixed(1); });
  window.pcdTetris = { pause, start };
  reset();
  const params = new URLSearchParams(location.search);
  if (params.get("view") === "tetris" && params.has("autorun")) start();
})();
