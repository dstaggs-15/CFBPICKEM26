/* CFB Pick'em Model — board renderer.
 * Reads predictions.json (produced weekly by the pipeline) and team_colors.json,
 * and renders one card per game. No framework, no build step — GitHub Pages
 * serves these files as-is.
 */

const DEFAULT_COLORS = { primary: "#6b7280", secondary: "#9ca3af" };

const state = { games: [], colors: {}, filter: "" };

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

function colorsFor(team) {
  // team_colors.json keys should match the team names in predictions.json.
  return state.colors[team] || DEFAULT_COLORS;
}

function pct(p) { return `${Math.round(p * 100)}%`; }

function rankLabel(r) { return (r === null || r === undefined) ? "" : `#${r}`; }

function render() {
  const board = document.getElementById("board");
  const tpl = document.getElementById("card-tpl");
  const q = state.filter.trim().toLowerCase();

  const games = state.games.filter(g =>
    !q || g.home_team.toLowerCase().includes(q) || g.away_team.toLowerCase().includes(q)
  );

  board.innerHTML = "";
  if (!games.length) {
    board.innerHTML = `<p class="empty">${q ? "No games match that filter." : "No games on the board yet."}</p>`;
    return;
  }

  for (const g of games) {
    const node = tpl.content.cloneNode(true);
    const homeP = g.model_prob_home;
    const awayP = 1 - homeP;
    const homeColors = colorsFor(g.home_team);
    const awayColors = colorsFor(g.away_team);

    // Names + ranks (away on the left, home on the right — the "@" convention)
    const away = node.querySelector(".team--away");
    away.querySelector(".rank").textContent = rankLabel(g.away_rank);
    away.querySelector(".name").textContent = g.away_team;
    away.querySelector(".name").style.color = awayColors.primary;

    const home = node.querySelector(".team--home");
    home.querySelector(".rank").textContent = rankLabel(g.home_rank);
    home.querySelector(".name").textContent = g.home_team;
    home.querySelector(".name").style.color = homeColors.primary;

    if (g.neutral) node.querySelector(".at").textContent = "vs";

    // Percentages
    node.querySelector(".pct--away").textContent = pct(awayP);
    node.querySelector(".pct--home").textContent = pct(homeP);

    // Signature meter: each side filled with its team's color, meeting at a seam.
    const aFill = node.querySelector(".meter--away");
    const hFill = node.querySelector(".meter--home");
    aFill.style.width = `${awayP * 100}%`;
    aFill.style.background = awayColors.primary;
    hFill.style.width = `${homeP * 100}%`;
    hFill.style.background = homeColors.primary;
    node.querySelector(".meter").setAttribute(
      "aria-label",
      `${g.away_team} ${pct(awayP)} versus ${g.home_team} ${pct(homeP)}`
    );

    // Pick + market comparison
    node.querySelector(".pick-team").textContent = g.pick;
    const marketEl = node.querySelector(".market");
    if (g.market_prob_home === null || g.market_prob_home === undefined) {
      marketEl.textContent = "no line";
    } else {
      const modelHome = homeP >= 0.5;
      const marketHome = g.market_prob_home >= 0.5;
      const agree = modelHome === marketHome;
      const spread = (g.spread_home ?? null);
      const spreadStr = spread === null ? "" :
        `line ${spread > 0 ? "+" : ""}${spread}`;
      marketEl.innerHTML =
        `${spreadStr}<span class="tag ${agree ? "agree" : "disagree"}">${agree ? "agree" : "disagree"}</span>`;
    }

    // AI take slot — populated later by Layer 3.
    const aiBody = node.querySelector(".ai-body");
    if (g.ai_note) {
      aiBody.textContent = g.ai_note;
    } else {
      aiBody.textContent = "coming soon";
      aiBody.classList.add("pending");
    }

    board.appendChild(node);
  }
}

async function main() {
  const dek = document.getElementById("dek");
  try {
    const [preds, colors] = await Promise.all([
      loadJSON("predictions.json"),
      loadJSON("team_colors.json").catch(() => ({})), // colors optional
    ]);
    state.games = preds.games || [];
    state.colors = colors || {};

    const wk = preds.week ? `Week ${preds.week}` : "";
    const ssn = preds.season || "";
    dek.textContent = `${ssn} ${wk} — ${state.games.length} games`.trim();
    document.getElementById("stamp").textContent =
      preds.generated_at ? `Generated ${preds.generated_at}` : "";

    render();
  } catch (err) {
    document.getElementById("board").innerHTML =
      `<p class="error">Couldn't load the board (${err.message}). ` +
      `If you just set this up, make sure predictions.json exists in /docs.</p>`;
    dek.textContent = "";
  }
}

document.getElementById("filter").addEventListener("input", (e) => {
  state.filter = e.target.value;
  render();
});

main();
