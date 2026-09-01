/* CFB Pick'em Model — board renderer with expandable per-game breakdown. */
const DEFAULT_COLORS = { primary: "#6b7280", secondary: "#9ca3af" };
const state = { games: [], colors: {}, news: {}, filter: "" };

async function loadJSON(path, optional = false) {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) { if (optional) return null; throw new Error(`${path} → ${res.status}`); }
    return await res.json();
  } catch (e) { if (optional) return null; throw e; }
}

const colorsFor = (t) => state.colors[t] || DEFAULT_COLORS;
const pct = (p) => `${Math.round(p * 100)}%`;
const rankLabel = (r) => (r === null || r === undefined) ? "" : `#${r}`;

function ordinal(n) {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function renderStatList(ul, stats, homeSide) {
  ul.innerHTML = "";
  (stats || []).forEach((s) => {
    const li = document.createElement("li");
    const rank = (s.rank != null)
      ? `<span class="stat-rank">${ordinal(s.rank)}${s.of ? ` / ${s.of}` : ""}</span>` : "";
    li.innerHTML =
      `<span class="stat-label">${s.label}</span>` +
      `<span class="stat-figure"><span class="stat-val">${s.value}</span>${rank}</span>`;
    ul.appendChild(li);
  });
}

function render() {
  const board = document.getElementById("board");
  const tpl = document.getElementById("card-tpl");
  const q = state.filter.trim().toLowerCase();
  const games = state.games.filter(g =>
    !q || (g.home_team || "").toLowerCase().includes(q) || (g.away_team || "").toLowerCase().includes(q));

  board.innerHTML = "";
  if (!games.length) {
    board.innerHTML = `<p class="empty">${q ? "No games match that filter." : "No games on the board yet."}</p>`;
    return;
  }

  for (const g of games) {
    const node = tpl.content.cloneNode(true);
    if (g.error || g.model_prob_home == null) {
      // graceful card for a game we couldn't locate
      node.querySelector(".game").classList.add("notfound");
      node.querySelector(".pick-team").textContent = "—";
      node.querySelector(".team--away .name").textContent = g.away_team || "?";
      node.querySelector(".team--home .name").textContent = g.home_team || "?";
      node.querySelector(".market").textContent = g.error ? "not found" : "";
      board.appendChild(node);
      continue;
    }

    const homeP = g.model_prob_home, awayP = 1 - homeP;
    const homeFav = homeP >= 0.5;

    const away = node.querySelector(".team--away");
    away.querySelector(".rank").textContent = rankLabel(g.away_rank);
    const awayName = away.querySelector(".name");
    awayName.textContent = g.away_team;
    const home = node.querySelector(".team--home");
    home.querySelector(".rank").textContent = rankLabel(g.home_rank);
    const homeName = home.querySelector(".name");
    homeName.textContent = g.home_team;
    // the favored (picked) team's name goes red
    (homeFav ? homeName : awayName).classList.add("name--pick");
    if (g.neutral) node.querySelector(".at").textContent = "vs";

    const pctAway = node.querySelector(".pct--away");
    const pctHome = node.querySelector(".pct--home");
    pctAway.textContent = pct(awayP);
    pctHome.textContent = pct(homeP);
    (homeFav ? pctHome : pctAway).classList.add("lead");

    // meter: red = the pick, graphite = the other side
    const aFill = node.querySelector(".meter--away"), hFill = node.querySelector(".meter--home");
    aFill.style.width = `${awayP * 100}%`;
    hFill.style.width = `${homeP * 100}%`;
    aFill.classList.add(homeFav ? "is-dim" : "is-pick");
    hFill.classList.add(homeFav ? "is-pick" : "is-dim");

    node.querySelector(".pick-team").textContent = g.pick;
    const marketEl = node.querySelector(".market");
    if (g.market_prob_home == null) {
      marketEl.textContent = "no line";
    } else {
      const agree = (homeP >= 0.5) === (g.market_prob_home >= 0.5);
      const sp = g.spread_home;
      const spStr = sp == null ? "" : `line ${sp > 0 ? "+" : ""}${sp}`;
      marketEl.innerHTML = `${spStr}<span class="tag ${agree ? "agree" : "disagree"}">${agree ? "agree" : "disagree"}</span>`;
    }

    // Drawer content
    const whyUl = node.querySelector(".why-list");
    (g.why || []).forEach((w) => { const li = document.createElement("li"); li.textContent = w; whyUl.appendChild(li); });

    const teams = g.teams || {};
    node.querySelector(".statcol--away .statcol-team").textContent = g.away_team;
    node.querySelector(".statcol--home .statcol-team").textContent = g.home_team;
    renderStatList(node.querySelector(".statcol--away .stat-list"), teams.away && teams.away.stats, false);
    renderStatList(node.querySelector(".statcol--home .stat-list"), teams.home && teams.home.stats, true);

    // News (from news.json keyed by team), optional
    const newsWrap = node.querySelector(".news");
    const newsUl = node.querySelector(".news-list");
    const items = []
      .concat((state.news[g.away_team] || []).map(x => ({ ...x, team: g.away_team })))
      .concat((state.news[g.home_team] || []).map(x => ({ ...x, team: g.home_team })))
      .slice(0, 6);
    if (items.length) {
      newsWrap.hidden = false;
      items.forEach((it) => {
        const li = document.createElement("li");
        li.innerHTML = `<a href="${it.url}" target="_blank" rel="noopener">${it.title}</a> ` +
          `<span class="news-src">${it.source || it.team || ""}</span>`;
        newsUl.appendChild(li);
      });
    }
    if (g.ai_note) node.querySelector(".ai-body").textContent = g.ai_note;

    // Expand / collapse
    const head = node.querySelector(".game-head");
    const drawer = node.querySelector(".drawer");
    head.addEventListener("click", () => {
      const open = head.getAttribute("aria-expanded") === "true";
      head.setAttribute("aria-expanded", String(!open));
      drawer.hidden = open;
    });

    board.appendChild(node);
  }
}

async function main() {
  const dek = document.getElementById("dek");
  try {
    const [preds, colors, news] = await Promise.all([
      loadJSON("predictions.json"),
      loadJSON("team_colors.json", true),
      loadJSON("news.json", true),
    ]);
    state.games = preds.games || [];
    state.colors = colors || {};
    state.news = (news && news.teams) || news || {};
    const wk = preds.week ? `Week ${preds.week}` : "";
    dek.textContent = `${preds.season || ""} ${wk} — ${state.games.length} games`.trim();
    document.getElementById("stamp").textContent = preds.generated_at ? `Generated ${preds.generated_at}` : "";
    render();
  } catch (err) {
    document.getElementById("board").innerHTML =
      `<p class="error">Couldn't load the board (${err.message}).</p>`;
    dek.textContent = "";
  }
}

document.getElementById("filter").addEventListener("input", (e) => { state.filter = e.target.value; render(); });
main();
