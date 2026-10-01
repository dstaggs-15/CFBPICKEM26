/* CFB Pick'em Model — board renderer with expandable per-game breakdown. */
const state = { games: [], colors: {}, news: {}, rankings: {}, filter: "" };
const RANKINGS_URL = "https://dstaggs-15.github.io/cfbranking/data/rankings.json";

async function loadJSON(path, optional = false) {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) { if (optional) return null; throw new Error(`${path} → ${res.status}`); }
    return await res.json();
  } catch (e) { if (optional) return null; throw e; }
}

const pct = (p) => `${Math.round(p * 100)}%`;
const rankLabel = (r) => (r === null || r === undefined) ? "" : `#${r}`;

function showRank(team, element) {
  element.textContent = rankLabel(state.rankings[team]);
}

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

function renderContributions(panel, game) {
  const terms = game.model_log_odds_terms;
  const intercept = game.model_log_odds_intercept;
  if (!terms || !Number.isFinite(intercept) || !Object.values(terms).every(Number.isFinite)) return;
  const groups = [
    { label: "Current matchup stats", color: "var(--factor-stats)", magnitude: 0, net: 0 },
    { label: "Elo", color: "var(--factor-elo)", magnitude: 0, net: 0 },
    { label: "Venue, rest & postseason", color: "var(--factor-context)", magnitude: 0, net: 0 },
    { label: "Starting score & data availability", color: "var(--factor-baseline)", magnitude: Math.abs(intercept), net: intercept },
  ];
  for (const [name, value] of Object.entries(terms)) {
    const index = name === "elo_home_prob" ? 1 :
      ["neutral_site", "rest_diff", "is_postseason"].includes(name) ? 2 :
      name.startsWith("missingindicator_") ? 3 : 0;
    groups[index].magnitude += Math.abs(value);
    groups[index].net += value;
  }
  const total = groups.reduce((sum, group) => sum + group.magnitude, 0);
  if (!(total > 0)) return;
  // Largest remainders make the displayed whole percentages total exactly 100.
  const shares = groups.map(group => group.magnitude / total * 100);
  const percentages = shares.map(Math.floor);
  const order = shares.map((share, index) => ({ index, remainder: share - percentages[index] }))
    .sort((a, b) => b.remainder - a.remainder);
  const remaining = 100 - percentages.reduce((a, b) => a + b, 0);
  for (let i = 0; i < remaining; i++) percentages[order[i].index]++;
  let end = 0;
  const slices = groups.map((group, index) => {
    const start = end;
    end += shares[index];
    return `${group.color} ${start}% ${end}%`;
  });
  const donut = panel.querySelector(".contribution-donut");
  donut.style.background = `conic-gradient(${slices.join(",")})`;
  const legend = panel.querySelector(".contribution-legend");
  const descriptions = [];
  groups.forEach((group, index) => {
    const direction = Math.abs(group.net) < .00001 ? "Balanced" :
      `Net toward ${group.net > 0 ? game.home_team : game.away_team}`;
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "contribution-swatch";
    swatch.style.background = group.color;
    swatch.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    const label = document.createElement("span");
    const shownShare = percentages[index] === 0 && shares[index] > 0 ? `${shares[index].toFixed(2)}%` : `${percentages[index]}%`;
    label.textContent = `${group.label} · ${shownShare}`;
    const detail = document.createElement("small");
    detail.textContent = direction;
    text.append(label, detail);
    item.append(swatch, text);
    legend.append(item);
    descriptions.push(`${group.label}: ${percentages[index]}% of term magnitude, ${direction}`);
  });
  donut.setAttribute("aria-label", `${game.away_team} at ${game.home_team}. ${descriptions.join(". ")}. These are not win probabilities.`);
  const explanation = document.createElement("div");
  explanation.className = "contribution-explanation";
  const paragraph = text => {
    const p = document.createElement("p");
    p.textContent = text;
    explanation.append(p);
  };
  const sigmoid = value => 1 / (1 + Math.exp(-value));
  const score = intercept + Object.values(terms).reduce((sum, value) => sum + value, 0);
  paragraph("Why the slices change: the fitted formula stays the same, but each matchup has different stats and Elo. Each slice is that group's share of the sizes of all fitted terms, including the starting score. A bigger slice means a larger term in this game; it is not a win chance or a fixed model weight. A slice can also shrink when another group's terms grow.");
  paragraph(`The starting score alone gives ${game.home_team} ${(sigmoid(intercept) * 100).toFixed(1)}%. The model adds the adjustments below to reach its forecast. Directions compare each input with its training average, not with two equal teams.`);
  const eloInput = game.model_input_values?.elo_home_prob;
  const eloAverage = game.model_reference_values?.elo_home_prob;
  if (Number.isFinite(eloInput) && Number.isFinite(terms.elo_home_prob)) {
    const eloFav = eloInput >= .5 ? game.home_team : game.away_team;
    const eloFavP = eloInput >= .5 ? eloInput : 1 - eloInput;
    let text = `Elo alone favors ${eloFav} at ${(eloFavP * 100).toFixed(2)}%, including venue. `;
    if (Number.isFinite(eloAverage)) text += `Its home-team estimate is ${(eloInput * 100).toFixed(2)}%, compared with ${(eloAverage * 100).toFixed(2)}% across training games. `;
    if (shares[1] < .5) text += `Elo's slice is only ${shares[1].toFixed(2)}% because its input is close to the training average. Elo is still used; this does not mean the teams have equal ratings. `;
    text += `The chart's direction describes the adjustment from that average, so it can differ from the team Elo alone favors.`;
    paragraph(text);
  }
  groups.forEach((group, index) => {
    if (index === 3) return;
    const without = sigmoid(score - group.net);
    const change = (sigmoid(score) - without) * 100;
    const team = change >= 0 ? game.home_team : game.away_team;
    paragraph(`${group.label}: holding the other inputs fixed, this group's combined adjustment moves the forecast ${Math.abs(change).toFixed(2)} percentage points toward ${team}, compared with putting these inputs at their training averages.${index === 0 ? " The slice counts each stat's size, while this change combines them; stats pointing in opposite directions can cancel." : ""}`);
  });
  paragraph("Those percentage-point changes are separate comparisons and do not add up. The model combines all terms first, then converts the total into a win probability.");
  panel.append(explanation);
  panel.hidden = false;
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
      showRank(g.away_team, node.querySelector(".team--away .rank"));
      showRank(g.home_team, node.querySelector(".team--home .rank"));
      node.querySelector(".market").textContent = g.error ? "not found" : "";
      board.appendChild(node);
      continue;
    }

    const homeP = g.model_prob_home, awayP = 1 - homeP;
    const homeFav = homeP >= 0.5;
    const hc = state.colors[g.home_team];   // undefined if not in team_colors.json
    const ac = state.colors[g.away_team];

    const away = node.querySelector(".team--away");
    showRank(g.away_team, away.querySelector(".rank"));
    const awayName = away.querySelector(".name");
    awayName.textContent = g.away_team;
    const home = node.querySelector(".team--home");
    showRank(g.home_team, home.querySelector(".rank"));
    const homeName = home.querySelector(".name");
    homeName.textContent = g.home_team;

    // Name of the pick: real team color if we have one, else the red fallback class.
    const pickName = homeFav ? homeName : awayName;
    const pickColor = homeFav ? hc : ac;
    if (pickColor) pickName.style.color = pickColor.primary;
    else pickName.classList.add("name--pick");

    if (g.neutral) node.querySelector(".at").textContent = "vs";

    const pctAway = node.querySelector(".pct--away");
    const pctHome = node.querySelector(".pct--home");
    pctAway.textContent = `${100 - Math.round(homeP * 100)}%`;
    pctHome.textContent = pct(homeP);
    (homeFav ? pctHome : pctAway).classList.add("lead");

    // Meter: real team colors when both/either are known, else red-vs-graphite.
    const aFill = node.querySelector(".meter--away"), hFill = node.querySelector(".meter--home");
    aFill.style.width = `${awayP * 100}%`;
    hFill.style.width = `${homeP * 100}%`;
    if (ac) aFill.style.background = ac.primary; else aFill.classList.add(homeFav ? "is-dim" : "is-pick");
    if (hc) hFill.style.background = hc.primary; else hFill.classList.add(homeFav ? "is-pick" : "is-dim");

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
    renderContributions(node.querySelector(".contribution-panel"), g);
    const whyUl = node.querySelector(".why-list");
    (g.why || []).forEach((w) => { const li = document.createElement("li"); li.textContent = w; whyUl.appendChild(li); });

    const historical = g.historical_profile;
    if (historical && historical.available) {
      (historical.examples || []).forEach((example) => {
        const li = document.createElement("li");
        li.textContent = `Similar profile: ${example.season} ${example.away_team} at ${example.home_team}, ` +
          `${example.away_points}–${example.home_points}. Profile distance ${example.distance} (lower is closer).`;
        whyUl.appendChild(li);
      });
    }
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
    const [preds, colors, news, ranks] = await Promise.all([
      loadJSON("predictions.json"),
      loadJSON("team_colors.json", true),
      loadJSON("news.json", true),
      loadJSON(RANKINGS_URL, true),
    ]);
    state.games = preds.games || [];
    state.colors = colors || {};
    state.news = (news && news.teams) || news || {};
    // The other site publishes only a Top 25. Missing teams stay unnumbered.
    state.rankings = Object.fromEntries(
      ((ranks && ranks.season === preds.season && ranks.top25) || [])
        .filter(t => t.team && Number.isInteger(t.rank) && t.rank >= 1 && t.rank <= 25)
        .map(t => [t.team, t.rank]));
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
