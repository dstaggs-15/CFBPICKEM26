/* Season record. Only archived picks with final scores enter the per-pick analysis. */
const root = document.getElementById("record-page");

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function section(title, description) {
  const panel = element("section", "record-panel");
  panel.append(element("h2", "", title));
  if (description) panel.append(element("p", "subtle", description));
  return panel;
}

function rate(wins, total) { return total ? `${Math.round(wins / total * 100)}%` : "—"; }
function dateLabel(value) {
  if (!value) return "Date unavailable";
  const date = new Date(`${value.slice(0, 10)}T12:00:00Z`);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric", year: "numeric" });
}

function chartRow(label, wins, losses, alt = false) {
  const total = wins + losses;
  const row = element("div", "chart-row");
  const track = element("div", "chart-track");
  track.setAttribute("role", "img");
  track.setAttribute("aria-label", `${label}: ${wins} wins, ${losses} losses, ${rate(wins, total)}`);
  const fill = element("div", alt ? "chart-fill chart-fill--alt" : "chart-fill");
  fill.style.width = rate(wins, total) === "—" ? "0%" : rate(wins, total);
  track.append(fill);
  row.append(element("span", "chart-label", label), track,
    element("span", "chart-value", `${wins}–${losses} · ${rate(wins, total)}`));
  return row;
}

function metric(label, value, note) {
  const card = element("div", "record-metric");
  card.append(element("span", "metric-label", label), element("strong", "", value), element("small", "", note));
  return card;
}

function renderPickTable(week) {
  const panel = section(`Week ${week.week} · every pick`, `${week.wins}–${week.losses} · picks saved ${dateLabel(week.picks[0].published_at)}`);
  const wrap = element("div", "record-table-wrap");
  const table = element("table", "record-table");
  const thead = element("thead");
  const header = element("tr");
  ["Date", "Matchup / final", "Model pick", "Confidence", "Favorite", "Result"].forEach(label => header.append(element("th", "", label)));
  thead.append(header);
  const tbody = element("tbody");
  for (const p of week.picks) {
    const row = element("tr");
    const date = element("td", "", p.game_date ? dateLabel(p.game_date) : `Picked ${dateLabel(p.published_at)}`);
    if (p.game_date && p.published_at) date.append(element("span", "game-line", `Picked ${dateLabel(p.published_at)}`));
    const matchup = element("td", "", `${p.away_team} @ ${p.home_team}`);
    matchup.append(element("span", "game-line", `Final: ${p.away_points}–${p.home_points}`));
    const correct = p.pick === p.winner;
    const favoriteCorrect = p.market_favorite === p.winner;
    const favorite = element("td", p.market_favorite ? "" : "muted",
      p.market_favorite ? `${p.market_favorite} · ${favoriteCorrect ? "W" : "L"}` : "No line");
    const confidence = p.model_confidence == null ? "—" : `${Math.round(p.model_confidence * 100)}%`;
    row.append(date, matchup, element("td", "", p.pick), element("td", "", confidence),
      favorite, element("td", correct ? "result-win" : "result-loss", correct ? "WIN" : "LOSS"));
    tbody.append(row);
  }
  table.append(thead, tbody);
  wrap.append(table);
  panel.append(wrap);
  if (week.picks.some(p => !p.game_date)) {
    panel.append(element("p", "coverage-note", "Where the game date was not saved, the date column shows when the pick was published."));
  }
  return panel;
}

function render(data) {
  const weeks = (data.weeks || []).filter(w => Number.isInteger(w.wins) && Number.isInteger(w.losses)).sort((a, b) => a.week - b.week);
  if (!weeks.length) throw new Error("No weekly results yet");
  root.replaceChildren();
  const wins = weeks.reduce((n, w) => n + w.wins, 0);
  const losses = weeks.reduce((n, w) => n + w.losses, 0);
  const detailed = weeks.filter(w => Array.isArray(w.picks) && w.picks.length === w.wins + w.losses);
  const picks = detailed.flatMap(w => w.picks);
  const comparable = picks.filter(p => p.market_favorite && p.winner && p.pick);
  const modelWins = comparable.filter(p => p.pick === p.winner).length;
  const favoriteWins = comparable.filter(p => p.market_favorite === p.winner).length;
  const disagreements = comparable.filter(p => p.pick !== p.market_favorite);

  root.append(element("p", "record-intro", `${data.season} season · ${weeks.length} recorded weeks`));
  const hero = element("div", "record-hero");
  hero.append(metric("Overall record", `${wins}–${losses}`, "All recorded weeks"),
    metric("Overall win rate", rate(wins, wins + losses), `${wins + losses} picks`),
    metric("Favorite baseline", `${favoriteWins}–${comparable.length - favoriteWins}`,
      `${comparable.length} games with saved picks and a line`));
  root.append(hero);

  const grid = element("div", "record-grid");
  const weekly = section("Week by week", "Win rate for every recorded week");
  const weeklyRows = element("div", "chart-list");
  weeks.forEach(w => weeklyRows.append(chartRow(`Week ${w.week}${w.source === "reported" ? " *" : ""}`, w.wins, w.losses)));
  weekly.append(weeklyRows);
  if (weeks.some(w => w.source === "reported"))
    weekly.append(element("p", "coverage-note", "* Owner-reported total; the individual picks are not archived."));
  grid.append(weekly);

  const versus = section("Model vs. favorite", `Same ${comparable.length} games, using the saved pregame line`);
  const versusRows = element("div", "chart-list");
  versusRows.append(chartRow("Model", modelWins, comparable.length - modelWins));
  versusRows.append(chartRow("Favorite", favoriteWins, comparable.length - favoriteWins, true));
  versus.append(versusRows);
  if (comparable.length) {
    const modelDisagreeWins = disagreements.filter(p => p.pick === p.winner).length;
    versus.append(element("p", "chart-key", `They disagreed on ${disagreements.length} games: model ${modelDisagreeWins}–${disagreements.length - modelDisagreeWins}, favorite ${disagreements.length - modelDisagreeWins}–${modelDisagreeWins}.`));
  }
  grid.append(versus);
  root.append(grid);

  if (picks.length) {
    const splits = element("div", "record-grid");
    const market = section("Agreement with the line", "How the model's picks did when they matched or opposed the favorite");
    const marketRows = element("div", "chart-list");
    for (const [label, subset] of [
      ["Agreed", comparable.filter(p => p.pick === p.market_favorite)],
      ["Disagreed", disagreements],
    ]) {
      if (subset.length) marketRows.append(chartRow(label, subset.filter(p => p.pick === p.winner).length,
        subset.filter(p => p.pick !== p.winner).length));
    }
    market.append(marketRows);
    splits.append(market);

    const confidence = section("By model confidence", "The saved probability assigned to the picked team");
    const confidenceRows = element("div", "chart-list");
    for (const [label, min, max] of [["50–64%", .5, .65], ["65–79%", .65, .8], ["80%+", .8, 1.01]]) {
      const subset = picks.filter(p => p.model_confidence >= min && p.model_confidence < max);
      if (subset.length) confidenceRows.append(chartRow(label, subset.filter(p => p.pick === p.winner).length,
        subset.filter(p => p.pick !== p.winner).length));
    }
    confidence.append(confidenceRows);
    splits.append(confidence);
    root.append(splits);

    root.append(element("p", "record-callout neutral",
      `These splits cover ${picks.length} saved picks across ${detailed.length} week${detailed.length === 1 ? "" : "s"}. There is not enough pick-level history yet to identify a dependable “trust it more” situation.`));
    for (const week of detailed.slice().reverse()) root.append(renderPickTable(week));
  }
  if (detailed.length !== weeks.length) {
    root.append(element("p", "coverage-note", "Pick-by-pick charts exclude weeks without saved picks. Their reported wins and losses still count toward the overall record."));
  }
}

fetch("results.json", { cache: "no-store" })
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(render)
  .catch(() => { root.replaceChildren(element("p", "error", "Couldn't load the model record. Please try again shortly.")); });
