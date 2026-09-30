function el(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value != null) node.textContent = value;
  return node;
}

function numberBox(value, label) {
  const box = el("div", "watch-number");
  box.append(el("strong", "", value), el("span", "", label));
  return box;
}

async function main() {
  const board = document.getElementById("watches");
  try {
    const response = await fetch("upset_watch.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    document.getElementById("week").textContent =
      `${data.season} Week ${data.week} · ${data.games.length} model-picked underdogs · ${data.qualified_count || 0} pass a supported screen`;
    const policy = data.upset_screen;
    if (policy && policy.validation) {
      const validation = policy.validation;
      board.append(el("p", "watch-history",
        `Later-season test of the upset-screen selection: ${validation.wins}/${validation.n} wins ` +
        `(${Math.round(validation.win_rate * 100)}%). ` +
        (validation.supported ? "The screen has historical support; future results still need tracking." :
          "It has not shown that picking these underdogs beats picking the favorites. No strong upset recommendation is established.")));
      if (policy.selected) {
        const r = policy.selected.rule;
        board.append(el("p", "watch-history",
          `Research screen for this season: model probability at least ${Math.round(r.min_probability*100)}%, ` +
          `underdog spread at most ${r.max_spread} points, at least ${r.min_games} earlier FBS games per team. ` +
          `Statistical support requirement: ${r.support === "any" ? "none" : r.support === "stats" ? "stats favor the underdog" : "stats and Elo favor the underdog"}. ` +
          "These conditions are a research filter, not a proven picking rule."));
      }
    }
    if (!data.games.length) {
      board.append(el("p", "empty", "The model did not pick an underdog on this slate."));
    }
    for (const game of data.games) {
      const card = el("article", "watch-card");
      const head = el("div", "watch-head");
      head.append(el("h3", "", game.underdog));
      if (game.crowd_picked_pct != null && game.crowd_picked_pct <= 25)
        head.append(el("span", "watch-pill", "Against the crowd"));
      card.append(head, el("p", "watch-line",
        `${game.away_team} @ ${game.home_team} · ${game.underdog} +${game.spread_for_underdog}`));
      if (game.screening) {
        card.append(el("p", "watch-history", game.screening.label));
        for (const reason of game.screening.reasons || []) card.append(el("p", "watch-line", reason));
      }
      const numbers = el("div", "watch-numbers");
      numbers.append(numberBox(`${Math.round(game.model_prob_underdog * 100)}%`, "Model's underdog win estimate"));
      numbers.append(numberBox(game.crowd_picked_pct == null ? "—" : `${game.crowd_picked_pct}%`,
        "ESPN entries picking this team"));
      const hist = game.historical;
      numbers.append(numberBox(hist.dog_win_rate == null ? "—" : `${Math.round(hist.dog_win_rate * 100)}%`,
        `Earlier underdogs in ${hist.count} comparable games`));
      card.append(numbers, el("h4", "watch-subhead", "What the pregame numbers say"));
      const list = el("ul", "watch-clues");
      for (const clue of game.clues || []) list.append(el("li", "", clue));
      card.append(list);
      card.append(el("p", "watch-history",
        `Historical comparison: ${hist.dog_wins}/${hist.count} underdogs won with ${hist.criteria} ` +
        `(line range ${hist.spread_bucket}). This is a descriptive group, not a second prediction.`));
      board.append(card);
    }
    if (data.crowd_source) document.getElementById("source").textContent = data.crowd_source;
  } catch (error) {
    board.append(el("p", "error", `Couldn't load the upset watch (${error.message}).`));
  }
}
main();
