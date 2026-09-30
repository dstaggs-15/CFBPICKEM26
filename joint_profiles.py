"""Current-season opponent-adjusted matchup profiles for the joint predictor.

Each weekly snapshot solves offense and defense together using only earlier
completed FBS games. Alpha is fixed at 1 from the prior-season selection audit.
No team rating is carried from one season to the next.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

METRICS = ['off_ppa', 'off_success', 'off_explosive', 'points']
PROFILE = [f'joint_{m}_edge' for m in METRICS]
CONTEXT = ['neutral_site', 'rest_diff', 'is_postseason']


def build_profiles(base, advanced, alpha):
    """Fit both sides together using only games before the week's Monday.

    Aggregate game observations receive equal weight. Ridge shrinks small
    samples toward this season's mean. Alpha is in game units, not play units.
    """
    out = base.copy()
    for column in PROFILE + ['joint_min_games']:
        out[column] = np.nan
    out['date'] = pd.to_datetime(out.date, utc=True)
    out['_cutoff'] = out.date.dt.normalize() - pd.to_timedelta(out.date.dt.weekday, unit='D')
    fbs = out.home_classification.str.lower().eq('fbs') & out.away_classification.str.lower().eq('fbs')
    adv = advanced.set_index(['game_id', 'team'])
    if not adv.index.is_unique:
        raise ValueError('Duplicate game/team advanced-stat observations')
    for season, season_games in out[fbs].groupby('season'):
        teams = sorted(set(season_games.home_team) | set(season_games.away_team))
        lookup = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        for cutoff, target in season_games.groupby('_cutoff'):
            past = season_games[(season_games.date < cutoff) & season_games.home_points.notna() & season_games.away_points.notna()]
            counts = pd.concat([past.home_team, past.away_team]).value_counts()
            # X models attacking team + defending opponent + signed home field.
            rows, values = [], []
            for g in past.itertuples():
                for attack, defense, points, venue in [(g.home_team,g.away_team,g.home_points,1), (g.away_team,g.home_team,g.away_points,-1)]:
                    x = np.zeros(2*n+1)
                    x[lookup[attack]] = 1
                    x[n+lookup[defense]] = 1
                    x[-1] = 0 if g.neutral_site else venue
                    a = adv.loc[(g.game_id,attack)] if (g.game_id,attack) in adv.index else {}
                    rows.append(x)
                    values.append([a.get(m,np.nan) for m in METRICS[:-1]]+[points])
            for idx, g in target.iterrows():
                out.loc[idx,'joint_min_games'] = min(counts.get(g.home_team,0),counts.get(g.away_team,0))
            if not rows:
                continue
            X, Y = np.array(rows), np.array(values)
            for j, metric in enumerate(METRICS):
                valid = np.isfinite(Y[:,j])
                if valid.sum() < 4:
                    continue
                fit = Ridge(alpha=alpha).fit(X[valid],Y[valid,j])
                for idx, g in target.iterrows():
                    if min(counts.get(g.home_team,0),counts.get(g.away_team,0)) < 2:
                        continue
                    h, a = lookup[g.home_team], lookup[g.away_team]
                    out.loc[idx,f'joint_{metric}_edge'] = (fit.coef_[h] + fit.coef_[n+a] - fit.coef_[a] - fit.coef_[n+h] + (0 if g.neutral_site else 2*fit.coef_[-1]))
    return out.drop(columns='_cutoff')
