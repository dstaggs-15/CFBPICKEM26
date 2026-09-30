"""Research only: current-season simultaneous opponent ratings, dated snapshots.

Team names identify this season's schedule equations; they are never inputs to
the cross-season winner model. No previous-season team rating is carried over.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from backtest import _metrics

from joint_profiles import METRICS, PROFILE, CONTEXT, build_profiles


def main():
    base = pd.read_parquet('data/derived/training.parquet')
    advanced = pd.read_parquet('data/raw/advanced_raw.parquet')
    existing = pd.read_parquet('data/derived/profile_oos.parquet')
    output = existing[['game_id','season','week','home_points','away_points','p_original','p_logistic','market_home_prob']].copy()
    for alpha in [1.,4.,16.]:
        cache = Path(f'data/derived/joint_profiles_{alpha:g}.parquet')
        if cache.exists():
            profiles = pd.read_parquet(cache)
        else:
            profiles = build_profiles(base,advanced,alpha)
            profiles.to_parquet(cache,index=False)
        fbs = profiles.home_classification.str.lower().eq('fbs') & profiles.away_classification.str.lower().eq('fbs')
        played = profiles.home_points.notna() & profiles.away_points.notna()
        for year in sorted(output.season.unique()):
            train = profiles[fbs & played & profiles.season.lt(year)]
            test = profiles.set_index('game_id').loc[output[output.season.eq(year)].game_id]
            for label, cols, learner in [
                ('stats',PROFILE+CONTEXT,LogisticRegression(C=.1,max_iter=2000)),
                ('stats_elo',PROFILE+CONTEXT+['elo_home_prob'],LogisticRegression(C=.1,max_iter=2000)),
                ('similar',PROFILE+CONTEXT,KNeighborsClassifier(n_neighbors=75,weights='distance')),
            ]:
                fit = make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),learner)
                fit.fit(train[cols],train.home_points.gt(train.away_points))
                output.loc[output.season.eq(year),f'p_{label}_{alpha:g}'] = fit.predict_proba(test[cols])[:,1]
        print('Finished alpha',alpha,flush=True)
    # Second-level selection: choose shrinkage and blending only on earlier
    # held-out predictions. The target season cannot select its own settings.
    selections=[]
    for year in sorted(output.season.unique()):
        if year < 2020:
            continue
        older=output[output.season.lt(year)]
        target=output[output.season.eq(year)]
        y=older.home_points.gt(older.away_points)
        for family in ['stats','stats_elo']:
            cols=[f'p_{family}_{a:g}' for a in [1.,4.,16.]]
            chosen=min(cols,key=lambda c:np.mean((older[c]-y)**2))
            output.loc[target.index,'p_selected_'+family]=target[chosen]
            selections.append(dict(season=int(year),family=family,chosen=chosen))
        choices=[(f'p_stats_{a:g}',w) for a in [1.,4.,16.] for w in np.arange(0,1.01,.1)]
        c,w=min(choices,key=lambda cw:np.mean(((1-cw[1])*older.p_logistic+cw[1]*older[cw[0]]-y)**2))
        output.loc[target.index,'p_prior_selected_blend']=(1-w)*target.p_logistic+w*target[c]
        selections.append(dict(season=int(year),family='blend',stats_column=c,stats_weight=float(w)))
    Path('data/derived/joint_selection.json').write_text(json.dumps(selections,indent=2))
    output.to_parquet('data/derived/joint_profile_oos.parquet',index=False)
    records=[]
    for subset, part in [('2017_2025',output[output.season.le(2025)]),('2023_2025',output[output.season.between(2023,2025)]),('2026_partial',output[output.season.eq(2026)])]:
        for col in output.filter(regex='^p_').columns:
            if part[col].isna().any():
                continue
            record={'subset':subset,'model':col,**_metrics(part.home_points.gt(part.away_points),part[col])}
            records.append(record)
            print(record,flush=True)
    Path('data/derived/joint_profile_metrics.json').write_text(json.dumps(records,indent=2))
    recent=output[output.season.between(2020,2025)]
    for c in ['p_logistic','p_selected_stats','p_selected_stats_elo','p_prior_selected_blend']:
        print('prior-season selection 2020_2025',c,_metrics(recent.home_points.gt(recent.away_points),recent[c]),flush=True)


if __name__ == '__main__':
    main()
