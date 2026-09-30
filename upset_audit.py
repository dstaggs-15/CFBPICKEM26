"""Audit upset patterns using earlier-season models and later-season screening."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from model import JointStatisticalModel
import schema
from upset_policy import choose_rule, mask, summary


def build_evidence(profiles):
    outputs=[]
    fbs=profiles.home_classification.str.lower().eq('fbs') & profiles.away_classification.str.lower().eq('fbs')
    played=profiles.home_points.notna() & profiles.away_points.notna()
    for year in sorted(profiles.season.unique()):
        if year<2017:
            continue
        train=profiles[fbs & played & profiles.season.lt(year)]
        test=profiles[fbs & played & profiles.season.eq(year) & profiles.spread_home.notna() & profiles.spread_home.ne(0)].copy()
        if test.empty:
            continue
        fit=JointStatisticalModel().fit(train)
        home_probability=fit.predict_proba(test)
        terms=fit.contributions(test)
        orientation=np.where(test.spread_home.gt(0),1,-1)
        test['dog_prob']=np.where(orientation==1,home_probability,1-home_probability)
        test['line_abs']=test.spread_home.abs()
        test['min_games']=test.joint_min_games
        test['stats_support']=(terms[schema.JOINT_PROFILE_FEATURES].sum(axis=1)*orientation).gt(0) & test[schema.JOINT_PROFILE_FEATURES].notna().all(axis=1)
        test['elo_support']=np.where(orientation==1,test.elo_home_prob,1-test.elo_home_prob)>.5
        test['dog_win']=test.home_points.gt(test.away_points).eq(test.spread_home.gt(0))
        outputs.append(test[['game_id','season','week','date','home_team','away_team','dog_prob','line_abs','min_games','stats_support','elo_support','dog_win']])
        print('Finished honest upset evidence',year,flush=True)
    return pd.concat(outputs,ignore_index=True)


def audit(evidence, season=2026):
    older=evidence[evidence.season.lt(season)].copy()
    folds=[];selected=[]
    for year in range(2020,season):
        choice=choose_rule(older[older.season.lt(year)])
        test=older[older.season.eq(year)]
        chosen=test[mask(test,choice['rule'])] if choice else test.iloc[:0]
        selected.append(chosen)
        folds.append(dict(season=year,selected=choice,later_season_results=summary(chosen)))
    validation=summary(pd.concat(selected,ignore_index=True))
    validation['supported']=bool(validation['n']>=100 and validation['confidence_interval'][0] is not None and validation['confidence_interval'][0]>.5)
    policy=dict(season=season,model_version='joint-stats-elo-v3',selected=choose_rule(older),
                validation=validation,folds=folds,
                note='Screen settings are selected only from older season-held-out predictions. Retrospective research; frozen future picks remain the prospective test.')
    groups=[]
    picked=older[older.dog_prob.ge(.5)]
    for label,frame in [('All model-picked underdogs',picked),('Stats and Elo both support the underdog',picked[picked.stats_support & picked.elo_support]),('Stats support without Elo support',picked[picked.stats_support & ~picked.elo_support]),('Elo support without stats support',picked[~picked.stats_support & picked.elo_support])]:
        groups.append(dict(group=label,**summary(frame)))
    for lo,hi in [(0,3),(3,7),(7,14),(14,1000)]:
        groups.append(dict(group=f'Spread above {lo} through {hi}',**summary(picked[picked.line_abs.gt(lo)&picked.line_abs.le(hi)])))
    for lo,hi in [(.5,.55),(.55,.6),(.6,1.01)]:
        groups.append(dict(group=f'Model confidence {lo:.0%} to under {hi:.0%}',**summary(picked[picked.dog_prob.ge(lo)&picked.dog_prob.lt(hi)])))
    for lo,hi in [(0,3),(3,5),(5,1000)]:
        groups.append(dict(group=f'Prior FBS games {lo} to under {hi}',**summary(picked[picked.min_games.ge(lo)&picked.min_games.lt(hi)])))
    return policy,groups


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--if-missing',action='store_true')
    args=parser.parse_args()
    profiles=pd.read_parquet('data/derived/joint_training.parquet')
    season=int(profiles.season.max())
    target=Path(f'historicals/model_audits/{season}-upset-policy.json')
    if args.if_missing and target.exists():
        print('Using frozen season upset screen:',target)
        return
    evidence=build_evidence(profiles)
    evidence.to_parquet('data/derived/upset_evidence.parquet',index=False)
    policy,groups=audit(evidence,season=season)
    target.write_text(json.dumps(policy,indent=2)+'\n')
    Path(f'historicals/model_audits/{season}-upset-groups.json').write_text(json.dumps(groups,indent=2)+'\n')
    print(json.dumps({'selected':policy['selected'],'validation':policy['validation'],'groups':groups},indent=2),flush=True)


if __name__=='__main__':
    main()
