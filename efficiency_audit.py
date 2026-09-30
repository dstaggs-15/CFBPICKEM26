"""Compare direct team profiles without reputation ratings on held-out seasons."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from model import V1Model
from profile_analogs import PROFILE_FEATURES
from backtest import _metrics
import schema

FEATURES = schema.CONTEXT_FEATURES + PROFILE_FEATURES


def main():
    d = pd.read_parquet('data/derived/training.parquet')
    out = pd.read_parquet('data/derived/profile_oos.parquet')
    for year in sorted(out.season.unique()):
        train = d[(d.season < year) & d.home_points.notna() & d.away_points.notna()]
        train = train[train.home_classification.str.lower().eq('fbs') & train.away_classification.str.lower().eq('fbs')]
        test = out[out.season.eq(year)]
        for name, estimator in [('profile_linear', make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=2000))),
                                ('profile_boosted', V1Model(features=FEATURES))]:
            if name == 'profile_linear':
                estimator.fit(train[FEATURES],train.home_points.gt(train.away_points))
            else:
                estimator.fit(train)
            out.loc[test.index,'p_'+name] = estimator.predict_proba(test[FEATURES])[:,1] if name == 'profile_linear' else estimator.predict_proba(test)
        print('Finished',year,flush=True)
    out.to_parquet('data/derived/efficiency_oos.parquet',index=False)
    for label, part in [('2017_2025',out[out.season.le(2025)]),('2023_2025',out[out.season.between(2023,2025)]),('2026_partial',out[out.season.eq(2026)])]:
        print(label)
        for name in ['original','corrected','logistic','neighbors','profile_linear','profile_boosted']:
            print(name,_metrics(part.home_points.gt(part.away_points),part['p_'+name]))


if __name__ == '__main__':
    main()
