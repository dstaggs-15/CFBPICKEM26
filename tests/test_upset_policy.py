import unittest
import pandas as pd
from upset_policy import assess, choose_rule, wilson
from upset_audit import audit
from upset_watch import build
import schema


class UpsetPolicyTests(unittest.TestCase):
    def evidence(self):
        return pd.DataFrame([dict(season=year,dog_prob=.56,line_abs=2.,min_games=6,
            stats_support=True,elo_support=True,dog_win=i%4!=0)
            for year in range(2017,2027) for i in range(120)])

    def test_current_and_future_outcomes_cannot_choose_this_seasons_rule(self):
        d=self.evidence();before,_=audit(d,2026)
        d.loc[d.season.ge(2026),['dog_win','dog_prob']]=[False,.99]
        after,_=audit(d,2026)
        self.assertEqual(before,after)

    def test_matching_an_unvalidated_rule_never_becomes_a_recommendation(self):
        policy={'selected':{'rule':{'min_probability':.55,'max_spread':3.,'support':'both','min_games':5}},'validation':{'supported':False}}
        result=assess(.65,2.,True,True,6,policy)
        self.assertTrue(result['qualifies']);self.assertEqual(result['status'],'unvalidated')
        policy['validation']['supported']=True
        self.assertEqual(assess(.65,2.,True,True,6,policy)['status'],'qualified')
        missed=assess(.51,5.5,True,True,3,policy)
        self.assertEqual(missed['status'],'watch_only');self.assertEqual(len(missed['reasons']),3)

    def test_small_samples_cannot_select_a_rule(self):
        self.assertIsNone(choose_rule(self.evidence().head(80)))
        self.assertEqual(wilson(0,0),[None,None])

    def test_watch_uses_joint_evidence_and_leaves_original_pick_intact(self):
        row=dict(game_id='1',season=2026,home_team='B',away_team='A',home_classification='fbs',away_classification='fbs',home_points=float('nan'),away_points=float('nan'),spread_home=5.5,elo_home_prob=.6,joint_min_games=3,
            home_off_vs_away_def_ppa=.2,away_off_vs_home_def_ppa=.1,**{c:.1 for c in schema.JOINT_PROFILE_FEATURES})
        game=dict(game_id='1',home_team='B',away_team='A',pick='B',spread_home=5.5,model_prob_home=.514,model_log_odds_terms={c:.1 for c in schema.JOINT_PROFILE_FEATURES})
        predictions=dict(season=2026,week=5,model_version='joint-stats-elo-v3',games=[game])
        choice=choose_rule(self.evidence())
        policy=dict(selected=choice,validation={'supported':False})
        watch=build(predictions,pd.DataFrame([row]),{},None,policy)
        self.assertEqual(watch['qualified_count'],0)
        self.assertEqual(watch['games'][0]['screening']['status'],'watch_only')
        self.assertEqual(game['pick'],'B')
