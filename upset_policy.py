"""Time-tested outright-upset screening; never changes the win-model forecast."""
import math
import numpy as np


def wilson(wins, n):
    if not n:
        return [None, None]
    z=1.959963984540054
    p=wins/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [middle-half,middle+half]


def mask(frame, rule):
    selected=(frame.dog_prob.ge(rule['min_probability']) &
              frame.line_abs.le(rule['max_spread']) &
              frame.min_games.ge(rule['min_games']))
    if rule['support'] in ('stats','both'):
        selected &= frame.stats_support.eq(True)
    if rule['support']=='both':
        selected &= frame.elo_support.eq(True)
    return selected


def summary(frame):
    n=len(frame);wins=int(frame.dog_win.sum())
    return dict(n=n,wins=wins,win_rate=wins/n if n else None,
                confidence_interval=wilson(wins,n),
                seasons=sorted(int(y) for y in frame.season.unique()))


def choose_rule(older):
    """Small fixed grid; selection only sees earlier held-out forecasts.

    Require at least 100 games across three seasons. Selection by lower Wilson
    bound favors repeatable samples; later-season testing evaluates the whole
    selection procedure, including its multiple comparisons.
    """
    choices=[]
    for probability in [.5,.55,.60]:
        for spread in [3.,7.,14.,1000.]:
            for support in ['any','stats','both']:
                for count in [0,3,5]:
                    rule=dict(min_probability=probability,max_spread=spread,
                              support=support,min_games=count)
                    evidence=summary(older[mask(older,rule)])
                    if evidence['n']>=100 and len(evidence['seasons'])>=3:
                        choices.append((rule,evidence))
    if not choices:
        return None
    rule,evidence=max(choices,key=lambda item:(item[1]['confidence_interval'][0],item[1]['n']))
    return dict(rule=rule,selection_evidence=evidence)


def assess(probability, spread, stats_support, elo_support, min_games, policy):
    """Apply a frozen historical screen without treating it as a probability."""
    if not policy or not policy.get('selected'):
        return dict(status='unvalidated',label='No tested upset screen available',qualifies=False)
    rule=policy['selected']['rule']
    checks=[(probability>=rule['min_probability'],f"Model estimate must be at least {rule['min_probability']:.0%}."),
            (spread<=rule['max_spread'],f"Underdog spread must be at most {rule['max_spread']:g} points."),
            (min_games>=rule['min_games'],f"Each team needs at least {rule['min_games']} earlier FBS games.")]
    if rule['support'] in ('stats','both'):
        checks.append((stats_support is True,'The combined statistical contributions must favor the underdog.'))
    if rule['support']=='both':
        checks.append((elo_support is True,'Pregame Elo must also favor the underdog.'))
    qualifies=all(ok for ok,_ in checks)
    validated=policy.get('validation',{}).get('supported',False)
    if qualifies and validated:
        status,label='qualified','Passes the historically tested upset screen'
    elif qualifies:
        status,label='unvalidated','Matches a research screen; reliability not established'
    else:
        status,label='watch_only','Watch only — does not pass the upset screen'
    return dict(status=status,label=label,qualifies=qualifies,
                reasons=[message for ok,message in checks if not ok],
                validation=policy.get('validation'))
