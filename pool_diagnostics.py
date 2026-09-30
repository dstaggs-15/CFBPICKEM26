"""Grade published pool picks, keeping them distinct from retrospective replays."""
import json
from pathlib import Path
import numpy as np


def report(payload):
    rows = []
    for week in payload.get('weeks', []):
        g = week.get('picks', [])
        comparable = [x for x in g if x.get('market_favorite')]
        disagreements = [x for x in comparable if x['pick'] != x['market_favorite']]
        rows.append({'week': week['week'], 'wins': week['wins'], 'losses': week['losses'],
                     'source': week.get('source'), 'favorite_comparison_n': len(comparable),
                     'favorite_wins': sum(x['market_favorite'] == x['winner'] for x in comparable),
                     'disagreements': len(disagreements),
                     'disagreement_wins': sum(x['pick'] == x['winner'] for x in disagreements)})
    wins = sum(w['wins'] for w in rows)
    n = sum(w['wins']+w['losses'] for w in rows)
    # Wilson interval describes finite-sample uncertainty, not future guarantees.
    z = 1.96
    if n:
        p = wins/n
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        interval = [round(center-half, 4), round(center+half, 4)]
    else:
        interval = None
    return {'season': payload['season'], 'wins': wins, 'games': n,
            'accuracy': wins/n if n else None, 'wilson_95_interval': interval,
            'weeks': rows,
            'note': 'Published pool picks only; reported weeks lack individual archived picks. ' 
                    'Historical full-FBS accuracy is a different evaluation population.'}


def main():
    result = report(json.loads(Path('docs/results.json').read_text()))
    Path('docs/model_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
