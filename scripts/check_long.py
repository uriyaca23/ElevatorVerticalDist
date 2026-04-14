import sys, json
sys.stdout.reconfigure(encoding='utf-8')
r=json.load(open('evaluation_output/kinematics/results.json'))
a=[x for x in r['bar_ilan_algo_a'] if not x['rejected']]
b=[x for x in r['bar_ilan_algo_b'] if not x['rejected']]
import numpy as np
print(f"AlgA: {len(a)} accepted, MAE={np.mean([x['error'] for x in a]):.2f}, Med={np.median([x['error'] for x in a]):.2f}")
print(f"AlgB: {len(b)} accepted, MAE={np.mean([x['error'] for x in b]):.2f}, Med={np.median([x['error'] for x in b]):.2f}")
