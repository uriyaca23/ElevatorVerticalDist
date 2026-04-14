import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
r = json.load(open('evaluation_output/kinematics/results.json'))

# Bar-Ilan AlgA only
ba = r['bar_ilan_algo_a']
ba_acc = [x for x in ba if not x['rejected']]
print(f"Bar-Ilan AlgA: {len(ba_acc)}/{len(ba)} acc, MAE={np.mean([x['error'] for x in ba_acc]):.2f}, Med={np.median([x['error'] for x in ba_acc]):.2f}")

# ADVIO AlgA only
av = r['advio_algo_a']
av_acc = [x for x in av if not x['rejected']]
print(f"ADVIO AlgA: {len(av_acc)}/{len(av)} acc, MAE={np.mean([x['error'] for x in av_acc]):.2f}, Med={np.median([x['error'] for x in av_acc]):.2f}")

# Combined
all_acc = ba_acc + av_acc
print(f"Combined AlgA: {len(all_acc)}/{len(ba)+len(av)} acc, MAE={np.mean([x['error'] for x in all_acc]):.2f}, Med={np.median([x['error'] for x in all_acc]):.2f}")
