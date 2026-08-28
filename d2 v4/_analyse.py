import json, collections, statistics
d = json.load(open('d2 v3/d2_fp8_expert_report.json', encoding='utf-8'))
e = d['experts']
print('total tenseurs:', len(e))
worst = sorted(e, key=lambda x: x['snr_db'])[:15]
print('\nTOP-15 EXPERTS LES PLUS SENSIBLES (FP8 -> GGUF):')
for w in worst:
    print(f"  blk.{w['layer']:>2} exp.{w['expert']:>3} {w['type']:>5} SNR={w['snr_db']} dB freq={w['freq']*100:.2f}%")
lay = collections.defaultdict(list)
for r in e:
    lay[r['layer']].append(r['snr_db'])
means = {l: statistics.mean(v) for l, v in lay.items()}
lo = sorted(means.items(), key=lambda x: x[1])[:5]
hi = sorted(means.items(), key=lambda x: -x[1])[:3]
print('\ncouches moins bien preservees :', [(f'blk{l}', round(m, 1)) for l, m in lo])
print('couches les mieux preservees  :', [(f'blk{l}', round(m, 1)) for l, m in hi])
# croisement freq x snr : experts TRES utilises ET peu preserves
cross = sorted(e, key=lambda r: -(r['freq'] * 1000 - r['snr_db']))[:10]
print('\ncroisement critique (fort trafic + SNR bas):')
for r in cross[:6]:
    print(f"  blk.{r['layer']:>2} exp.{r['expert']:>3} {r['type']} SNR={r['snr_db']} freq={r['freq']*100:.2f}%")
