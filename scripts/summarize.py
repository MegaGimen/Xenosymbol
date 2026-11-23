"""Summarize observed paired outcomes including the uncompressed baseline."""
import argparse
import json
import random
import statistics
from pathlib import Path


def mean(rows, key): return statistics.mean(r[key] for r in rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--output',required=True)
    args=p.parse_args(); root=Path(args.run)
    rows=[json.loads(l) for l in (root/'observations.jsonl').read_text().splitlines()]
    config=json.loads((root/'config.json').read_text())
    report={'run':str(root),'lambda':config['length_weight'],'groups':{}}
    for phase,step in sorted({(r['phase'],r['step']) for r in rows if r['phase']!='train' and r['phase']!='sanity_identity'}):
        rs=[r for r in rows if r['phase']==phase and r['step']==step]
        report['groups'][f'{phase}@{step}']={
            'n':len(rs),'objective':-mean(rs,'reward'),'distance':mean(rs,'distance'),
            'pre_distance':mean(rs,'pre_distance'),'message_tokens':mean(rs,'length'),
            'original_tokens':mean(rs,'original_length'),
            'mean_per_prompt_length_ratio':mean(rs,'length_ratio'),
            'total_token_ratio':sum(r['length'] for r in rs)/sum(r['original_length'] for r in rs),
            'original_objective':config['length_weight']*mean(rs,'original_length'),
            'exact_output_match_rate':mean(rs,'exact_output_match')}
    before={r['id']:r for r in rows if r['phase']=='test_initial'}
    after={r['id']:r for r in rows if r['phase']=='test_best'}
    if before and after:
        assert before.keys()==after.keys()
        paired={}
        for key in ['reward','distance','length']:
            diffs=[after[k][key]-before[k][key] for k in before]
            rng=random.Random(73498)
            samples=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(10000))
            paired[key]={'mean_best_minus_initial':statistics.mean(diffs),
                         'paired_bootstrap_95_percent_interval':[samples[249],samples[9749]]}
        report['paired_test']=paired
        report['uncertainty_note']=f'Only {len(before)} test tasks; intervals are exploratory, not evidence of broad language generalization.'
        summary_path=root/'summary.json'
        if summary_path.exists() and json.loads(summary_path.read_text()).get('best_step')==0:
            report['selection_note']='Selected checkpoint is the initialization itself. Identical test outcomes and zero paired intervals do not demonstrate learning or perfect task performance.'
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
