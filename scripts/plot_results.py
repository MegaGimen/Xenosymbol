"""Static research figures, generated only from actual run records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run'); p.add_argument('--analysis'); p.add_argument('--output',default='reports')
    args=p.parse_args(); root=Path(args.run); out=Path(args.output); out.mkdir(exist_ok=True,parents=True)
    events=read(root/'events.jsonl')
    val=[x for x in events if x['event']=='evaluation' and x['phase']=='validation']
    train=[x for x in events if x['event']=='update']
    rows=read(root/'observations.jsonl'); config=json.loads((root/'config.json').read_text())
    initial=[r for r in rows if r['phase']=='validation' and r['step']==0]
    original_length=sum(r['original_length'] for r in initial)/len(initial)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,key,title in zip(axes,['reward','distance','length'],['Objective (lower is better)','Last-token cosine distance','Message tokens']):
        sign=-1 if key=='reward' else 1
        ax.plot([r['step'] for r in train],[sign*r[key] for r in train],color='#aaa',alpha=.7,label='Training sample mean')
        ax.plot([r['step'] for r in val],[sign*r[key] for r in val],'o-',color='#2056a0',label='Validation (greedy)')
        reference={'reward':config['length_weight']*original_length,'distance':0,'length':original_length}[key]
        ax.axhline(reference,ls='--',color='#b04a34',label='Original prompt')
        ax.set(title=title,xlabel='RL step'); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{root.name}: frozen receiver, layer {config["layer_counted_from_one"]}, lambda={config["length_weight"]}')
    fig.tight_layout(); fig.savefig(out/f'{root.name}-learning.png',dpi=160); plt.close(fig)
    if args.analysis:
        data=read(Path(args.analysis)/'observations_with_embeddings.jsonl')
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for ax,(a,b) in zip(axes,[('pre_similarity','last_similarity'),('pre_similarity','embedding_similarity'),('last_similarity','embedding_similarity')]):
            for phase,color in [('train','#8798ab'),('validation','#2056a0'),('test_initial','#ba7737'),('test_best','#30935a')]:
                rs=[r for r in data if r['phase']==phase]
                if rs: ax.scatter([r[a] for r in rs],[r[b] for r in rs],s=20,alpha=.55,label=phase)
            ax.set(xlabel=a.replace('_',' '),ylabel=b.replace('_',' ')); ax.grid(alpha=.15)
        axes[0].legend(fontsize=7)
        fig.suptitle('Descriptive correlations; repeated task observations are not independent')
        fig.tight_layout(); fig.savefig(out/f'{root.name}-correlations.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
