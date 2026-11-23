"""Replay the exact failed message, rather than silently testing a different sample."""
import json
from types import SimpleNamespace
import torch
from latent_protocol.train import Experiment


def main():
    source='runs/main-01/truncated_receiver.pt'
    failed=torch.load(source,weights_only=True)
    exp=Experiment(SimpleNamespace(output='runs/truncation-check',seed=73498,
        model='/workspace/model_nf4_native',adapter=None,rank=8,lr=1e-4,layer=32,
        message_tokens=48,response_tokens=1024,length_weight=.005,temperature=.85))
    full=exp.receiver(failed['message_ids'])
    torch.save(full,exp.root/'completed_receiver.pt')
    report={'source':source,'previous_tokens':len(failed['content_ids']),
        'completed_tokens':len(full['content_ids']),'complete':full['complete'],
        'same_prefix':full['content_ids'][:len(failed['content_ids'])]==failed['content_ids'],
        'text':full['text']}
    (exp.root/'replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    assert report['complete'] and report['same_prefix']


if __name__=='__main__':main()
