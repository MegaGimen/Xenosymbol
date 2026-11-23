"""Posthoc context-only intervention. No learning and no change to encoder instructions."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from latent_protocol.train import Experiment,read_data
from latent_protocol.context import context_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--sample-id',required=True)
    p.add_argument('--old-id',required=True);p.add_argument('--new-id',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.run);config=json.loads((root/'config.json').read_text())
    item=next(x for x in read_data(root/'data/test.jsonl') if x['id']==a.sample_id)
    assert a.old_id not in item['prompt'] and a.new_id not in item['prompt']
    changed=copy.deepcopy(item)
    history_json=json.dumps(item['history'],ensure_ascii=False)
    count=history_json.count(a.old_id);assert count>0
    changed['history']=json.loads(history_json.replace(a.old_id,a.new_id))
    changed['context_sha256']=context_hash(changed)
    for key in ['sender_input_tokens','original_receiver_input_tokens']:
        if key in changed:changed['source_'+key]=changed.pop(key)
    assert changed['prompt']==item['prompt']
    exp=Experiment(SimpleNamespace(output=a.output,seed=config['seed'],model=config['model'],
        adapter=str(root/'best'),rank=config['rank'],lr=config['lr'],layer=config['layer_counted_from_one'],
        message_tokens=config['message_tokens'],response_tokens=config['response_tokens'],
        length_weight=config['length_weight'],temperature=config['temperature'],
        contextual=True,data_dir=str(root/'data')))
    m=exp.message(changed,sample=False);text=exp.tok.decode(m['message_ids'],skip_special_tokens=False)
    response=exp.receiver(m['message_ids'],changed)
    torch.save(response,exp.root/'receiver.pt')
    original=next(json.loads(l) for l in (root/'observations.jsonl').read_text().splitlines()
                  if json.loads(l)['id']==item['id'] and json.loads(l)['phase']=='test_best')
    result={'type':'posthoc context intervention, not an independent generalization score',
        'source_id':item['id'],'old_id':a.old_id,'new_id':a.new_id,'history_replacements':count,
        'unchanged_user_prompt':item['prompt'],'original_rewrite':original['message'],
        'intervened_rewrite':text,'receiver_text':response['text'],
        'new_id_in_rewrite':a.new_id in text,'old_id_in_rewrite':a.old_id in text,
        'new_id_in_receiver':a.new_id in response['text'],'message_stopped':m['message_stopped']}
    (exp.root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (exp.root/'intervened_input.json').write_text(json.dumps(changed,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
