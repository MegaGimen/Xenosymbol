"""Select prefix-complete mid-user cuts before looking at any model outputs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from latent_protocol.context import canonical,make_cut,receiver_parts,sender_ids,context_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='data/trajectories')
    p.add_argument('--model',default='/workspace/model_nf4_native')
    p.add_argument('--max-input-tokens',type=int,default=6500)
    p.add_argument('--train',type=int,default=16);p.add_argument('--validation',type=int,default=4);p.add_argument('--test',type=int,default=4)
    args=p.parse_args();root=Path(args.root);tok=AutoTokenizer.from_pretrained(args.model)
    candidates=[];rejected=Counter();seen=set()
    for line in (root/'raw/source.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['uuid'] in seen:rejected['duplicate_trajectory']+=1;continue
        seen.add(row['uuid'])
        eligible=[i for i,m in enumerate(row['messages']) if m['role']=='user'
                  and any(x['role']=='user' for x in row['messages'][:i])
                  and any(x['role']=='tool' for x in row['messages'][:i])
                  and i+1<len(row['messages']) and row['messages'][i+1]['role']=='assistant']
        if not eligible:rejected['no_mid_user_after_tool']+=1;continue
        # Closest eligible user turn to the temporal middle, independent of outputs.
        index=min(eligible,key=lambda i:abs(i-(len(row['messages'])-1)/2))
        item=make_cut(row,index);before,after=receiver_parts(tok,item)
        prompt_ids=tok.encode(item['prompt'],add_special_tokens=False)
        sender_length=len(sender_ids(tok,item))
        receiver_length=len(before)+len(prompt_ids)+len(after)
        if max(sender_length,receiver_length)>args.max_input_tokens:
            rejected['input_budget_no_truncation']+=1;continue
        if not prompt_ids:rejected['empty_target']+=1;continue
        item.update(context_sha256=context_hash(item),sender_input_tokens=sender_length,
                    original_receiver_input_tokens=receiver_length,target_tokens=len(prompt_ids),
                    history_messages=len(item['history']))
        candidates.append(item)
    candidates.sort(key=lambda x:hashlib.sha256(('trajectory-split-v1:'+x['trajectory_id']).encode()).hexdigest())
    need=args.train+args.validation+args.test
    if len(candidates)<need:raise RuntimeError(f'Need {need} structural candidates; found {len(candidates)}; rejects {rejected}')
    offset=0;ids={}
    for split,n in [('train',args.train),('validation',args.validation),('test',args.test)]:
        selected=candidates[offset:offset+n];offset+=n
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected))
        ids[split]=[x['trajectory_id'] for x in selected]
    manifest={'source':json.loads((root/'source.json').read_text()),'selected_trajectories':ids,
        'available_candidates':len(candidates),'rejections':dict(rejected),'settings':vars(args),
        'policy':'one cut per trajectory; full prefix and tools; no future messages; no output-based selection',
        'selected_lengths':[{'id':x['id'],'sender':x['sender_input_tokens'],'receiver':x['original_receiver_input_tokens'],'target':x['target_tokens']} for x in candidates[:need]]}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
