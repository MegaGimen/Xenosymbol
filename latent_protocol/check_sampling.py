"""Check cached sampling vs teacher-forced policy likelihood on the actual model."""
import argparse
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from latent_protocol.train import Experiment, read_data


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    args=p.parse_args()
    exp=Experiment(SimpleNamespace(output=args.output, seed=73498,
        model='/workspace/model_nf4_native',adapter=None,rank=8,lr=1e-4,layer=32,
        message_tokens=16,response_tokens=192,length_weight=.005,temperature=.85))
    prompt=exp.sender_prompt(read_data('data/train.jsonl')[0]['prompt'])
    ids=exp.tensor(prompt); exp.model.eval()
    with torch.no_grad():
        generated=exp.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            max_new_tokens=16,do_sample=True,use_cache=True,temperature=.85,
            top_k=0,top_p=1.,repetition_penalty=1.,logits_processor=[exp.restrict],
            pad_token_id=exp.tok.pad_token_id,eos_token_id=sorted(exp.stops),
            return_dict_in_generate=True,output_scores=True)
        actions=generated.sequences[0,len(prompt):]
        sample_logp=torch.stack([F.log_softmax(s[0].float(),dim=-1)[a]
                                for s,a in zip(generated.scores,actions)])
        full=generated.sequences
        del generated
        # Train/eval use identical zero dropout; train activates checkpointing.
        exp.model.train()
        logits=exp.model(input_ids=full,attention_mask=torch.ones_like(full),
            use_cache=False,logits_to_keep=len(actions)+1).logits[0,:-1].float()/.85
        logits[:,exp.excluded]=-float('inf')
        teacher_logp=-F.cross_entropy(logits,actions,reduction='none')
        error=(teacher_logp-sample_logp).abs()
    result={'sampled_tokens':len(actions),'max_abs_logp_error':float(error.max()),
            'mean_abs_logp_error':float(error.mean()),
            'sample_logp':sample_logp.tolist(),'teacher_logp':teacher_logp.tolist(),
            'note':'BF16 cached/chunked numerical differences are reported, not hidden.'}
    (exp.root/'sampling_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if float(error.max())>.25:
        raise RuntimeError('Unexpectedly large sampling/update likelihood mismatch')


if __name__=='__main__': main()
