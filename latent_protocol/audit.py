"""Recompute rewards and indices from saved tensors, without loading the 27B model."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from latent_protocol.core import reward, selected_positions
from latent_protocol.context import context_hash


def main():
    p = argparse.ArgumentParser(); p.add_argument('run'); args = p.parse_args()
    root = Path(args.run); config = json.loads((root/'config.json').read_text())
    rows = [json.loads(l) for l in (root/'observations.jsonl').read_text().splitlines()]
    errors = []
    items={};split_ids=[]
    if config.get('contextual'):
        for split in ['train','validation','test']:
            entries=[json.loads(l) for l in (root/'data'/f'{split}.jsonl').read_text().splitlines()]
            items.update({r['id']:r for r in entries})
            split_ids.append(set(r['trajectory_id'] for r in entries))
        if any(split_ids[i]&split_ids[j] for i in range(3) for j in range(i)):
            errors.append(['dataset','trajectory split leakage'])
    for i, row in enumerate(rows):
        ref = torch.load(root/row['reference_artifact'], weights_only=True)
        obs = torch.load(root/row['artifact'], weights_only=True)
        if not all(torch.isfinite(x[k]).all() for x in [ref,obs] for k in ['pre','last']):
            errors.append([i,'non-finite hidden state']);continue
        expected = selected_positions(len(obs['receiver_prompt_ids']), len(obs['content_ids']))
        if tuple(obs['positions']) != expected: errors.append([i, 'bad positions'])
        if not obs['complete']: errors.append([i, 'truncated output'])
        if obs['message_ids'] != row['message_ids']: errors.append([i, 'message ID mismatch'])
        if config.get('contextual'):
            if ref['shared_prefix_ids']!=obs['shared_prefix_ids'] or ref['shared_suffix_ids']!=obs['shared_suffix_ids']:
                errors.append([i,'shared context changed'])
            if obs['receiver_prompt_ids']!=obs['shared_prefix_ids']+obs['message_ids']+obs['shared_suffix_ids']:
                errors.append([i,'replacement boundaries incorrect'])
            if obs['context_sha256']!=context_hash(items[row['id']]):
                errors.append([i,'context fingerprint mismatch'])
        distance = max(0., float(1-F.cosine_similarity(ref['last'], obs['last'], dim=0)))
        pre_distance = max(0., float(1-F.cosine_similarity(ref['pre'], obs['pre'], dim=0)))
        expected_reward = reward(distance, len(obs['message_ids']), config['length_weight'])
        if abs(row['distance']-distance)>1e-7: errors.append([i, 'distance mismatch'])
        if abs(row['pre_distance']-pre_distance)>1e-7: errors.append([i, 'pre-distance mismatch'])
        if abs(row['reward']-expected_reward)>1e-7: errors.append([i, 'reward mismatch'])
    updates = {}
    initial = load_file(str(root/'initial/adapter_model.safetensors'))
    for path in sorted(root.glob('checkpoint-*/adapter_model.safetensors')):
        current = load_file(str(path))
        changed = sum(not torch.equal(value, current[key]) for key,value in initial.items())
        max_delta = max(float((value-current[key]).abs().max()) for key,value in initial.items())
        updates[path.parent.name] = {'changed_tensors': changed, 'max_abs_delta': max_delta}
        if not changed: errors.append([path.parent.name, 'adapter not updated'])
    report = {'checked_observations':len(rows), 'errors':errors, 'checkpoints':updates}
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
