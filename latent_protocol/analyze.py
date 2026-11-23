"""Independent output embeddings and state correlations; never used for RL."""
import argparse
import difflib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer
from latent_protocol.tool_comparison import tool_comparison


def correlations(rows):
    metrics = ['last_similarity', 'pre_similarity', 'embedding_similarity', 'text_similarity']
    result = {'n': len(rows), 'n_tasks': len({r['id'] for r in rows}), 'pairs': {}}
    for i, left in enumerate(metrics):
        for right in metrics[i + 1:]:
            a = np.array([r[left] for r in rows]); b = np.array([r[right] for r in rows])
            if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                pair = {'pearson': None, 'spearman': None, 'reason': 'too few observations or constant metric'}
            else:
                pair = {'pearson': float(pearsonr(a, b).statistic),
                        'spearman': float(spearmanr(a, b).statistic)}
            result['pairs'][left + '__' + right] = pair
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--embedding-model', default='intfloat/multilingual-e5-small')
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model).eval().cpu()
    rows = []
    for path in args.runs:
        for line in (Path(path) / 'observations.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['phase'] == 'sanity_identity': continue
            rows.append(dict(row, run=str(path)))
    texts = list(dict.fromkeys(t for r in rows for t in [r['original_text'], r['text']]))
    vectors = {}; truncations = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            inputs = ['query: ' + t for t in batch]
            for t, text in zip(batch, inputs):
                if len(tok.encode(text)) > 512: truncations.append(t)
            encoded = tok(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')
            h = model(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).to(h.dtype)
            e = F.normalize((h * mask).sum(1) / mask.sum(1), dim=-1)
            vectors.update({t: v for t, v in zip(batch, e)})
    for row in rows:
        row.update(tool_comparison(row['original_text'],row['text']))
        row['last_similarity'] = 1 - row['distance']
        row['pre_similarity'] = 1 - row['pre_distance']
        row['embedding_similarity'] = (1.0 if row['original_text']==row['text'] else
            max(-1.,min(1.,float(vectors[row['original_text']] @ vectors[row['text']]))))
        row['text_similarity'] = difflib.SequenceMatcher(None, row['original_text'], row['text']).ratio()
    with (out / 'observations_with_embeddings.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'embedding_model': args.embedding_model,
        'embedding_revision': getattr(model.config, '_commit_hash', None),
        'embedding_pooling': 'attention-mask mean, L2 normalize, query: prefix on both outputs',
        'warning': 'Descriptive correlations only; repeated task observations are not independent. No human semantic ratings collected.',
        'embedding_truncated_texts': truncations,
        'all': correlations(rows),
        'non_control':correlations([r for r in rows if not r['phase'].startswith('control_')]),
        'tool_comparison_note':'Posthoc only: same generated function names and arguments, not execution success or reward.',
        'tool_call_match_by_phase':{phase:{'comparable':sum(r['tool_call_match'] is not None for r in rows if r['phase']==phase),
            'matching':sum(r['tool_call_match'] is True for r in rows if r['phase']==phase)} for phase in sorted({r['phase'] for r in rows})},
        'by_phase': {phase: correlations([r for r in rows if r['phase'] == phase])
                     for phase in sorted({r['phase'] for r in rows})}}
    (out / 'correlations.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    torch.save(vectors, out / 'output_embeddings.pt')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
