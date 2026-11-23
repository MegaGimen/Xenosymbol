#!/usr/bin/env python3
"""Check versions, actual CUDA/NF4 backward, and optional local model metadata."""
import argparse
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,help='Optional existing local NF4 model directory')
    p.add_argument('--output',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    expected={}
    for line in (root/'requirements.txt').read_text().splitlines():
        if line and not line.startswith(('#','-')):
            name,version=line.split('==');expected[name]=version
    versions={name:metadata.version(name) for name in expected}
    mismatches={k:{'expected':v,'actual':versions[k]} for k,v in expected.items()
                if versions[k]!=v and not (k in {'torch','torchvision'} and versions[k].split('+')[0]==v)}
    if mismatches:raise RuntimeError(f'Version mismatch: {mismatches}')
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Validated interpreter is Python 3.11')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    import torch
    import bitsandbytes as bnb
    import peft
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    assert Qwen3_5ForConditionalGeneration is not None and peft.LoraConfig is not None
    if not torch.cuda.is_available():raise RuntimeError('CUDA is not available')
    if torch.version.cuda!='13.2':raise RuntimeError(f'Expected CUDA 13.2 torch build, got {torch.version.cuda}')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('GPU does not support BF16')
    layer=bnb.nn.Linear4bit(64,32,bias=False,compute_dtype=torch.bfloat16,
                           quant_type='nf4',compress_statistics=True).to('cuda')
    x=torch.randn(4,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    loss=layer(x).float().square().mean();loss.backward();torch.cuda.synchronize()
    if not torch.isfinite(loss) or x.grad is None or not torch.isfinite(x.grad).all() or not x.grad.abs().sum()>0:
        raise RuntimeError('NF4 forward/backward check failed')
    report={'python':platform.python_version(),'versions':versions,'torch_build':torch.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'nf4_backward':'passed','full_model_loaded':False}
    if a.model:
        model_root=a.model.resolve()
        cfg=json.loads((model_root/'config.json').read_text())
        if cfg.get('model_type')!='qwen3_5':raise RuntimeError('This runner was validated with qwen3_5')
        quant=cfg.get('quantization_config',{})
        if not (quant.get('load_in_4bit') or quant.get('_load_in_4bit')) or quant.get('bnb_4bit_quant_type')!='nf4':
            raise RuntimeError('Expected the existing prequantized NF4 model, not the BF16 original')
        index=json.loads((model_root/'model.safetensors.index.json').read_text())
        shards=set(index['weight_map'].values())
        missing=[s for s in shards if not (model_root/s).is_file()]
        if missing:raise RuntimeError(f'Missing weight shards: {missing}')
        tokenizer=AutoTokenizer.from_pretrained(model_root,local_files_only=True)
        rendered=tokenizer.apply_chat_template([{'role':'user','content':'ping'}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if not rendered:raise RuntimeError('Empty chat template output')
        report['model']={'path':str(model_root),'architecture':cfg.get('architectures'),
                         'weight_shards':len(shards),'tokenizer_size':len(tokenizer),
                         'chat_template':'passed','quantization':'nf4'}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
