#!/usr/bin/env python3
"""Export package versions and check installed Python sources against wheel RECORDs."""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    freeze=subprocess.run([sys.executable,'-m','pip','freeze','--all'],capture_output=True,text=True,check=True)
    # Refuse to publish direct URLs/local install paths as a purported version lock.
    lines=freeze.stdout.splitlines()
    if any(l and not l.startswith('#') and not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',l) for l in lines):
        raise RuntimeError('Non-version dependency found; review its origin before exporting')
    (root/'remote-freeze.txt').write_text(freeze.stdout)
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (root/'pip-check.txt').write_text(check.stdout+check.stderr)
    packages={d.metadata['Name']:d.version for d in metadata.distributions()}
    audit={}
    for name in ['torch','transformers','peft','bitsandbytes','accelerate','llamafactory','trl']:
        dist=metadata.distribution(name);changed=[];checked=0
        for f in dist.files or []:
            if f.suffix!='.py' or not f.hash or f.hash.mode!='sha256':continue
            path=Path(dist.locate_file(f))
            if not path.is_file():changed.append({'file':str(f),'status':'missing'});continue
            digest=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
            checked+=1
            if digest!=f.hash.value:changed.append({'file':str(f),'status':'different_from_RECORD'})
        audit[name]={'version':dist.version,'checked_python_files':checked,'changes':changed}
    import torch
    smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True)
    report={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
            'os_release':Path('/etc/os-release').read_text(),'packages':dict(sorted(packages.items())),
            'torch_import_version':torch.__version__,'torch_cuda':torch.version.cuda,
            'cudnn':torch.backends.cudnn.version(),'cuda_available':torch.cuda.is_available(),
            'gpu':smi.stdout.strip(),'pip_check_returncode':check.returncode,
            'source_record_checks':audit,
            'scope':'RECORD checks cover listed Python files only, not provenance of the original wheels or unlisted files.'}
    (root/'runtime.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'pip_check':check.returncode,'torch':torch.__version__,
                      'source_changes':{k:v['changes'] for k,v in audit.items()}},indent=2))
    if check.returncode:raise SystemExit(check.returncode)


if __name__=='__main__':main()
