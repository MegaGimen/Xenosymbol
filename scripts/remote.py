#!/usr/bin/env python3
"""Transfer/run through an interactive SSH PTY; never request an exec/scp channel."""
import argparse
import base64
import io
import shlex
import sys
import tarfile
import uuid
from pathlib import Path

import pexpect


HOST = 'nosana@AATASuSvJYZcysGJD6fnA9BdWVehXevXQs4jnRXT6n8d-0-ssh.node.k8s.prd.nos.ci'
PROXY = 'socat - PROXY:node.k8s.prd.nos.ci:%h:%p,proxyport=5002'


def execute(code, timeout):
    child = pexpect.spawn('ssh', ['-tt', '-i', '/root/.ssh/nosana_gpu',
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ProxyCommand=' + PROXY,
        HOST], encoding='utf-8', codec_errors='replace', timeout=timeout, maxread=65536)
    try:
        child.expect(r'[#$] ')
        child.sendline("stty -echo; export PS1='LATENT_REMOTE> '")
        child.expect('LATENT_REMOTE> ')
        # The initial command can echo the prompt string; synchronize on a fresh marker.
        ready = 'READY_' + uuid.uuid4().hex
        child.sendline('printf ' + shlex.quote(ready + '\\n'))
        child.expect(ready + r'\r?\n')
        marker = 'DONE_' + uuid.uuid4().hex
        begin = 'BEGIN_' + uuid.uuid4().hex
        script = f"import traceback\nprint({begin!r}, flush=True)\ntry:\n" + '\n'.join('    ' + l for l in code.splitlines())
        script += "\nexcept BaseException:\n    traceback.print_exc()\n    print('REMOTE_FAILURE')\n"
        script += f"finally:\n    print({marker!r}, flush=True)\n"
        encoded = base64.b64encode(script.encode()).decode()
        child.sendline('/opt/venv311/bin/python -u - <<\'PYREMOTE\'')
        child.sendline('import base64')
        child.sendline('payload = (')
        for i in range(0, len(encoded), 1800): child.sendline(repr(encoded[i:i+1800]))
        child.sendline(')')
        child.sendline("exec(compile(base64.b64decode(payload), '<remote>', 'exec'))")
        child.sendline('PYREMOTE')
        child.expect(begin + r'\r?\n')
        child.expect(marker + r'\r?\n')
        output = child.before.replace('\r', '')
        if 'REMOTE_FAILURE' in output: raise RuntimeError(output)
        return output
    finally:
        child.sendline('exit')
        child.close(force=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timeout', type=int, default=120)
    sub = p.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run'); run.add_argument('command')
    upload = sub.add_parser('upload'); upload.add_argument('paths', nargs='+')
    download = sub.add_parser('download'); download.add_argument('paths', nargs='+')
    download.add_argument('--destination', default='runs/remote')
    args = p.parse_args()
    if args.action == 'run':
        code = f"import os,subprocess\nos.chdir('/workspace/latent-protocol')\nr=subprocess.run({args.command!r},shell=True)\nprint('EXIT_CODE',r.returncode)\n"
        print(execute(code, args.timeout))
    elif args.action == 'upload':
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:gz') as t:
            for path in args.paths:
                if Path(path).is_absolute() or '..' in Path(path).parts: raise ValueError('Use relative paths')
                t.add(path, filter=lambda entry: None if '__pycache__' in entry.name else entry)
        encoded = base64.b64encode(archive.getvalue()).decode()
        code = f"import io,tarfile,base64\nwith tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))) as t:\n    t.extractall('/workspace/latent-protocol',filter='data')\nprint('UPLOADED')"
        print(execute(code, args.timeout))
    else:
        for path in args.paths:
            if Path(path).is_absolute() or '..' in Path(path).parts: raise ValueError('Use relative paths')
        code = f"import io,tarfile,base64,os\nos.chdir('/workspace/latent-protocol')\nb=io.BytesIO()\nwith tarfile.open(fileobj=b,mode='w:gz') as t:\n    for p in {args.paths!r}: t.add(p)\nprint('ARCHIVE_BEGIN')\nprint(base64.b64encode(b.getvalue()).decode())\nprint('ARCHIVE_END')"
        output = execute(code, args.timeout)
        payload = output.split('ARCHIVE_BEGIN\n', 1)[1].split('\nARCHIVE_END', 1)[0]
        destination = Path(args.destination); destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload))) as t:
            t.extractall(destination, filter='data')
        print(f'Downloaded to {destination}')


if __name__ == '__main__': main()
