#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
training_pid="$2"
while kill -0 "$training_pid" 2>/dev/null; do
  sleep 3
done
test -f "$run_dir/summary.json"
/opt/venv311/bin/python -m latent_protocol.audit "$run_dir"
/opt/venv311/bin/python -m latent_protocol.analyze "$run_dir" --output "$run_dir-analysis"
/opt/venv311/bin/python scripts/summarize.py "$run_dir" --output "$run_dir/comparison.json"
printf '%s\n' 'POSTPROCESS_COMPLETE'
