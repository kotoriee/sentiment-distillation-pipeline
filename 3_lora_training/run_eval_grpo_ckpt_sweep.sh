#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/kotoriee/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="/home/kotoriee/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

CKPT_ROOT="${1:-outputs_grpo_9b_rewardfix_v3_nothink_continue_300}"
RESULT_DIR="${2:-../6_experiments_results}"
DATA_FILE="${3:-../data/test_answer_first.json}"
LOG_DIR="${CKPT_ROOT}/eval_logs"
SUMMARY_FILE="${RESULT_DIR}/qwen35_9b_grpo_rewardfix_v3_nothink_continue_300_checkpoint_sweep.tsv"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

write_summary() {
  python - "$RESULT_DIR" "$SUMMARY_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
summary_file = Path(sys.argv[2])
rows = []

for path in sorted(result_dir.glob("qwen35_9b_grpo_rewardfix_v3_nothink_continue_300_checkpoint_*_eval.json")):
    match = re.search(r"checkpoint_(\d+)_eval\.json$", path.name)
    if not match:
        continue
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    rows.append({
        "checkpoint": int(match.group(1)),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("f1_macro"),
        "f1_class_0": metrics.get("f1_class_0"),
        "f1_class_1": metrics.get("f1_class_1"),
        "f1_class_2": metrics.get("f1_class_2"),
        "correct": metrics.get("correct"),
        "parse_failures": metrics.get("parse_failures"),
    })

rows.sort(key=lambda row: row["checkpoint"])
summary_file.parent.mkdir(parents=True, exist_ok=True)
with summary_file.open("w", encoding="utf-8") as f:
    f.write("checkpoint\taccuracy\tmacro_f1\tf1_class_0\tf1_class_1\tf1_class_2\tcorrect\tparse_failures\n")
    for row in rows:
        f.write(
            f"{row['checkpoint']}\t{row['accuracy']}\t{row['macro_f1']}\t"
            f"{row['f1_class_0']}\t{row['f1_class_1']}\t{row['f1_class_2']}\t"
            f"{row['correct']}\t{row['parse_failures']}\n"
        )

if rows:
    best = max(rows, key=lambda row: (row["macro_f1"] or 0, row["accuracy"] or 0))
    print(
        "CURRENT_BEST "
        f"checkpoint={best['checkpoint']} accuracy={best['accuracy']} "
        f"macro_f1={best['macro_f1']} correct={best['correct']}"
    )
PY
}

mapfile -t CHECKPOINTS < <(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

echo "Checkpoint sweep started: $(date --iso-8601=seconds)"
echo "Checkpoint root: $CKPT_ROOT"
echo "Results: $RESULT_DIR"
echo "Summary: $SUMMARY_FILE"
echo "Found ${#CHECKPOINTS[@]} checkpoints"

for ckpt_path in "${CHECKPOINTS[@]}"; do
  ckpt_name="$(basename "$ckpt_path")"
  step="${ckpt_name#checkpoint-}"
  output_file="${RESULT_DIR}/qwen35_9b_grpo_rewardfix_v3_nothink_continue_300_checkpoint_${step}_eval.json"
  log_file="${LOG_DIR}/${ckpt_name}_eval.log"

  if [[ -s "$output_file" ]]; then
    echo "Skipping $ckpt_name; result exists: $output_file"
    write_summary
    continue
  fi

  echo
  echo "Evaluating $ckpt_name: $(date --iso-8601=seconds)"
  python -u evaluate_lora.py \
    --model "$ckpt_path" \
    --data "$DATA_FILE" \
    --qwen \
    --batch-size 8 \
    --max-new-tokens 32 \
    --max-seq-length 512 \
    --max-input-length 512 \
    --output "$output_file" \
    > "$log_file" 2>&1

  status=$?
  if [[ $status -ne 0 ]]; then
    echo "FAILED $ckpt_name status=$status log=$log_file"
  else
    echo "DONE $ckpt_name result=$output_file"
  fi

  write_summary
done

echo
echo "Checkpoint sweep finished: $(date --iso-8601=seconds)"
write_summary
