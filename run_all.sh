#!/usr/bin/env bash
# Runs all required experiments on Mac CPU, sequentially, with logs & resume.
# Usage:
#   chmod +x run_all.sh
#   nohup ./run_all.sh > master.log 2>&1 & disown
#   # or: tmux new -s llama && ./run_all.sh

set -euo pipefail

# ---------- config ----------
PYTHON_BIN="${PYTHON_BIN:-python3}"       # override if needed
VENV_DIR="${VENV_DIR:-.venv}"
LOGDIR="${LOGDIR:-logs}"
DONE="${DONE_DIR:-.done}"
SEED="${SEED:-42}"                        # stick to one seed unless you want to loop

# macOS CPU niceness (lower priority so your Mac stays usable)
NICENESS="${NICENESS:-10}"

# Optional CPU threading caps (helps battery/thermals on laptops)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOGDIR" "$DONE"

# ---------- helpers ----------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_step () {
  local tag="$1"; shift
  local doneflag="$DONE/$tag.done"
  local logfile="$LOGDIR/$tag.log"
  local cmd="$*"

  if [[ -f "$doneflag" ]]; then
    echo "[${tag}] $(timestamp) — already completed, skipping."
    return 0
  fi

  echo ""
  echo "================================================================"
  echo "[${tag}] $(timestamp) — starting"
  echo "Log: $logfile"
  echo "Cmd: $cmd"
  echo "================================================================"
  echo ""

  # Run and tee to log. 'nice' lowers priority.
  if /usr/bin/nice -n "$NICENESS" bash -lc "$cmd" 2>&1 | tee "$logfile"; then
    touch "$doneflag"
    echo "[${tag}] $(timestamp) — COMPLETED ✔"
  else
    echo "[${tag}] $(timestamp) — FAILED ✖ (see $logfile)"
    exit 1
  fi
}

# ---------- environment ----------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Install python3 (e.g., 'brew install python')."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Keep pip low-noise & deterministic-ish
python -m pip install --upgrade pip >/dev/null
# Run your provided setup script (as per assignment notes)
if [[ -f setup.sh ]]; then
  echo "Running setup.sh to install dependencies..."
  bash setup.sh
else
  echo "WARNING: setup.sh not found. Make sure dependencies are installed per assignment."
fi

# Quick sanity: verify required files exist
need_file () {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "Required file missing: $f"
    exit 1
  fi
}
need_file "run_llama.py"
need_file "llama.py"
need_file "classifier.py"
need_file "optimizer.py"
need_file "rope.py"
need_file "lora.py"
need_file "stories42M.pt"

for f in \
  data/sst-train.txt data/sst-dev.txt data/sst-test.txt data/sst-label-mapping.json \
  data/cfimdb-train.txt data/cfimdb-dev.txt data/cfimdb-test.txt data/cfimdb-label-mapping.json
do
  need_file "$f"
done

# ---------- steps ----------
# 0) Text generation sanity check (optional but nice to ensure model runs)
run_step "00_generate" \
  "python run_llama.py --option generate"

# 1) Zero-shot prompting — SST
run_step "10_prompt_sst" \
  "python run_llama.py --option prompt --batch_size 10 \
   --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
   --label-names data/sst-label-mapping.json \
   --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt \
   --seed $SEED"

# 2) Zero-shot prompting — CFIMDB
run_step "11_prompt_cfimdb" \
  "python run_llama.py --option prompt --batch_size 10 \
   --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
   --label-names data/cfimdb-label-mapping.json \
   --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt \
   --seed $SEED"

# 3) Classification fine-tuning — SST
run_step "20_finetune_sst" \
  "python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 \
   --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
   --label-names data/sst-label-mapping.json \
   --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt \
   --seed $SEED"

# 4) Classification fine-tuning — CFIMDB
run_step "21_finetune_cfimdb" \
  "python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 \
   --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
   --label-names data/cfimdb-label-mapping.json \
   --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt \
   --seed $SEED"

# 5) LoRA fine-tuning — SST
run_step "30_lora_sst" \
  "python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 80 \
   --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
   --label-names data/sst-label-mapping.json \
   --dev_out sst-dev-lora-output.txt --test_out sst-test-lora-output.txt \
   --lora_rank 4 --lora_alpha 1.0 \
   --seed $SEED"

# 6) LoRA fine-tuning — CFIMDB
run_step "31_lora_cfimdb" \
  "python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 10 \
   --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
   --label-names data/cfimdb-label-mapping.json \
   --dev_out cfimdb-dev-lora-output.txt --test_out cfimdb-test-lora-output.txt \
   --lora_rank 4 --lora_alpha 1.0 \
   --seed $SEED"

echo ""
echo "================================================================"
echo "ALL DONE 🎉  See logs in $LOGDIR"
echo "Outputs:"
echo "  * sst-dev-prompting-output.txt, sst-test-prompting-output.txt"
echo "  * cfimdb-dev-prompting-output.txt, cfimdb-test-prompting-output.txt"
echo "  * sst-dev-finetuning-output.txt, sst-test-finetuning-output.txt"
echo "  * cfimdb-dev-finetuning-output.txt, cfimdb-test-finetuning-output.txt"
echo "  * sst-dev-lora-output.txt, sst-test-lora-output.txt"
echo "  * cfimdb-dev-lora-output.txt, cfimdb-test-lora-output.txt"
echo "Completion flags in: $DONE"
echo "================================================================"