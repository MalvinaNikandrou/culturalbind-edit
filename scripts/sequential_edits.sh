#!/bin/bash
# Sequential-edit experiments: edits are applied one after another to the same model, which is evaluated
# after all edits. By default this edits 100 distinct concepts.
#
# Usage: scripts/sequential_edits.sh MODEL [extra multimodal_edit.py args]
#   MODEL: qwen3_vl_4B | qwen3_vl_8B | gemma3_12B
#   Extra args are passed to every run, e.g. `--wandb_mode disabled`.
#   NUM_EDITS_SWEEP=1 instead varies the number of edits (10-250) over 5 seeds.
#
# MEND needs an editor trained with scripts/mend_train.sh; set MEND_ARCHIVE to its checkpoint to include it.
set -euo pipefail

MODEL=${1:?"usage: $0 {qwen3_vl_4B|qwen3_vl_8B|gemma3_12B} [extra args]"}
shift
case "$MODEL" in
  qwen3_vl_4B) DATASET=Malvinan/CulturalBind-edit-qwen3-vl-4b ;;
  qwen3_vl_8B) DATASET=Malvinan/CulturalBind-edit-qwen3-vl-8b ;;
  gemma3_12B) DATASET=Malvinan/CulturalBind-edit-gemma3-12b ;;
  *) echo "Unknown model: $MODEL" >&2; exit 1 ;;
esac
EXTRA_ARGS=("$@")

run() {
  local name=$1 method=$2 config=$3
  shift 3
  echo "=== ${name} (${MODEL}) ${RUN_ARGS[*]}"
  uv run python multimodal_edit.py \
    --knowledge_editing_method "$method" \
    --knowledge_editing_hparams_config "$config" \
    --task_name crope \
    --train_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    "${RUN_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --options sequential_edit=True test_generation=False eval_only=True "${RUN_OPTIONS[@]}" "$@"
}

run_all_methods() {
  run FT-V finetune_multimodal "hparams/FT/${MODEL}_proj.yaml"
  run FT-T finetune_multimodal "hparams/FT/${MODEL}.yaml"
  run DPO dpo_multimodal "hparams/DPO/${MODEL}.yaml"
  run WISE-T wise_multimodal "hparams/WISE/${MODEL}.yaml"
  run WISE-V wise_multimodal "hparams/WISE/${MODEL}.yaml" training_locality=multimodal
  run GRACE grace_multimodal "hparams/GRACE/${MODEL}.yaml"
  if [ -n "${MEND_ARCHIVE:-}" ]; then
    run MEND mend_multimodal "hparams/MEND/${MODEL}.yaml" archive="$MEND_ARCHIVE"
  else
    echo "=== Skipping MEND: set MEND_ARCHIVE to a checkpoint from scripts/mend_train.sh"
  fi
}

if [ "${NUM_EDITS_SWEEP:-0}" = "1" ]; then
  for num_edits in 10 20 40 50 100 150 200 250; do
    for seed in 42 5678 12340 989898 730; do
      RUN_ARGS=(--eval_size "$num_edits")
      RUN_OPTIONS=(seed="$seed")
      run_all_methods
    done
  done
else
  RUN_ARGS=()
  RUN_OPTIONS=()
  run_all_methods
fi
