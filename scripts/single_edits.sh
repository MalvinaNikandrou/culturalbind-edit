#!/bin/bash
# Single-edit experiments: each edit is applied to the unedited model and evaluated on its own.
#
# Usage: scripts/single_edits.sh MODEL [extra multimodal_edit.py args]
#   MODEL: qwen3_vl_4B | qwen3_vl_8B | gemma3_12B
#   Extra args are passed to every run, e.g. `--eval_size 20 --wandb_mode disabled` for a quick check.
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
  echo "=== ${name} (${MODEL})"
  uv run python multimodal_edit.py \
    --knowledge_editing_method "$method" \
    --knowledge_editing_hparams_config "$config" \
    --task_name crope \
    --train_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    "${EXTRA_ARGS[@]}" \
    --options sequential_edit=False test_generation=False "$@"
}

run FT-V finetune_multimodal "hparams/FT/${MODEL}_proj.yaml" eval_only=True
run FT-T finetune_multimodal "hparams/FT/${MODEL}.yaml" eval_only=True
run DPO dpo_multimodal "hparams/DPO/${MODEL}.yaml" eval_only=True
run IKE ike_multimodal "hparams/IKE/${MODEL}.yaml" eval_only=False
run WISE-T wise_multimodal "hparams/WISE/${MODEL}.yaml" eval_only=True
run WISE-V wise_multimodal "hparams/WISE/${MODEL}.yaml" eval_only=True training_locality=multimodal
run GRACE grace_multimodal "hparams/GRACE/${MODEL}.yaml" eval_only=True
if [ -n "${MEND_ARCHIVE:-}" ]; then
  run MEND mend_multimodal "hparams/MEND/${MODEL}.yaml" eval_only=True archive="$MEND_ARCHIVE"
else
  echo "=== Skipping MEND: set MEND_ARCHIVE to a checkpoint from scripts/mend_train.sh"
fi
