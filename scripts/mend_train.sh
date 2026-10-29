#!/bin/bash
# Train the MEND editor network for one model. The best checkpoint is saved to
#   results/mend/MEND-<model>-layers<...>-maxiters<...>-lr<...>/models/MEND/<model>
# Pass that path as MEND_ARCHIVE to scripts/single_edits.sh and scripts/sequential_edits.sh.
#
# Usage: scripts/mend_train.sh MODEL [extra multimodal_edit.py args]
#   MODEL: qwen3_vl_4B | qwen3_vl_8B | gemma3_12B
set -euo pipefail

MODEL=${1:?"usage: $0 {qwen3_vl_4B|qwen3_vl_8B|gemma3_12B} [extra args]"}
shift
case "$MODEL" in
  qwen3_vl_4B) DATASET=Malvinan/CulturalBind-edit-qwen3-vl-4b ;;
  qwen3_vl_8B) DATASET=Malvinan/CulturalBind-edit-qwen3-vl-8b ;;
  gemma3_12B) DATASET=Malvinan/CulturalBind-edit-gemma3-12b ;;
  *) echo "Unknown model: $MODEL" >&2; exit 1 ;;
esac

uv run python multimodal_edit.py \
  --knowledge_editing_method mend_multimodal \
  --knowledge_editing_hparams_config "hparams/MEND/${MODEL}.yaml" \
  --task_name crope \
  --train_dataset "$DATASET" \
  --eval_dataset "$DATASET" \
  "$@" \
  --options eval_only=False results_root_dir=results/mend
