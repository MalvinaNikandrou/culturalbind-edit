# CulturalBind: Knowledge Editing Experiments

Code for the cultural knowledge editing experiments in [Evaluating Cultural Knowledge Editing in Vision-Language Models
](https://arxiv.org/abs/2610.39602). 

This repository is a trimmed fork of [EasyEdit](https://github.com/zjunlp/EasyEdit) that keeps only the code these experiments use.

Edit vision-language models (Qwen3-VL 4B/8B and Gemma 3 12B) with cultural concepts and evaluates reliability, generality, locality and portability of each edit.


## Methods

| Name | `--knowledge_editing_method` | Config | Extra options |
|---|---|---|---|
| FT-T | `finetune_multimodal` | `hparams/FT/<model>.yaml` | |
| FT-V | `finetune_multimodal` | `hparams/FT/<model>_proj.yaml` | |
| DPO | `dpo_multimodal` | `hparams/DPO/<model>.yaml` | |
| IKE | `ike_multimodal` | `hparams/IKE/<model>.yaml` | `eval_only=False` |
| MEND | `mend_multimodal` | `hparams/MEND/<model>.yaml` | `archive=<checkpoint>` |
| WISE-T | `wise_multimodal` | `hparams/WISE/<model>.yaml` | |
| WISE-V | `wise_multimodal` | `hparams/WISE/<model>.yaml` | `training_locality=multimodal` |
| GRACE | `grace_multimodal` | `hparams/GRACE/<model>.yaml` | |

`<model>` is one of `qwen3_vl_4B`, `qwen3_vl_8B`, `gemma3_12B`. The configs contain the selected hyperparameters.

## Setup

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

Each model has its own edit set on the Hugging Face Hub, because edits are built from that
model's pre-edit answers incorrectly:

- `Malvinan/CulturalBind-edit-qwen3-vl-4b`
- `Malvinan/CulturalBind-edit-qwen3-vl-8b`
- `Malvinan/CulturalBind-edit-gemma3-12b`

Each has `train`, `validation` and `test` splits, stratified
at the concept level. MEND trains its editor on `train`, and all reported results are on `test`.

## Running the experiments

```bash
# Single edits: each edit is applied to the unedited model
scripts/single_edits.sh qwen3_vl_4B

# Sequential edits: 100 edits applied one after another to the same model
scripts/sequential_edits.sh qwen3_vl_4B
# Vary the number of sequential edits (10-250, 5 seeds each)
NUM_EDITS_SWEEP=1 scripts/sequential_edits.sh qwen3_vl_4B

# MEND trains an editor network first; pass the saved checkpoint to the scripts above
scripts/mend_train.sh qwen3_vl_4B
MEND_ARCHIVE=results/mend/<run>/models/MEND/Qwen3-VL-4B-Instruct scripts/single_edits.sh qwen3_vl_4B
```

Any extra arguments are passed to `multimodal_edit.py`. For a quick check on 20 samples without
logging:

```bash
scripts/single_edits.sh qwen3_vl_4B --eval_size 20 --wandb_mode disabled
```

A single run looks like this:

```bash
uv run python multimodal_edit.py \
  --knowledge_editing_method wise_multimodal \
  --knowledge_editing_hparams_config hparams/WISE/qwen3_vl_4B.yaml \
  --task_name crope \
  --train_dataset Malvinan/CulturalBind-edit-qwen3-vl-4b \
  --eval_dataset Malvinan/CulturalBind-edit-qwen3-vl-4b \
  --options sequential_edit=False test_generation=False eval_only=True
```

`--options key=value ...` overrides any field of the config.

### Logging

Metrics are logged to [Weights & Biases](https://wandb.ai). Use `--project_name` and `--wandb_entity`
to choose where runs go. Use `--wandb_mode offline` or `--wandb_mode disabled` to run without an account. Per-sample scores are also written under `results/`.

### Pre-edit cache

The unedited model's answers to the locality and portability questions are cached per sample in
`storage/pre_edit_cache/`. Later runs on the same dataset only generate answers for samples that are
not cached yet. Delete the cache if you change the model or the data.

## Acknowledgements

Built on [EasyEdit](https://github.com/zjunlp/EasyEdit), released under the MIT licence (see
`LICENSE`). If you use this code, please also cite EasyEdit:

```bibtex
@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}
```
