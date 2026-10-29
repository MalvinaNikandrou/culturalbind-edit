from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

from .editors.multimodal_editor import MultimodalEditor
from .models.ike.util import encode_ike_facts_multimodal
from .trainer.MultimodalTrainer import MultimodalTrainer
from .util.seed import seed_everything

seed_everything(4567)


def ike(config, train_ds: Dataset, eval_ds: Dataset):
    sentence_model = SentenceTransformer(config.sentence_model_name).to(config.device)
    encode_ike_facts_multimodal(sentence_model, train_ds, config)
    editor = MultimodalEditor.from_hparams(config)
    editor.edit_dataset(ds=eval_ds, train_ds=train_ds, keep_original_weight=True)


def train(config, train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset):
    trainer = MultimodalTrainer(config=config, train_set=train_ds, val_set=eval_ds, test_set=test_ds)
    trainer.run()


def train_editor(config, train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset):
    editor = MultimodalEditor.from_hparams(config)
    editor.edit_dataset(ds=eval_ds, train_ds=train_ds, keep_original_weight=True)


def evaluate_editor(config, train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset):
    editor = MultimodalEditor.from_hparams(config)
    editor.edit_dataset(ds=test_ds, train_ds=train_ds, keep_original_weight=True)


def evaluate(config, train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset):
    config.final_eval = True
    config.eval_only = True
    trainer = MultimodalTrainer(config=config, train_set=train_ds, val_set=eval_ds, test_set=test_ds)
    val_info = trainer.validate()
    trainer.final_eval(val_info)


def run_multimodal_ke_training(config, train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset):
    if config.alg_name == "IKE":
        ike(config, train_ds, test_ds)
    elif config.alg_name == "MEND":
        train(config, train_ds, eval_ds, test_ds)
    elif config.alg_name in ["WISE", "GRACE", "FT", "DPO"]:
        print("Model does not require training, directly evaluating on the validation set...")
        train_editor(config, train_ds, eval_ds, test_ds)


def run_multimodal_ke_evaluation(config, train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset = None):
    if config.alg_name in ["WISE", "GRACE", "FT", "DPO", "MEND", "IKE"]:
        evaluate_editor(config, train_ds, eval_ds, test_ds)
    elif config.alg_name in ["MEND"]:
        evaluate(config, train_ds, eval_ds, test_ds)
    # editor = MultimodalEditor.from_hparams(config)
    # metrics, _, _ = editor.edit_dataset(ds=eval_ds, train_ds=train_ds, keep_original_weight=True)
    # print_result(metrics)
