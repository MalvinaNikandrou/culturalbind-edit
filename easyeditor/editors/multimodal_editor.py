import gc
import logging
from time import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb

from ..evaluate.multimodal_evaluator import MultimodalEditEvaluator
from ..trainer.models import *
from ..trainer.utils import RunningStatAverager
from ..util import nethook
from ..util.alg_dict import *
from ..util.globals import *
from ..util.hparams import HyperParams

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

LOG = logging.getLogger(__name__)


def make_logs():
    f_h, s_h = get_handler("logs/", log_name="run.log")
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditor:
    """Multimodal editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):

        assert hparams is not None or print("Error: hparams is None.")

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()
        self.hparams = hparams
        self.initialize_model()
        self.evaluator = MultimodalEditEvaluator(hparams=hparams)

    def initialize_model(self):
        """Initialize the model."""
        LOG.info("Instantiating model")
        self.model = get_model(self.hparams)
        if type(self.model_name) is str:
            if "qwen3-vl" not in self.hparams.model_name.lower() and "gemma-3" not in self.hparams.model_name.lower():
                raise NotImplementedError(f"Model {self.model_name} not supported yet.")

        if self.alg_name == "UNKE" and self.model.config.tie_word_embeddings:
            # Get the tied weight
            input_embeds = self.model.get_input_embeddings()  # nn.Embedding

            # Create a new independent Linear layer for the LM head
            output_embeds = self.model.get_output_embeddings()  # usually nn.Linear

            # Deep copy the weight so it's no longer shared
            output_embeds.weight = torch.nn.Parameter(input_embeds.weight.detach().clone())

        self.model.to(self.hparams.device)

    @torch.inference_mode()
    def reset(self, weights_copy):
        """Reset the model to original weights."""
        if self.alg_name == "IKE":
            return
        if self.alg_name in ["KN", "GRACE", "WISE", "LiveEdit"]:
            weights_copy()
        else:
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(self.hparams.device)

    def echo(self, train_step, info_dict, pretty=False):
        sep = "\n" if pretty else "; "

        def key_format(k):
            return k.ljust(20) if pretty else k

        LOG.info(f"Step {train_step}:")
        LOG.info(sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()]))

    def edit_dataset(self, ds: Dataset, keep_original_weight=False, verbose=True, **kwargs):
        # Make sure dataset supported
        assert (
            sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0
        ), f"Dataset {ds} not supported yet."
        self._set_processor_and_tokenizer(ds)
        self.averager = RunningStatAverager()
        eval_start_time = time()
        sequential_edit = getattr(self.hparams, "sequential_edit", False)
        self._initialize_dataset_locality_answers(ds)

        if sequential_edit:
            self.eval_dataset_sequentially(ds, verbose, **kwargs)
        else:
            self.eval_dataset_single_edit(ds, verbose, **kwargs)

        del self.model
        # MEND's executor keeps its own references to the model and editor network
        if hasattr(self.apply_algo, "__self__"):
            self.apply_algo.__self__.reset()
        gc.collect()
        torch.cuda.empty_cache()
        stats = self.averager.average()
        judge_stats = self.evaluator.evaluate_portability_with_judge()
        stats.update(judge_stats)
        if verbose:
            self.echo(0, stats, pretty=True)
        self._inline_validation_log(step=len(ds), stats=stats, start_time=eval_start_time, steps=len(ds))
        return stats

    def eval_dataset_sequentially(self, ds: Dataset, verbose, **kwargs):
        # Editing
        original_weights = None
        start = time()
        if self.alg_name == "IKE":
            train_ds = kwargs["train_ds"]
            train_ds.add_data(ds)
        else:
            for eval_step, request in enumerate(tqdm(ds, desc="Editing dataset", total=len(ds))):
                edited_model, original_weights = self.apply_algo(
                    self.model,
                    self.processor,
                    [request],
                    self.hparams,
                    copy=False,
                    original_weights=original_weights,
                    return_orig_weights=True,
                    # keep_original_weight=keep_original_weight
                )
                # if edited_model != self.model:
                #     breakpoint()
        exec_time = time() - start
        LOG.info(f"Execution {eval_step} editing took {exec_time}")
        # Post edit evaluations
        for eval_step, request in enumerate(tqdm(ds, desc="Evaluating dataset (post-edits)", total=len(ds))):
            if self.alg_name == "IKE":
                assert "train_ds" in kwargs.keys() or print("IKE need train_ds (For getting In-Context prompt)")
                edited_model = self.model
                icl_examples = self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    # keep_original_weight=keep_original_weight,
                    train_ds=train_ds,
                    reencode_facts=eval_step == 0,
                )
            else:
                icl_examples = None
            metrics = self.evaluator.evaluate_post_edit(
                model=edited_model,
                exec_time=exec_time,
                request_id=eval_step,
                request=request,
                icl_examples=icl_examples,
            )
            if verbose:
                LOG.info(f"{eval_step} editing: {request['prompt']} -> {request['target']}  \n {metrics}")

            self.averager.add(metrics["post"])

    def eval_dataset_single_edit(self, ds: Dataset, verbose, **kwargs):
        original_weights = None
        for eval_step, request in enumerate(tqdm(ds, desc="Editing dataset", total=len(ds))):
            # Apply the editing algorithm
            start = time()
            if self.alg_name == "IKE":
                assert "train_ds" in kwargs.keys() or print("IKE need train_ds (For getting In-Context prompt)")
                edited_model = self.model
                icl_examples = self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    return_orig_weights=True,
                    train_ds=kwargs["train_ds"],
                )
            else:
                edited_model, original_weights = self.apply_algo(
                    self.model,
                    self.processor,
                    [request],
                    self.hparams,
                    return_orig_weights=True,
                    original_weights=original_weights,
                )
                icl_examples = None
            exec_time = time() - start
            LOG.info(f"Execution {eval_step} editing took {exec_time}")
            # Get post-edit evaluation metrics
            start = time()
            metrics = self.evaluator.evaluate_post_edit(
                model=edited_model,
                exec_time=exec_time,
                request_id=eval_step,
                request=request,
                icl_examples=icl_examples,
            )
            LOG.info(f"Evaluation took {time() - start}")
            if verbose:
                LOG.info(f"{eval_step} editing: {request['prompt']} -> {request['target']}  \n {metrics}")
            self.averager.add(metrics["post"])
            # Reset the model to original weights
            self.reset(original_weights)

    def _initialize_dataset_locality_answers(self, dataset):
        dataset.update_pre_edit_answers(self.model)

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for index in range(0, len(arr), n):
            yield arr[index : index + n]

    def _set_processor_and_tokenizer(self, ds: Dataset):
        self.processor = ds.collater.processor
        self.tok = ds.collater.processor.tokenizer
        self.vis_tok = ds.collater.processor.processor.image_processor
        self.evaluator.processor = self.processor

    def _inline_validation_log(self, step: int, stats: dict[str, float], start_time: float, steps: int) -> None:
        log_info = f"Step {step+1}/{steps}".ljust(20)
        elapsed = (time() - start_time) / (step + 1)
        log_info += f"Elapsed: {elapsed:<12.5f} "
        metrics = [
            "reliability_accuracy",
            "text_generality_accuracy",
            "image_generality_accuracy",
            "text_locality_accuracy",
            "image_locality_accuracy",
            # "text_portability_accuracy",
            # "image_portability_accuracy",
            "text_judge_portability_accuracy",
            "image_judge_portability_accuracy",
            "average_accuracy",
        ]
        aggregate_stats = {}
        for metric in metrics:
            acc = stats.get(metric, None)
            if acc is not None:
                aggregate_stats[metric] = acc
                log_info += f"{metric}: {acc:<12.5f} "
        average = sum(stats.values()) / len(stats)
        aggregate_stats["average_accuracy"] = average
        wandb.log(aggregate_stats)
        self._print_for_table(metrics, aggregate_stats)
        LOG.info(log_info)

    def _print_for_table(self, metrics, aggregate_stats: dict[str, float]) -> None:
        values_str = ""
        total = 0.0
        for metric in metrics:
            acc = aggregate_stats.get(metric, 0)
            values_str += f" & {acc*100:.2f}"
            total += acc
        LOG.info(values_str)
