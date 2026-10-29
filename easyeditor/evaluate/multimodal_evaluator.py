import logging

import torch
from .judge_metrics import BackgroundAccuracy

import wandb

from ..util.globals import *
from ..util.hparams import HyperParams
from .metrics import EditSuccessMetric, EntityOverlapMetric, LocalityMetric, TFMetric
from .multimodal_evaluation_utils import EVALUATION_UTILS_MAPPING

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name="run.log")
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditEvaluator:
    def __init__(self, hparams: HyperParams) -> None:
        self.hparams = hparams
        self.editing_algorithm = hparams.alg_name
        self.test_with_generation = hparams.test_generation
        if self.test_with_generation:
            self.initialize_metrics_for_generation()
        else:
            self.initialize_metrics_for_teacher_forcing()

    def initialize_metrics_for_generation(self) -> None:
        """Initialize generation metrics for the first time."""
        assert self.test_with_generation
        LOG.info("Initializing evaluation metrics for test_generation=True")
        self.metrics = {
            "reliability": EditSuccessMetric(metric="edit_reliability", output_dir=self.hparams.results_dir),
            "edit_generality": EditSuccessMetric(metric="edit_generality", output_dir=self.hparams.results_dir),
            "edit_generality_image": EditSuccessMetric(
                metric="edit_generality_image", output_dir=self.hparams.results_dir
            ),
            "loc": EditSuccessMetric(metric="loc", output_dir=self.hparams.results_dir),
            "loc_image": EditSuccessMetric(metric="loc_image", output_dir=self.hparams.results_dir),
            "text_portability": EntityOverlapMetric(metric="text_portability", output_dir=self.hparams.results_dir),
            "text_portability_consistency": LocalityMetric(
                metric="text_portability_consistency", output_dir=self.hparams.results_dir, batch_key="text_portability"
            ),
            "image_portability": EntityOverlapMetric(metric="image_portability", output_dir=self.hparams.results_dir),
            "image_portability_consistency": LocalityMetric(
                metric="image_portability_consistency",
                output_dir=self.hparams.results_dir,
                batch_key="image_portability",
            ),
            "background": EntityOverlapMetric(metric="background", output_dir=self.hparams.results_dir),
        }

    def initialize_metrics_for_teacher_forcing(self) -> None:
        """Initialize teacher-forcing metrics for the first time."""
        LOG.info("Initializing evaluation metrics for test_generation=False")
        self.metrics = {
            "reliability": TFMetric(metric="tf_edit_reliability", output_dir=self.hparams.results_dir),
            "edit_generality": TFMetric(metric="tf_edit_generality", output_dir=self.hparams.results_dir),
            "edit_generality_image": TFMetric(metric="tf_edit_generality_image", output_dir=self.hparams.results_dir),
            "loc": TFMetric(metric="tf_loc", output_dir=self.hparams.results_dir),
            "loc_image": TFMetric(metric="tf_loc_image", output_dir=self.hparams.results_dir),
            "image_portability": EntityOverlapMetric(metric="image_portability", output_dir=self.hparams.results_dir),
            "text_portability": EntityOverlapMetric(metric="text_portability", output_dir=self.hparams.results_dir),
            # "text_portability": TFMetric(metric="tf_text_portability", output_dir=self.hparams.results_dir),
            # "text_portability_consistency": TFMetric(
            #     metric="tf_text_portability_consistency",
            #     output_dir=self.hparams.results_dir,
            #     batch_key="tf_text_portability",
            # ),
            # "image_portability": TFMetric(metric="tf_image_portability", output_dir=self.hparams.results_dir),
            # "image_portability_consistency": TFMetric(
            #     metric="tf_image_portability_consistency",
            #     output_dir=self.hparams.results_dir,
            #     batch_key="tf_image_portability",
            # ),
            # "background": TFMetric(metric="tf_background", output_dir=self.hparams.results_dir),
        }

    def evaluate_post_edit(
        self,
        model,
        exec_time,
        request_id: int,
        request: dict,
        pre_edit_results: dict = None,
        icl_examples=None,
    ):
        """Get the evaluation results after editing.

        This includes the locality metrics based on the pre-edit results.
        """
        if self.test_with_generation:
            return self._evaluate_post_edit_request_with_generation(
                model, exec_time, request_id, request, icl_examples=icl_examples
            )
        else:
            return self._evaluate_post_edit_request_with_teacher_forcing(
                model, exec_time, request_id, request, pre_edit_results=pre_edit_results, icl_examples=icl_examples
            )

    def _evaluate_post_edit_request_with_teacher_forcing(
        self,
        model,
        exec_time,
        request_id: int,
        request: dict,
        pre_edit_results: dict | None = None,
        icl_examples=None,
    ):
        """Evaluate a single request with teacher forcing."""
        if self.editing_algorithm == "IKE":
            results = {
                "request_id": request_id,
                "time": exec_time,
                "post": EVALUATION_UTILS_MAPPING["IKE_with_teacher_forcing"](
                    model=model,
                    processor=self.processor,
                    record=request,
                    metrics=self.metrics,
                    icl_examples=icl_examples,
                    add_new_fact=True,
                ),
            }
        else:
            results = {
                "request_id": request_id,
                "time": exec_time,
                "post": EVALUATION_UTILS_MAPPING["multimodal_with_teacher_forcing"](
                    model=model,
                    processor=self.processor,
                    record=request,
                    metrics=self.metrics,
                ),
            }

        return results

    def _evaluate_post_edit_request_with_generation(
        self,
        model,
        exec_time,
        request_id: int,
        request: dict,
        icl_examples=None,
    ):
        if self.editing_algorithm == "IKE":
            results = {
                "request_id": request_id,
                "time": exec_time,
                "post": EVALUATION_UTILS_MAPPING["IKE_with_generation"](
                    model=model,
                    metrics=self.metrics,
                    icl_examples=icl_examples,
                    record=request,
                    processor=self.processor,
                    add_new_fact=True,
                ),
            }
        else:
            results = {
                "request_id": request_id,
                "time": exec_time,
                "post": EVALUATION_UTILS_MAPPING["multimodal_with_generation"](
                    model=model,
                    metrics=self.metrics,
                    record=request,
                    processor=self.processor,
                ),
            }
        return results

    def evaluate_portability_with_judge(self):
        text_portability_file = self.metrics["text_portability"].output_file
        image_portability_file = self.metrics["image_portability"].output_file
        output_dir = Path(self.hparams.results_dir)
        results = {}
        llm_judge = BackgroundAccuracy(
            model_type="post-edit",
            run_name=wandb.run.name,
            wandb_project=wandb.run.project,
            predictions_path=text_portability_file,
            judge_model="Qwen/Qwen3-32B",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        
        llm_judge.scored_predictions_path = output_dir.joinpath("text_portability_judge_scored_predictions.json")
        results["text_judge_portability_accuracy"] = llm_judge.compute()
        llm_judge.load_prediction_samples(image_portability_file)
        llm_judge.scored_predictions_path = output_dir.joinpath("image_portability_judge_scored_predictions.json")
        results["image_judge_portability_accuracy"] = llm_judge.compute()

        wandb.log(results)
        return results
