import datetime
import logging

import pandas as pd

import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class WandbLogger:
    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        tags: str | None = None,
    ):
        self.run_name = self._get_run_name(run_name)
        if wandb.run is not None:
            logger.warning(
                f"Already in a wandb run: {wandb.run.name}. Using existing run instead of creating a new one."
            )
            self.run = wandb.run
        else:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=self.run_name,
                tags=tags,
            )

    def log_metrics(self, metrics: dict):
        wandb.log(metrics)

    def log_table(self, results: list[dict], table_name: str = "results"):
        table = wandb.Table(dataframe=pd.DataFrame(results))
        wandb.log({f"Tables/{table_name}": table})

    def finish(self):
        self.run.finish()

    def _get_run_name(self, run_name: str | None) -> str:
        if wandb.run is not None:
            return wandb.run.name
        time = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if run_name is None:
            return f"run_{time}"
        return f"{run_name}_{time}"
