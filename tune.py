# %%
import torch
import pytorch_lightning as pl
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from train import KaggleHousePrices
from pytorch_lightning.loggers.csv_logs import CSVLogger
import json

DATA_DIR = "./data/"
LOG_DIR = "./logs/"

ray.init(num_gpus=1, num_cpus=8)


# %%
def train_model(config, data_dir, log_dir, max_epochs):
    assert torch.cuda.is_available()

    model = KaggleHousePrices(data_dir, **config)
    metrics = {"loss": "val_loss"}
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_progress_bar=False,
        logger=CSVLogger(save_dir=log_dir),
        log_every_n_steps=1,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
    )
    trainer.fit(model)


# %%
config = dict(
    hidden_layers=[
        tune.choice([128, 256, 512]),
        tune.choice([64, 128, 256]),
        tune.choice([32, 64, 128]),
        tune.choice([16, 32, 64]),
        tune.choice([4, 8, 16]),
        tune.choice([4, 8, 16]),
    ],
    learning_rate=tune.loguniform(1e-4, 1),
    weight_decay=tune.loguniform(1e-5, 1e-1),
)

# %%
trainable = tune.with_parameters(
    train_model,
    data_dir=DATA_DIR,
    log_dir=LOG_DIR,
    max_epochs=50,
)

trainable = tune.with_resources(trainable, resources=dict(gpu=1, cpu=6))

# %%
analysis = tune.run(
    trainable,
    metric="loss",
    mode="min",
    config=config,
    num_samples=50,
    name="tune_kaggle_housing_prices",
    chdir_to_trial_dir=False,
)

best_config = analysis.best_config
print(best_config)
with open(DATA_DIR + "best_config.json", "w") as fp:
    json.dump(best_config, fp)

# %%
