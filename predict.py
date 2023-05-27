# %%
import json
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers.csv_logs import CSVLogger

from train import KaggleHousePrices

DATA_DIR = "./data/"
LOG_DIR = "./logs/"

# %%
with open(DATA_DIR + "best_config.json") as fp:
    params = json.load(fp)

# %%
model = KaggleHousePrices(DATA_DIR, **params)
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=50,
    logger=CSVLogger(LOG_DIR),
    log_every_n_steps=1,
    enable_progress_bar=False,
)
trainer.fit(model)

# %%
predicted = trainer.predict(model)[0]
df = pd.DataFrame(predicted, columns=["SalePrice"])
df.index += 1461
df.index.names = ["Id"]
df.to_csv(DATA_DIR + "submission.csv")

# %%
