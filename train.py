# %%
import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Subset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder
from data_cleanup import prepare_data


TRAIN_TO_DEV_TO_TEST_RATIO = np.array([4, 2, 1])


# %%
def onehot_encode(
    df: pd.DataFrame, columns: list[str], encoder=None
) -> tuple[pd.DataFrame, OneHotEncoder]:
    encoder = OneHotEncoder(sparse_output=False)
    cols = df.loc[:, columns]
    encoded = encoder.fit_transform(cols)
    features = encoder.get_feature_names_out(columns)
    encoded = pd.DataFrame(encoded, columns=features, index=df.index)
    other_cols = df.loc[:, ~df.columns.isin(columns)]
    df = pd.concat([other_cols, encoded], axis=1)

    return df, encoder


# %%
class KaggleHousePricesDataset(TensorDataset):
    def __init__(self, path):
        df: pd.DataFrame = pd.read_pickle(path)
        categoricals = df.select_dtypes("category")
        cat_columns = categoricals.columns.tolist()
        df, encoder = onehot_encode(df, cat_columns)
        columns = df.columns.tolist()
        columns.remove("SalePrice")
        columns.append("SalePrice")
        df = df[columns]
        df = df.astype(np.float32)

        encoded_columns = encoder.get_feature_names_out(cat_columns)
        mean = df.mean(axis=0)
        mean[encoded_columns] = 0
        std = df.std(axis=0)
        std[encoded_columns] = 1

        self.df: pd.DataFrame = df
        self.encoder: OneHotEncoder = encoder
        self.mean = mean
        self.std = std

        values = torch.Tensor(self.df.values)
        values = (values - mean.values) / std.values
        super().__init__(values)


# %%
from typing import Any


class KaggleHousePrices(pl.LightningModule):
    def __init__(
        self,
        data_dir,
        hidden_layers: list[int],
        batch_size=5000,
        learning_rate=5e-4,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.save_hyperparameters(ignore=["data_dir"])

        def layers():
            for l in self.hparams.hidden_layers:
                yield nn.LazyLinear(l, bias=False)
                yield nn.BatchNorm1d(l)
                yield nn.PReLU()

            # output layer
            yield nn.LazyLinear(1)

        self.model = nn.Sequential(*layers())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        x = batch[:, :-1]
        y = batch[:, -1]
        y_hat = self(x)
        y = y.resize_(y_hat.size())
        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", torch.sqrt(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        x = batch[:, :-1]
        y = batch[:, -1]
        y_hat = self(x)
        y = y.resize_(y_hat.size())
        loss = F.mse_loss(y_hat, y)

        self.log("val_loss", torch.sqrt(loss))

    def test_step(self, batch, batch_idx):
        batch = batch[0]
        x = batch[:, :-1]
        y = batch[:, -1]
        y_hat = self(x)
        y = y.resize_(y_hat.size())
        loss = F.mse_loss(y_hat, y)

        self.log("test_loss", torch.sqrt(loss))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch = batch[0]
        x = batch[:, :-1]
        y_hat = self(x)
        y_hat = y_hat.cpu().data.numpy()
        mean = self.dataset.mean.values[-1]
        std = self.dataset.std.values[-1]

        return y_hat * std + mean

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def prepare_data(self) -> None:
        prepare_data(self.data_dir)

    def setup(self, stage=None) -> None:
        self.dataset = KaggleHousePricesDataset(self.data_dir + "data.pkl")

        train_dataset = Subset(self.dataset, range(1460))
        train_dataset_len = len(train_dataset)
        split = np.floor(
            TRAIN_TO_DEV_TO_TEST_RATIO
            / TRAIN_TO_DEV_TO_TEST_RATIO.sum()
            * train_dataset_len
        )
        split[0] = train_dataset_len - split[1:].sum()
        split = split.astype("int")

        self.data_train, self.data_val, self.data_test = random_split(
            train_dataset, split
        )

        self.data_eval = Subset(self.dataset, range(1460, self.dataset.__len__()))

    def train_dataloader(self):
        return DataLoader(
            self.data_train, batch_size=self.hparams.batch_size, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.hparams.batch_size, num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=self.hparams.batch_size, num_workers=2
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_eval, batch_size=self.hparams.batch_size, num_workers=4
        )


# %%
# L.seed_everything(42, workers=True)
# model = KaggleHousePrices(
#     data_dir=DATA_DIR,
#     hidden_layers=[200, 100, 50, 25, 25, 12, 10, 8],
#     learning_rate=0.84,
#     weight_decay=1e-3,
#     batch_size=2000,
# )
# trainer = L.Trainer(
#     accelerator="auto",
#     devices=1,
#     logger=CSVLogger(save_dir="./logs/"),
#     max_epochs=250,
#     log_every_n_steps=1,
#     callbacks=[EarlyStopping("val_loss", patience=20)],
# )

# trainer.fit(model)
# trainer.test(model)

# %%
# predicted = trainer.predict(model)[0]
# df = pd.DataFrame(predicted, columns=["SalePrice"])
# df.index += 1461
# df.index.names = ["Id"]
# df.to_csv(DATA_DIR + "submission.csv")

# %%
