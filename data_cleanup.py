# %%
import pandas as pd
import json
from itertools import *


# %%
def prepare_data(path):
    with open(path + "column_attributes.json") as file:
        attributes = json.load(file)

    numerical: list[str] = attributes["numerical"]
    na_values: dict[str, str] = attributes["na_values"]

    train_df = pd.read_csv(
        "data/train.csv", index_col="Id", na_values=na_values, keep_default_na=False
    ).convert_dtypes()
    test_df = pd.read_csv(
        "data/test.csv", index_col="Id", na_values=na_values, keep_default_na=False
    ).convert_dtypes()

    categoricals = train_df.columns.difference(numerical)

    df = pd.concat([train_df, test_df])
    df[categoricals] = df[categoricals].astype("category")

    na_keys = list(na_values.keys())
    df[numerical] = df[numerical].astype("float32")
    df[na_keys] = df[na_keys].fillna(df[na_keys].mean())

    df.to_pickle(path + "data.pkl")

    return df


# %%
