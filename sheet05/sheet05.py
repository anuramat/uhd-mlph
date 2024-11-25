# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sheet 5

# %%
import os
import pandas as pd

# %% [markdown]
# ## 2 Logistic regression: an LLM lie detector

# %% [markdown]
# Download the data from https://heibox.uni-heidelberg.de/f/38bd3f2a9b7944248cc2/
# Unzip it and place the lie_detection folder in the folder named `data` to get the following structure:
# "data/lie_detection/datasets" and "data/lie_detection/acts".

# %% [markdown]
# This is how you can load a dataset of LLM activations. Use a new Datamanager if you want to have a new dataset. Use the same data manager if you want to combine datasets.

# %%
from lie_detection_utils import DataManager

path_to_datasets = "data/lie_detection/datasets"
path_to_acts = "data/lie_detection/acts"

# check if the datasets and activations are available
assert os.path.exists(path_to_datasets), "The path to the datasets does not exist."
assert os.path.exists(path_to_acts), "The path to the activations does not exist."

# these are the different datasets containing true and false factual statements about different topics
dataset_names = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"]
dataset_name = dataset_names[
    0
]  # choose some dataset from the above datasets, index "0" loads the "cities" dataset for example

# the dataloader automatically loads the training data for us
dm = DataManager()
dm.add_dataset(
    dataset_name,
    "Llama3",
    "8B",
    "chat",
    layer=12,
    split=0.8,
    center=False,
    device="cpu",
    path_to_datasets=path_to_datasets,
    path_to_acts=path_to_acts,
)
acts_train, labels_train = dm.get("train")  # train set
acts_test, labels_test = dm.get("val")
print(acts_train.shape, labels_train.shape)

# %%
# have a look at the statements that were fed to the LLM to produce the activations:
df = pd.read_csv(f"{path_to_datasets}/{dataset_name}.csv")
print(df.head(10))

# %% [markdown]
# ## 3 Log-sum-exp and soft(arg)max
# ### (b)

# %% [markdown]
# ### (c)

# %% [markdown]
# ## 4 Linear regions of MLPs
#
# ### (a)

# %%
import torch
import numpy as np
from torch import nn

get_num_params = lambda x: sum(p.numel() for p in x.parameters())


class MLP(nn.Module):
    def __init__(self, dims: list[int], example: None | torch.Tensor):
        super().__init__()
        """

        Args:
            dims: list of layer output sizes
        """
        layers = []
        for i in dims:
            layers.append(nn.LazyLinear(i))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

        if example is not None:
            self.net(example)

    def forward(self, x):
        return self.net(x)


# %%
model = MLP([20, 1], torch.Tensor([0, 0])).eval()
print("Number of parameters", get_num_params(model))

# %% [markdown]
# ### (b)

# %%
import matplotlib.pyplot as plt


def get_square_output(model: nn.Module, side: int, points: int = 500) -> torch.Tensor:
    assert side > 0
    x = y = np.linspace(-side, side, points)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    grid_tensor = torch.Tensor(grid)
    with torch.no_grad():
        outputs = model(grid_tensor).numpy()

    return outputs.reshape(points, points)


def plot_square_output(model: nn.Module, side: int, points: int = 500):
    outputs = get_square_output(model, side, points)

    plt.figure(figsize=(2, 2))
    plt.title(f"box {side}x{side} around origin")
    plt.imshow(outputs)
    plt.show()


def plot_square_grad(model: nn.Module, side: int, points: int = 500):
    output = get_square_output(model, side, points)

    xgr, ygr = np.gradient(output)

    plt.figure()
    plt.title("df/dx")
    plt.imshow(xgr, cmap="prism")
    plt.show()

    plt.figure()
    plt.title("df/dy")
    plt.imshow(ygr, cmap="prism")
    plt.show()


plot_square_output(model, 2)
plot_square_output(model, 20)
plot_square_output(model, 200)
plot_square_output(model, 2000)
plot_square_output(model, 20000)

# %% [markdown]
# Structure stops changing already at 20x20, because the network is shallow and
# parameters are initialized tightly around zero, including the bias term.

# %% [markdown]
# ### (c)

# %%
plot_square_grad(model, 20)

# %% [markdown]
# Here can see the regions of constant value of the function, since the
# activation function is piecewise linear. We can see that discontinuities
# intersect roughly at the origin, which once again demonstrates the small
# initial bias term

# %% [markdown]
# ### (d)

# %%
model = MLP([5, 5, 5, 5, 1], torch.Tensor([0, 0])).eval()
print("Number of parameters", get_num_params(model))

# %%
plot_square_output(model, 2)
plot_square_output(model, 20)
plot_square_output(model, 200)
plot_square_output(model, 2000)
plot_square_output(model, 20000)

# %%
plot_square_grad(model, 20)

# %% [markdown]
# Despite having almost the same number of parameters, the deeper model shows
# structure on a much larger scale, which is especially easy to see on the
# gradient plot. Successively applying ReLU and linear transformations allows us
# to gradually shift the inputs.
