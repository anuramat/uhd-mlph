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
# ### (a)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_data(dataset_name, split_name):
    # split_name - train/val
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
    x, y = dm.get(split_name)
    return x.numpy(), y.numpy().astype(int)


def get_multi(names, split_name):
    xl = []
    yl = []
    for name in names:
        x, y = get_data(name, split_name)
        xl.append(x)
        yl.append(y)

    return np.concatenate(xl, axis=0), np.concatenate(yl, axis=0)


for name in dataset_names:
    x_train, y_train = get_data(name, "train")
    x_val, y_val = get_data(name, "val")

    clf = LogisticRegression(C=99999).fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    print(f"dataset {name}, accuracy: {accuracy_score(y_val, y_pred)}")


# %% [markdown]
# Activation vectors are linearly separable, since we get a perfect accuracy.

# %% [markdown]
# ### (b)

# %%
datasets = dataset_names[1:]
print("validation datasets:", datasets)
for C in [1, 99999]:
    print(f"inverse regularization strength: {C}")
    for val_dataset in datasets:
        x_train, y_train = get_data("cities", "train")
        x_val, y_val = get_data(val_dataset, "val")
        clf = LogisticRegression(C=C).fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        print(f"validation: {val_dataset}, accuracy: {accuracy_score(y_val, y_pred)}")
    print("")

# %% [markdown]
# We get generalization, especially with regularization, but not on statements
# with negation.

# %% [markdown]
# ### (c)

# %%
x_train, y_train = get_multi(dataset_names[:2], "train")
x_val, y_val = get_multi(dataset_names[2:], "val")
clf = LogisticRegression().fit(x_train, y_train)
y_pred = clf.predict(x_val)
print(f"accuracy: {accuracy_score(y_val, y_pred)}")

# %% [markdown]
# After adding statements with negation to our training set, the situation improves.

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

# %% [markdown]
# # 5 Number of linear regions
#
# This is equivalent to finding the maximum number of regions a set of n lines can divide a plane into. If we have $n$ lines, the new line can introduce at most $n+1$ regions. Thus:
#
# $$
# a_{n+1} = a_n + n + 1
# $$
# $$
# a_0 = 1
# $$
# $$
# a_1 = 2
# $$
# Using induction hypothesis:
# $$
# a_n = \frac{(n+1)n}{2} + 1
# $$
# $$
# a_{n+1} = \frac{(n+1)(n+2)}{2} + 1 = a_n + n + 1
# $$
# qed
