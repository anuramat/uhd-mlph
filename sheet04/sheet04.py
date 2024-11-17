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
# # Sheet 4

# %%
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# ## 3 QDA
# ### (a)

# %%
pts = np.load("data/data1d.npy")
labels = np.load("data/labels1d.npy")

# %%
neg_pts = pts[labels == 0]
neg_mu = neg_pts.mean()
neg_sigma = neg_pts.std()

pos_pts = pts[labels == 1]
pos_mu = pos_pts.mean()
pos_sigma = pos_pts.std()

# %% [markdown]
# ### (b)

# %%
plt.title("basic histogram")
plt.hist(neg_pts, alpha=0.5, bins=10, label="negative")
plt.hist(pos_pts, alpha=0.5, bins=10, label="positive")

# %%
from scipy.stats import norm


def qda_plot(neg_prior: float):
    pos_prior = 1 - neg_prior

    x_min = -10
    x_max = 10

    x = np.linspace(x_min, x_max, 1000)
    neg_gauss: np.ndarray = norm(loc=neg_mu, scale=neg_sigma).pdf(x)  # pyright: ignore
    pos_gauss: np.ndarray = norm(loc=pos_mu, scale=pos_sigma).pdf(x)  # pyright: ignore
    prob_norm = neg_gauss + pos_gauss

    neg_post = neg_gauss * neg_prior / prob_norm
    pos_post = pos_gauss * pos_prior / prob_norm

    plt.plot(x, neg_gauss, label="negative gaussian")
    plt.plot(x, pos_gauss, label="positive gaussian")
    plt.plot(x, neg_post, label="negative posteriori")
    plt.plot(x, pos_post, label="positive posteriori")
    plt.title("gaussians and corresponding posterior distributions")
    plt.grid()
    plt.legend()
    plt.show()


# %%
qda_plot(0.5)

# %%
qda_plot(2.0 / 3)

# %% [markdown]
# ## 4 Trees and Random Forests

# %% [markdown]
# ### (b)

# %%
# load the data
pts = np.load("data/data1d.npy")
labels = np.load("data/labels1d.npy")

# TODO: Sort the points to easily split them
sorted_indices = np.argsort(pts)
pts = pts[sorted_indices]
labels = labels[sorted_indices]

# TODO: Implement or find implementation for Gini impurity, entropy and misclassifcation rate


def probabilities(partition):
    # divide counts by size of dataset to get cluster probabilites
    return np.unique(partition, return_counts=True)[1] / len(partition)


def compute_split_measure(l, l0, l1, method):
    p0 = probabilities(l0)
    p1 = probabilities(l1)
    p = probabilities(l)
    return method(p) - (len(l0) * method(p0) + len(l1) * method(p1)) / (len(l))


# TODO: Iterate over the possible splits, evaulating and saving the three criteria for each one
# TODO: Then, Compute the split that each criterion favours and visualize them
#       (e.g. with a histogram for each class and vertical lines to show the splits)


# %% [markdown]
# ### (b)

# %%
# load the dijet data
features = np.load("data/dijet_features_normalized.npy")
labels = np.load("data/dijet_labels.npy")

# TODO: define train, val and test splits as specified (make sure to shuffle the data before splitting it!)

# %%
from sklearn.ensemble import RandomForestClassifier

# TODO: train a random forest classifier for each combination of specified hyperparameters
#       and evaluate the performances on the validation set.

# %%
# TODO: for your preferred configuration, evaluate the performance of the best configuration on the test set
