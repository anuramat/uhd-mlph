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
# # Sheet 1

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from numpy import linalg as LA

# %matplotlib inline

# %% [markdown]
# ## 1 Principal Component Analysis
# ### (a)

# %%
# TODO: implement PCA (fill in the blanks in the function below)


def pca(data, n_components=None):
    """
    Principal Component Analysis on a p x N data matrix.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (p, N).
    n_components : int, optional
        Number of requested components. By default returns all components.

    Returns
    -------
    np.ndarray, np.ndarray
        the pca components (shape (n_components, p)) and the projection (shape (n_components, N))

    """
    # set n_components to p by default
    n_components = data.shape[0] if n_components is None else n_components
    assert (
        n_components <= data.shape[0]
    ), f"Got n_components larger than dimensionality of data!"

    # center the data
    mean = np.mean(data, axis=1, keepdims=True)
    X = data - mean

    # compute X times X transpose
    covmat = X @ X.T

    # compute the eigenvectors and eigenvalues
    eigval, eigvec = LA.eig(covmat)
    assert all(eigval > 0)
    assert np.isreal(eigvec).all()

    # sort the eigenvectors by eigenvalue and take the n_components largest ones
    indices = np.argsort(eigval)
    # weirdly, numpy docs don't tell us if the it's decreasing or increasing:
    if eigval[indices[0]] < eigval[indices[-1]]:
        indices = indices[::-1]
    components = eigvec[:, indices[:n_components]].T
    eigval = eigval[indices]

    # compute X_projected, the projection of the data to the components
    X_projected = components @ X

    return (
        components,
        X_projected,
    )  # return the n_components first components and the pca projection of the data


# %%
# Example data to test your implementation
# All the asserts on the bottom should go through if your implementation is correct

data = np.array(
    [[1, 0, 0, -1, 0, 0], [0, 3, 0, 0, -3, 0], [0, 0, 5, 0, 0, -5]], dtype=np.float32
)

# add a random offset to all samples. it should not affect the results
data += np.random.randn(data.shape[0], 1)

n_components = 2
components, projection = pca(
    data, n_components=n_components
)  # apply your implementation

# the correct results are known (up to some signs)
true_components = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32)
true_projection = np.array([[0, 0, 5, 0, 0, -5], [0, 3, 0, 0, -3, 0]], dtype=np.float32)

# check that components match, up to sign
assert isinstance(
    components, np.ndarray
), f"Expected components to be numpy array but got {type(components)}"
assert (
    components.shape == true_components.shape
), f"{components.shape}!={true_components.shape}"
assert np.allclose(
    np.abs(components * true_components).sum(1), np.ones(n_components)
), f"Components not matching"

# check that projections agree, taking into account potentially flipped components
assert isinstance(
    projection, np.ndarray
), f"Expected projection to be numpy array but got {type(projection)}"
assert projection.shape == (
    n_components,
    data.shape[1],
), f"Incorrect shape of projection: Expected {(n_components, data.shape[1])}, got {projection.shape}"
assert np.allclose(
    projection,
    true_projection * (components * true_components).sum(1, keepdims=True),
    atol=1e-6,
), f"Projections not matching"

print("Test successful!")

# %% [markdown]
# ### (b)

# %% [markdown]
# Load the data (it is a subset of the data at https://opendata.cern.ch/record/4910#)

# %%
features = np.load("data/dijet_features.npy")
labels = np.load("data/dijet_labels.npy")
label_names = ["b", "c", "q"]  # bottom, charm or light quarks

print(f"{features.shape=}, {labels.shape=}")  # print the shapes

# print how many samples of each class are present in the data (hint: numpy.unique)
print(f"unique labels: {np.unique(labels)}")

# %% [markdown]
# Normalize the data

# %%
# TODO: report range of features and normalize the data to zero mean and unit variance


# %% [markdown]
# ### (c)
# Compute a 2D PCA projection and make a scatterplot of the result, once without color, once coloring the dots by label. Interpret your results.

# %%
# TODO: apply PCA as implemented in (a)


# %%
# TODO: make a scatterplot of the PCA projection


# %%
# TODO: make a scatterplot, coloring the dots by their label and including a legend with the label names
# (hint: one way is to call plt.scatter once for each of the three possible labels. Why could it be problematic to scatter the data sorted by labels though?)


# %% [markdown]
# ## 2 Nonlinear Dimension Reduction

# %%
import umap  # import umap-learn, see https://umap-learn.readthedocs.io/

# %%
# if you have not done 1(b) yet, you can load the normalized features directly:
features = np.load("data/dijet_features_normalized.npy")
labels = np.load("data/dijet_labels.npy")
label_names = ["b", "c", "q"]  # bottom, charm or light quarks

# %% [markdown]
# ### (a)

# %%
# TODO: Apply umap on the normalized jet features from excercise 1. It will take a couple of seconds.
# note: umap uses a different convention regarding the feature- and sample dimension, N x p instead of p x N!

reducer = umap.UMAP()

# %%
# TODO: make a scatterplot of the UMAP projection

# TODO: make a scatterplot, coloring the dots by their label and including a legend with the label names

# %% [markdown]
# ### (b)

# %%
for n_neighbors in (2, 4, 8, 15, 30, 60, 100):
    # TODO: repeat the above, varying the n_neighbors parameter of UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors)
