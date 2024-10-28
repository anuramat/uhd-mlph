# Sheet 1

## 3. Ransac

Probability of a single $m$-subset being outlier-free:
$$
p^m
$$

P. of a single subset containing at least one outlier:
$$
1 - p^m
$$

P. of $r$ subsets having at least one outlier:
$$
(1-p^m)^r
$$

P. of at least one of the $r$ subsets not having outliers:
$$
P = 1 - (1-p^m)^r
$$

Now we can calculate the required $r$:
\begin{math}
P = 0.99 = 1 - (1-p^m)^r \\
\ln(0.01) = r \ln(1-p^m) \\
r = \frac{\ln(0.01)}{\ln(1-p^m)}
\end{math}

## 4. PCA meets Random Matrix Theory

### a) Directional distribution

Since the distribution is isotropic, all principal components will be uniformly 
distributed on a unit sphere.

### b) Eigenvalues

Due to the outliers, we can expect the largest eigenvalues to approach infinity, 
and the smallest eigenvalues to approach zero.
