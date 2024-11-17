# Sheet 4

## 4 Trees and Random Forests

### (a)

$$
\begin{gathered}
    H_M = 1 - \max_c p(y=c) \\
    H_G = 1 - \sum_c p(y=c)^2 \\
    H_E = - \sum_c p(y=c) \log p(y=c) \\
    \Delta H = H_0 - \frac{N_l H_l + N_r H_r}{N_l + N_r}
\end{gathered}
$$

$C=2, N=400$ data points per class, denoted as $(400,400)$

$$
\begin{gathered}
    H_{M_0} = 1 - \frac{1}{2} = \frac{1}{2} \\
    H_{G_0} = 1 - 2 \left(\frac{1}{2}\right) ^ 2 = \frac{1}{2} \\
    H_{E_0} = -\log \frac{1}{2} = 1
\end{gathered}
$$

#### Split A: $(300,100), (100,300)$

##### Misclassification error

$$
\begin{gathered}
    H_{M_l} = H_{M_r} = 1 - \frac{3}{4} = \frac{1}{4} \\
    \Delta H_M = \frac{1}{4} = 0.25
\end{gathered}
$$

##### Gini

$$
\begin{gathered}
    H_{G_l} = H_{G_r} =
        1 - \left(\frac{3}{4}\right) ^ 2 - \left(\frac{1}{4}\right) ^ 2 =
        \frac{6}{16} \\
    \Delta H_G = \frac{1}{2} - \frac{6}{16} = \frac{2}{16} = 0.125
\end{gathered}
$$

##### Entropy

$$
\begin{gathered}
    H_{E_l} = H_{E_r} = - \frac{3}{4} \log \frac{3}{4} - \frac{1}{4} \log
        \frac{1}{4} = \frac{1}{2} - \frac{3}{4} \log \frac{3}{4} \\
    \Delta H_E = 1 - \frac{1}{2} + \frac{3}{4} \log \frac{3}{4} = 0.18872
        \ldots
\end{gathered}
$$

#### Split B: $(200,0), (200,400)$

##### Misclassification error

$$
\begin{gathered}
    H_{M_l} = 0 \\
    H_{M_r} = 1 - \frac{2}{3} = \frac{1}{3} \\
    \Delta H_M = \frac{1}{6} = 0.16666\ldots
\end{gathered}
$$

##### Gini

$$
\begin{gathered}
    H_{G_l} = 0 \\
    H_{G_r} = 1 - \left(\frac{1}{3}\right)^2 - \left(\frac{2}{3}\right)^2 =
        \frac{4}{9} \\
    \Delta H_G = \frac{1}{2} - \frac{3}{4} \cdot \frac{4}{9} = \frac{1}{6} =
        0.16666\ldots
\end{gathered}
$$

##### Entropy

$$
\begin{gathered}
    H_{E_l} = 0 \\
    H_{E_r} = - \frac{1}{3} \log \frac{1}{3} - \frac{2}{3} \log \frac{2}{3} \\
    \Delta H_E = 1 - \frac{3}{4} H_{E_r} = 0.31127\ldots
\end{gathered}
$$

#### Results

Both Gini index and entropy indicate that the split B is the better choice, 
meanwhile misclassification error "prefers" split A.
