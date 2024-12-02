# 2 Bonus: Training of an MLP

## (a)

![No regularization](./pics/mlp.png) 

## (b)

We expect both variants to have smaller activations on average. Additionally, we 
expect L1 regularization to lead to sparse activations.

![L1 regularization](./pics/mlp_l1.png) 

![L2 regularization](./pics/mlp_l2.png) 

We observe sparse regularizations in L1 case, but with L2 the difference with 
the unregularized network is minimal.
