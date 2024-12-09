# 3 Receptive Field of VGG16

## (a)

$$
r_l = (r_{l+1} - 1) * s_{l+1} + k_{l+1}
$$

Where
- $r_l$ -- the size of the receptive field of a pixel on layer $l$
- $s_l$ -- stride
- $k_l$ -- kernel size

Convolutional layers in VGG16 have $k=3,\ s=1$, while max pooling layers can be 
represented as convolutional layers with $k=2,\ s=2$ with constant weights.

| Layer                 | Receptive field                                  |
| -                     | -                                                |
| 18. MaxPool (Output): | $r_{18} = 1$                                     |
| 17. Conv:             | $r_{17} = (r_{18} - 1) * 2 + 2 = r_{18} * 2 = 2$ |
| 16. Conv:             | $r_{16} = (r_{17} - 1) * 1 + 3 = r_{17} + 2 = 4$ |
| 15. Conv:             | $r_{15} = 6$                                     |
| 14. MaxPool:          | $r_{14} = 8$                                     |
| 13. Conv:             | $r_{13} = 16$                                    |
| 12. Conv:             | $r_{12} = 18$                                    |
| 11. Conv:             | $r_{11} = 20$                                    |
| 10. MaxPool:          | $r_{10} = 22$                                    |
| 9. Conv:              | $r_9  = 44$                                      |
| 8. Conv:              | $r_8  = 46$                                      |
| 7. Conv:              | $r_7  = 48$                                      |
| 6. MaxPool:           | $r_6  = 50$                                      |
| 5. Conv:              | $r_5  = 100$                                     |
| 4. Conv:              | $r_4  = 102$                                     |
| 3. MaxPool:           | $r_3  = 104$                                     |
| 2. Conv:              | $r_2  = 208$                                     |
| 1. Conv:              | $r_1  = 210$                                     |
| 0. Input:             | $r_0  = 212$                                     |

## (b)

```python
s = lambda x, y: x**2 + y
n_params_conv = 3 * (64 * 2 + 128 * 2 + 256 * 3 + 512 * 3 + 512 * 3)
n_params_fc = (512 * 7 * 7) * (4096) + 4096**2 + 4096 * 1000
print("Total parameters:", "%.5E" % (n_params_conv + n_params_fc))
print("Ratio:", "%.5E" % (n_params_conv / n_params_fc))
```

Total parameters: 1.23646E+08
Ratio: 1.02496E-04
