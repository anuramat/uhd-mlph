layers = "iccmccmcccmcccmcccm"

vals= {
    'm': {'kernel': 2, 'stride':2, 'name':'MaxPool'},
    'c': {'kernel': 3, 'stride':1, 'name':'Conv'},
    'i': {'kernel': 1, 'stride':1, 'name':'Input'},
}

lines = []
stride = lambda l: vals[layers[l]]["stride"]
kernel = lambda l: vals[layers[l]]["kernel"]
name = lambda l: vals[layers[l]]["name"]

def r(l):
    res = 0
    if l == len(layers) - 1:
        res = 1
    else:
        higher = r(l + 1)
        res = (higher - 1) * stride(l + 1) + kernel(l + 1)
    lines.append(f"{name(l)}: r_{l} = {res}")
    return res


print("receptive field size:", r(0))
print('\n'.join(lines))

s = lambda x, y: x**2 + y
n_params_conv = 3 * (64 * 2 + 128 * 2 + 256 * 3 + 512 * 3 + 512 * 3)
n_params_fc = (512 * 7 * 7) * (4096) + 4096**2 + 4096 * 1000
print("Total parameters:", "%.5E" % (n_params_conv + n_params_fc))
print("Ratio:", "%.5E" % (n_params_conv / n_params_fc))
