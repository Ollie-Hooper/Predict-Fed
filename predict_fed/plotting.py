import matplotlib.pyplot as plt
import numpy as np


def rounded_scatter(pred, actual):
    pairs, counts = np.unique([f'{pred}|{actual}' for pred, actual in zip(pred, actual)], return_counts=True)
    # sort zeros
    d = dict(zip(pairs, counts))
    if '-0.0|0.0' in d:
        d['0.0|0.0'] = d['0.0|0.0'] + d['-0.0|0.0']
        del d['-0.0|0.0']
    pred = []
    actual = []
    counts = []
    for s, c in d.items():
        pred.append(float(s.split('|')[0]))
        actual.append(float(s.split('|')[1]))
        counts.append(c)
    sizes = np.array(counts)
    sizes = 5000 * sizes / sum(sizes)
    plt.scatter(pred, actual, s=sizes)
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    lim = abs(max([*pred, *actual], key=abs))
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.plot(np.linspace(-lim, lim), np.linspace(-lim, lim), c='orange')
    plt.show()
