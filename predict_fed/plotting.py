import matplotlib.pyplot as plt
import numpy as np


def rounded_scatter(pred, actual):
    pairs, counts = np.unique([f'{pred}|{actual}' for pred, actual in zip(pred, actual)], return_counts=True)
    # sort zeros
    d = dict(zip(pairs, counts))
    for i in ['-1.0', '-0.75', '-0.5', '-0.25', '0.0', '0.25', '0.5', '0.75', '1.0']:
        if f'-0.0|{i}' in d:
            if f'0.0|{i}' not in d:
                d[f'0.0|{i}'] = d[f'-0.0|{i}']
            else:
                d[f'0.0|{i}'] = d[f'0.0|{i}'] + d[f'-0.0|{i}']
            del d[f'-0.0|{i}']
    pred = []
    actual = []
    counts = []
    for s, c in d.items():
        pred.append(float(s.split('|')[0]))
        actual.append(float(s.split('|')[1]))
        counts.append(c)
    sizes = np.array(counts)
    sizes = 5000 * sizes / sum(sizes)
    plt.scatter(pred, actual, s=sizes, alpha=0.5)
    plt.xlabel('Predicted Change in Interest Rate rounded (%)')
    plt.ylabel('Actual Change in Interest Rate rounded (%)')
    lim = abs(max([*pred, *actual], key=abs))
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.plot(np.linspace(-lim, lim), np.linspace(-lim, lim), c='orange')
    plt.show()

def plot_metrics(performance):
    history = performance[2]
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss', pad=13)
    plt.legend(loc='upper right')
    plt.show()


def plot_pred(y_pred, y_pred_rounded, y_test):
    x = [-1, 1]
    y = [-1, 1]

    # y_pred
    plt.grid()
    plt.scatter(y_pred, y_test)
    plt.plot(x, y, color='orange')
    plt.ylabel("Actual Change in Interest Rate (%)")
    plt.xlabel("Predicted Change in Interest Rate (%)")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

    # y_pred_rounded§