import sklearn
from sklearn.metrics import *
import numpy as np
import seaborn as sns

def display_confusion_matrix(y, pred, ax=None, normalize_display="all", cmap="bone"):
    cf_matrix = sklearn.metrics.confusion_matrix(y, pred)
    cf_norm_row = sklearn.metrics.confusion_matrix(y, pred, normalize="true")
    cf_norm_col = sklearn.metrics.confusion_matrix(y, pred, normalize="pred")

    group_names = ['TN','FP','FN','TP']
    group_counts = [
        f"{value:>5d}"
        for value in cf_matrix.flatten()
    ]

    group_percentages_row = [
        f"{value:.2f}"
        for value in cf_norm_row.flatten()
    ]

    group_percentages_col = [
        f"{value:.2f}"
        for value in cf_norm_col.flatten()
    ]

    labels = [
        # f"{g_name}:\ncount / true / pred\n{g_count} / {g_p_row} / {g_p_col}"
        f"{g_name}:\n{g_count} / {g_p_row} / {g_p_col}"
        for g_name, g_count, g_p_row, g_p_col in zip(group_names, group_counts, group_percentages_row, group_percentages_col)
    ]

    labels = np.asarray(labels).reshape(2,2)
    cf_display = sklearn.metrics.confusion_matrix(y, pred, normalize=normalize_display)
    sns.heatmap(cf_display, annot=labels, fmt="", cmap=cmap, cbar=False, ax=ax)

    if ax is not None:
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")




if __name__ == "__main__":
    s = np.random.uniform(0, 1, size=100000)

    # v1 = (s > 0.6).astype(int)
    # v2 = (s > 0.4).astype(int)

    # print(v1)
    # print(v2)
    # print(f1_loss(torch.tensor(v1), torch.tensor(v2)))
    # print(f1_loss(torch.tensor(s > 0.5), torch.tensor(s)))
    loss_fn = F1Loss()
    print(loss_fn(torch.tensor(s > 0.5), torch.tensor(s)))
