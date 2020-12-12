from typing import Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def make_uniform(labels: np.array, max_cnt: int) -> np.array:
    selection = np.zeros_like(labels, dtype=bool)
    classes = np.unique(labels)
    for c in classes:
        cnt = 0
        for i, y in enumerate(labels):
            if cnt == max_cnt:
                break
            bool_val = y == c
            if bool_val:
                cnt += int(bool_val)
                selection[i] = bool_val
    return selection


def print_model(model: object, X: np.array, y_true: np.array) -> None:
    y_pred = model.predict(X)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def get_uniform_version(X: np.array, y: np.array) -> Tuple[np.array]:
    classes = np.unique(y)
    classes.sort()
    x = [np.sum(y == c) for c in classes]
    split_idx = np.argmin(x)
    selections = make_uniform(y, x[split_idx])
    return X[selections], y[selections]
