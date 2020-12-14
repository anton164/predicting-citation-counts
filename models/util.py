from typing import Tuple, List, Dict, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def make_uniform(labels: np.array, max_cnt: int) -> np.array:
    selection = np.zeros(len(labels), dtype=bool)
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


def sample_indices(
    labels: np.array, N: int, random_state: Optional[int] = None
) -> List[int]:
    """
    Performs oversampling/undersampling depending on the sample size
    and the unique label counts

    Returns of sampled indices, where each class is sampled N items
    """
    class_counts = np.unique(labels, return_counts=True)
    classes = class_counts[0].reshape(-1)
    selected_indices = []
    if random_state is not None:
        np.random.seed(random_state)
    for c in classes:
        possible_indices = np.argwhere(labels == c).reshape(-1)
        if len(possible_indices) < N:
            # Oversample this class
            oversampled_count = N - len(possible_indices)
            selected_indices += possible_indices.tolist()
            selected_indices += np.random.choice(
                possible_indices, oversampled_count, replace=True
            ).tolist()
        elif len(possible_indices) == N:
            selected_indices += possible_indices.tolist()
        else:
            # Undersample this class
            selected_indices += np.random.choice(
                possible_indices, N, replace=False
            ).tolist()
    return selected_indices


def print_model(model: object, X: np.array, y_true: np.array) -> None:
    y_pred = model.predict(X)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def get_uniform_version(
    X: np.ndarray,
    y: np.ndarray,
    oversampling_rate: float = 1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray]:
    class_counts = np.unique(y, return_counts=True)
    under_represented_class = class_counts[0][np.argmin(class_counts[1])]
    under_represented_class_count = np.min(class_counts[1])
    selected_indices = sample_indices(
        y, np.int(under_represented_class_count * oversampling_rate), random_state
    )
    return X[selected_indices], y[selected_indices]


def average_scores(model_scores: Dict[str, float]) -> Dict[str, float]:
    models = model_scores[0].keys()
    average_model_scores = {}
    for model in models:
        sum_score = 0
        for scores in model_scores:
            sum_score += scores[model]

        average_model_scores[model] = sum_score / len(model_scores)

    return average_model_scores