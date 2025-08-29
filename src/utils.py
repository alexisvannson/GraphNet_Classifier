from sklearn.metrics import roc_auc_score

def compute_auroc_binary(y_true, y_score):
    """
    Args:
        y_true: array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_score: array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or binary decisions.
    Returns:
        float: AUROC value. Returns NaN if AUROC cannot be computed (e.g., only one class present).
    """
    try:
        # roc_auc_score computes the area under the ROC curve for binary classification.
        return roc_auc_score(y_true, y_score)
    except ValueError:
        # If computation is not possible (e.g., only one class in y_true), return NaN.
        return float('nan')


def compute_auroc_multiclass(y_true, y_score, average="macro"):
    """
    Args:
        y_true: array-like of shape (n_samples,)
            True class labels as integers.
        y_score: array-like of shape (n_samples, n_classes)
            Target scores for each class, typically predicted probabilities.
        average: str, default="macro"
            Determines the type of averaging performed on the data:
            - "macro": Calculate metrics for each label, and find their unweighted mean.
            - "weighted": Calculate metrics for each label, and find their average weighted by support.

    Returns:
        float: Mean AUROC over classes. Returns NaN if AUROC cannot be computed.
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_classes = y_score.shape[1]

    # If only two classes, fallback to binary
    if n_classes == 2:
        return compute_auroc_binary(y_true, y_score[:, 1])

    try:
        return roc_auc_score(y_true, y_score, multi_class="ovr", average=average)
    except ValueError:
        return float('nan')

def plot_roc_curve_binary(y_true, y_score, ax=None, title="ROC Curve (Binary)", **kwargs):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_score: array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or binary decisions.
        ax: matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise creates a new figure.
        title: str
            Title for the plot.
        **kwargs: Additional keyword arguments for matplotlib plot.

    Returns:
        ax: matplotlib Axes object with the ROC curve.
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})', **kwargs)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def plot_roc_curve_multiclass(y_true, y_score, class_names=None, ax=None, title="ROC Curve (Multiclass OvR)", **kwargs):
    """
    Args:
        y_true: array-like of shape (n_samples,)
            True class labels as integers.
        y_score: array-like of shape (n_samples, n_classes)
            Target scores for each class, typically predicted probabilities.
        class_names: list of str, optional
            Names of the classes. If None, uses integer class labels.
        ax: matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise creates a new figure.
        title: str
            Title for the plot.
        **kwargs: Additional keyword arguments for matplotlib plot.

    Returns:
        ax: matplotlib Axes object with the ROC curves.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import numpy as np

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_classes = y_score.shape[1]

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    if ax is None:
        fig, ax = plt.subplots()

    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, lw=2, color=colors(i),
            label=f"Class {class_names[i]} (AUC = {roc_auc:.2f})",
            **kwargs
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize='small')
    return ax
