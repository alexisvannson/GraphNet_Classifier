
def compute_auroc_sklearn(y_true, y_score, binary=True):
    from sklearn.metrics import roc_auc_score
    if binary:
        return roc_auc_score(y_true, y_score)
    else:
        return roc_auc_score(y_true, y_score, multi_class='ovr')


def roc_curve_sklearn(y_true, y_score, binary=True):
    from sklearn.metrics import roc_curve
    if binary:
        return roc_curve(y_true, y_score)
    else:
        return roc_curve(y_true, y_score, multi_class='ovr')

def plot_roc_curve_sklearn(y_true, y_score, binary=True, show=False, to_save=True, output_path='roc_curve.png'):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    if binary:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
    else:
        # For multiclass, we need to handle it differently
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        
        # Assuming y_true contains class labels and y_score contains probabilities
        n_classes = y_score.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure()
        colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
        plt.legend(loc="lower right")
    
    if show:
        plt.show()
    if to_save:
        plt.savefig(output_path)
    plt.close()
