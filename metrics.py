import jax.numpy as jnp

def precision_recall_f1(y_real, y_pred):
    classes = jnp.unique(y_real)
    precisions = []
    recalls = []
    f1s = []
    
    for c in classes:
        tp = jnp.sum((y_pred == c) & (y_real == c))
        fp = jnp.sum((y_pred == c) & (y_real != c))
        fn = jnp.sum((y_pred != c) & (y_real == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s

def confusion_matrix(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = len(jnp.unique(y_true))
    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    for i in range(num_classes):
        for j in range(num_classes):
            cm = cm.at[i, j].set(jnp.sum((y_true == i) & (y_pred == j)))
    return cm