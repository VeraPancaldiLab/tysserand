"""
Comparison of reconstructed networks with a ground truth.
"""


__all__ = [
    'confusion_stats',
    'score_method',
]


def confusion_stats(set_true, set_test):
    """
    Count the true positives, false positives and false
    negatives in a test set with respect to a "true" set. 
    True negatives are not counted.
    """
    true_pos = len(set_true.intersection(set_test))
    false_pos = len(set_test.difference(set_true))
    false_neg = len(set_true.difference(set_test))
    
    return true_pos, false_pos, false_neg


def score_method(pairs_true, pairs_test):
    """
    Compute a performance score from the counts of
    true positives, false positives and false negatives
    of predicted pairs of nodes that are "double sorted".
    
    Examples
    --------
    >>> pairs_true = np.array([[3,4],
                               [5,6],
                               [7,8]])
    >>> pairs_test = np.array([[1,2],
                               [3,4],
                               [5,6]])
    >>> score_method(pairs_true, pairs_test)
    (0.5, 0.5, 0.25, 0.25)
    """
    
    set_true = {tuple(e) for e in pairs_true}
    set_test = {tuple(e) for e in pairs_test}
    true_pos, false_pos, false_neg = confusion_stats(set_true, set_test)
    
    total = true_pos + false_pos + false_neg
    true_pos_rate = true_pos / total
    false_pos_rate = false_pos / total
    false_neg_rate = false_neg / total
    
    return true_pos_rate, false_pos_rate, false_neg_rate
