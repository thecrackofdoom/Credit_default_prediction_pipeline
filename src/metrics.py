def KS_statistic(y_true, y_pred_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    ks_statistic = max(tpr - fpr)
    return ks_statistic

def KS_stat(y_true, y_pred_prob):
    import scipy.stats as stats
    # Split the probabilities based on the true class
    prob_positive = y_pred_prob[y_true == 1]
    prob_negative = y_pred_prob[y_true == 0]
    scipy_ks_statistic = stats.ks_2samp(prob_positive, prob_negative).statistic
    return scipy_ks_statistic

def gini_coefficient(y_true, y_pred_prob):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred_prob)
    gini = 2 * auc - 1
    return gini