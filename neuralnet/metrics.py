def _check_shape(A, Y):
    """
    Helper method to run common shape checks for predictions and targets.
    """
    # Make sure the predictions and the targets have the same shape.
    assert A.shape == Y.shape, 'A and Y are of different shapes: {a}, {y}'.format(a=A.shape, y=Y.shape)

    # Make sure that predictions and targets contain more than one observation.
    assert (len(A.shape) > 1 and A.shape[0] > 1), \
        'A has shape {a}, but it must contain more than one observation'.format(a=A.shape)


def accuracy(A, Y):
    """
    Accuracy statistic for a vector of predictions `A` given a vector of targets
    `Y`, i.e. the proportion of correctly classified samples:

             TP + TN
        -----------------
        TP + FP + TN + FN
    """
    _check_shape(A, Y)

    pred_classes = np.argmax(A, axis=1)

    numerator = sum(1 for x, y in zip(pred_classes, Y) if np.array_equal(x, y))
    denominator = A.shape[0]

    return numerator / denominator


def precision(A, Y):
    """
    Precision statistic for a vector of predictions `A` given a vector of targets
    `Y`, i.e. the proportion of positive classifications that are truly positive:

          TP
        ------
        TP + FP
    """
    pass


def recall(A, Y):
    """
    Recall statistic for a vector of predictions `A` given a vector of targets
    `Y`, i.e. the proportion of truly positive samples that are identified as positive:

          TP
        -------
        TP + FN
    """
    pass


def roc_auc(A, Y):
    """
    The area under the curve (AUC) of the Receiver Operator Characteristic (ROC),
    i.e. the measure of the shape of the relationship between the true-positive
    rate and false-positive rate for all classification thresholds.

    An intuitive interpretation of the statistic is: If you ran an experiment
    where you randomly sampled one positive observation and one negative observation,
    in what proportion of the samples would the model score the positive sample
    higher than the negative sample?

    For more info on ROC AUC as a metric, see:
    https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    """
    pass
