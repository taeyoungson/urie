import numpy as np

from math import log10
from sklearn.metrics import average_precision_score


def count_match(prediction, label, axis=1, is_binary=False):
    # give match count given pred and ground truth
    if prediction.is_cuda:
        prediction = np.squeeze(prediction.detach().cpu().numpy())

    if label.is_cuda:
        label = np.squeeze(label.detach().cpu().numpy())

    if is_binary:
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        return np.count_nonzero(prediction == label)

    return np.count_nonzero(
        np.argmax(prediction, axis=axis) == label
    )


def mean_average_precision(prediction, labels, num_class=20):
    '''
    This functions is designed to evaluate performance of multi-class \
        multi-label classification task
    Args:
        prediction: (dataset_size, num_class) outputs of model
        label: (dataset_size, num_class) binary answer labels
    '''
    mAP = 0
    for i in range(num_class):
        pred = prediction[:, i]
        label = labels[:, i]
        mAP += average_precision_score(label, pred)

    return mAP / num_class * 100


def get_psnr(mse_loss):
    return  10 * log10(1 / mse_loss.item())
