import numpy as np


def NCE(source_label: np.ndarray, target_label: np.ndarray):
    """

    :param source_label: shape [N], elements in [0, C_s), often got from taken argmax from pre-trained predictions
    :param target_label: shape [N], elements in [0, C_t)
    :return:
    """
    C_t = int(np.max(target_label) + 1)  # the number of target classes
    C_s = int(np.max(source_label) + 1)  # the number of source classes
    N = len(source_label)
    joint = np.zeros(C_t, C_s, dtype=float)  # placeholder for the joint distribution
    for s, t in zip(source_label, target_label):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)
    p_z = joint.sum(axis=0).reshape(-1, 1)
    conditional_entropy = np.sum(entropy_y_given_z * p_z)
    return - conditional_entropy
