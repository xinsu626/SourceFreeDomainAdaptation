import random
import collections
import os
import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from anafora import evaluate


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    """https://github.com/PyTorchLightning/pytorch-lightning/blob/
        116027088262ab6cb2f094d61e63d991dd391d05/pytorch_lightning/utilities/seed.py"""
    seed = random.randint(min_seed_value, max_seed_value)
    warnings.warn('"No correct seed found, seed set to {}'.format(seed))

    return seed


def seed_everything(seed):
    """https://github.com/PyTorchLightning/pytorch-lightning/blob/
    116027088262ab6cb2f094d61e63d991dd391d05/pytorch_lightning/utilities/seed.py"""

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", _select_seed_randomly(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        warnings.warn('{} is not in bounds, numpy accepts from {} to {}'.format(seed,
                                                                                min_seed_value,
                                                                                max_seed_value))
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def negation_performance(labels_true, labels_pred):
    f1 = f1_score(y_true=labels_true, y_pred=labels_pred)
    precision = precision_score(y_true=labels_true, y_pred=labels_pred)
    recall = recall_score(y_true=labels_true, y_pred=labels_pred)

    return f1, precision, recall


def time_performance(gold_standard_dir,
                     predictions_dir):
    """
    Adapted from
    https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation/blob/master/scoring_program/evaluation.py
    """
    scores_type = evaluate.Scores
    exclude = {("Event", "*", "<span>")}
    file_named_scores = evaluate.score_dirs(
        reference_dir=gold_standard_dir,
        predicted_dir=predictions_dir,
        exclude=exclude)  # pairwise=True

    all_named_scores = collections.defaultdict(lambda: scores_type())
    for _, named_scores in file_named_scores:
        for name, scores in named_scores.items():
            all_named_scores[name].update(scores)

    return all_named_scores["*"].f1(), all_named_scores["*"].precision(), all_named_scores["*"].recall()
