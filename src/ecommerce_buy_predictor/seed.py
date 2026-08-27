"""Single entry point for seeding every random generator the pipeline touches.

Setting ``random_state`` on each estimator (see
:func:`~ecommerce_buy_predictor.models.train.build_model`) only pins the
estimators. It does *not* pin the process-wide generators that other steps may
draw from: Python's :mod:`random`, NumPy's global RNG and the hash seed. This
module seeds all of them from the same value so a full retrain
(``preprocess`` → ``train``) is deterministic end to end, not just per model.
"""
import os
import random

import numpy as np

from ecommerce_buy_predictor.config import settings


def set_global_seeds(seed: int = settings.random_seed) -> int:
    """Seed every global random generator the pipeline can draw from.

    Call this once at the start of each retraining stage, before any data is
    read or any model is built. It complements — does not replace — the
    ``random_state`` forwarded to the estimators and the train/test split.

    Args:
        seed: The single seed value. Defaults to
            :attr:`Settings.random_seed <ecommerce_buy_predictor.config.Settings>`,
            which mirrors ``random_seed`` in ``params.yaml`` (the DVC source of
            truth). The retraining stages pass ``params["random_seed"]``
            explicitly so a change in ``params.yaml`` alone re-seeds everything.

    Returns:
        The seed that was applied, so callers can log it.
    """
    # Affects child processes spawned after this point (the current process
    # inherited its hash seed at interpreter start).
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
