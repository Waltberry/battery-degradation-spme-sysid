from __future__ import annotations


def effective_lbfgs_epochs(use_lbfgs: bool, lbfgs_epochs: int) -> int:
    return int(lbfgs_epochs) if use_lbfgs else 0