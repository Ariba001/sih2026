"""Asymmetric Huber loss for XGBoost: 10× penalty on under-predicting drift.

Motivation: Under-predicting drift (runaway failure) is catastrophic for space payloads.
Over-predicting (conservative) is acceptable. Standard MSE treats both equally.

This module provides gradient and Hessian for XGBoost custom objectives.
"""
import numpy as np


def asymmetric_huber_loss(y_true, y_pred, alpha=10.0, delta=1.0):
    """Asymmetric Huber loss: penalize under-prediction 10× more.

    Args:
        y_true: Actual log drift (log V_168 / V_0)
        y_pred: Predicted log drift
        alpha: Asymmetry factor (under-prediction penalty multiplier)
        delta: Huber loss threshold

    Returns:
        Loss array (positive, element-wise)
    """
    residual = y_true - y_pred

    # Huber loss for stability
    huber_loss = np.where(
        np.abs(residual) <= delta,
        0.5 * residual**2,
        delta * (np.abs(residual) - 0.5 * delta)
    )

    # Asymmetric weighting
    weighted = np.where(
        residual > 0,  # Under-predicted (drift runaway) — high penalty
        alpha * huber_loss,
        1.0 * huber_loss  # Over-predicted (conservative) — low penalty
    )

    return weighted


def asymmetric_huber_grad(y_true, y_pred, alpha=10.0, delta=1.0):
    """Gradient and Hessian for XGBoost custom objective.

    Args:
        y_true: Actual log drift
        y_pred: Predicted log drift
        alpha: Asymmetry factor
        delta: Huber threshold

    Returns:
        (grad, hess): Gradient and Hessian arrays for XGBoost
    """
    residual = y_true - y_pred

    # Gradient of Huber loss
    grad_huber = np.where(
        np.abs(residual) <= delta,
        residual,
        delta * np.sign(residual)
    )

    # Apply asymmetric weighting and negate for XGBoost (which maximizes)
    grad = np.where(
        residual > 0,
        -alpha * grad_huber,  # Negative gradient for maximization
        -1.0 * grad_huber
    )

    # Hessian: second derivative of asymmetric Huber
    hess_huber = np.where(
        np.abs(residual) <= delta,
        np.ones_like(residual),
        np.zeros_like(residual)
    )

    hess = hess_huber * np.where(residual > 0, alpha, 1.0)

    return grad, hess
