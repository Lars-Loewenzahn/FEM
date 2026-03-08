"""
mi_toolbox.py
=============

A single-file "machine intelligence" toolbox that collects the *re-usable* parts of your
Tutorium code and your turtle-based evolutionary toy projects into one importable module.

Design goals (pragmatic, not dogmatic):
- **Reusability**: small building blocks that you can import in many notebooks/scripts.
- **Clarity**: every public function/method has an explanatory docstring.
- **Minimal dependencies**: NumPy for vector math; `turtle` is optional and only used in the
  maze visualisation helpers (you can still use Track/Ray for pure computation).

What’s inside
-------------
Core numerics
- stable sigmoid + a few small activation helpers
- bias-column helper for linear models

Datasets (synthetic)
- 1D regression with outliers
- 2D binary classification "two blobs"

Loss functions (callables)
- MSE, MAE, Huber, LogCosh
- BCEWithLogits (stable logistic loss on logits)
- Hinge (SVM-style hinge on logits)

Models / training
- `ConnectionistNeuron`: single neuron with step (perceptron) or sigmoid (logistic) training
- `train_linear_model_gd`: generic gradient-descent training for linear models with arbitrary
  loss objects from this toolbox

Evolutionary toy net
- `EvoMLP`: small MLP designed for *mutation-based* optimisation (no backprop),
  inspired by your "mutate + clone + select" loops.

Turtle maze utilities
- `Track`: grid-maze generator + optional turtle drawing
- `Ray`: fast grid ray-caster (lidar-like distance to walls)

Usage example
-------------
In a notebook:

    from mi_toolbox import ConnectionistNeuron, make_regression_1d, MSE, train_linear_model_gd

    X, y = make_regression_1d()
    w, hist = train_linear_model_gd(X, y, loss_fn=MSE(), lr=0.2, epochs=2000)

Or evolutionary policies:

    from mi_toolbox import EvoMLP, evolve_population

    pop = [EvoMLP([6, 16, 4], seed=i) for i in range(50)]
    pop = evolve_population(pop, fitness_fn=my_fitness, keep=10, mutate_scale=0.2)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import copy
import math
import random

import numpy as np

try:
    import turtle  # optional
except Exception:  # pragma: no cover
    turtle = None  # type: ignore


ArrayLike = Union[np.ndarray, Sequence[float]]


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def stable_sigmoid(z: ArrayLike) -> np.ndarray:
    """
    Numerically-stable logistic sigmoid.

    What it does
    ------------
    Computes the logistic function

        σ(z) = 1 / (1 + exp(-z))

    in a way that does *not* overflow for large |z|.

    Why you care
    ------------
    In "naive" code, `np.exp(800)` will overflow to `inf`, and then downstream you get
    `nan` losses or gradients. The stable implementation uses a simple algebraic trick:

    - For z >= 0:
        σ(z) = 1 / (1 + exp(-z))          (safe, because exp(-z) is small)
    - For z < 0:
        σ(z) = exp(z) / (1 + exp(z))     (safe, because exp(z) is small)

    Parameters
    ----------
    z:
        Scalar, list, or ndarray of any shape.

    Returns
    -------
    np.ndarray:
        Same shape as `z`, dtype float, with values in (0, 1).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softplus(z: ArrayLike) -> np.ndarray:
    """
    Numerically-stable softplus.

    What it does
    ------------
    Computes

        softplus(z) = log(1 + exp(z))

    in a stable way via `np.logaddexp(0, z)`.

    Why you care
    ------------
    Softplus is the smooth cousin of ReLU and it shows up naturally when you derive
    stable logistic losses (BCE on logits). Using `log(1 + exp(z))` directly can overflow
    for large z; `logaddexp` avoids that.

    Parameters
    ----------
    z:
        Scalar, list, or ndarray.

    Returns
    -------
    np.ndarray:
        Softplus evaluated elementwise.
    """
    z = np.asarray(z, dtype=float)
    return np.logaddexp(0.0, z)


def add_bias_column(X: ArrayLike) -> np.ndarray:
    """
    Adds a bias (intercept) column of ones to a feature matrix.

    What it does
    ------------
    If your linear model is

        y = X @ w

    then you often want an intercept term b:

        y = X_raw @ w_raw + b

    A clean trick is to augment the input matrix:

        X = [1, x1, x2, ...]
        w = [b, w1, w2, ...]

    so the same formula `X @ w` includes the bias.

    Parameters
    ----------
    X:
        Feature matrix of shape (n_samples, n_features) OR a 1D feature vector
        of shape (n_features,).

    Returns
    -------
    np.ndarray:
        If X was 2D -> shape (n_samples, n_features + 1).
        If X was 1D -> shape (n_features + 1,).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return np.concatenate([np.ones(1), X])
    if X.ndim == 2:
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])
    raise ValueError("X must be 1D or 2D.")


# -----------------------------------------------------------------------------
# Synthetic datasets
# -----------------------------------------------------------------------------

def make_regression_1d(
    n: int = 120,
    noise: float = 0.3,
    outliers: int = 10,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic 1D regression dataset with optional outliers.

    What it does
    ------------
    Generates data from a simple linear ground-truth model:

        y = 1 + 2x + ε

    where ε is Gaussian noise. Then it deliberately injects a number of "outliers"
    to stress-test robust losses (MAE, Huber, LogCosh) against MSE.

    Why you care
    ------------
    This is a tiny controlled sandbox where you can *see* the behaviour of different
    loss functions and optimisation methods without real-world dataset complexity.

    Output format
    -------------
    Returns a design matrix `X` **with a bias column** so that you can directly do:

        y_pred = X @ w

    with w = [b, w1].

    Parameters
    ----------
    n:
        Number of samples.
    noise:
        Standard deviation of the Gaussian noise.
    outliers:
        How many samples should be turned into outliers.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (X, y):
        X has shape (n, 2) = [bias, x]
        y has shape (n,)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=n)
    y = 1.0 + 2.0 * x + rng.normal(0.0, noise, size=n)

    if outliers > 0:
        idx = rng.choice(n, size=min(outliers, n), replace=False)
        # Outliers: add big noise bursts (both directions).
        y[idx] += rng.normal(0.0, 6.0 * noise + 3.0, size=len(idx))

    X = add_bias_column(x.reshape(-1, 1))
    return X, y


def make_binary_classification_2d(
    n: int = 300,
    seed: int = 0,
    centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.2, -1.0), (1.2, 1.0)),
    scale: float = 0.8,
    add_bias: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 2D binary classification dataset ("two Gaussian blobs").

    What it does
    ------------
    Samples two clusters in 2D:
    - Class 0 centered around centers[0]
    - Class 1 centered around centers[1]
    Each cluster is drawn from a normal distribution with standard deviation `scale`.

    Then it stacks, shuffles, and returns features X and labels y.

    Why you care
    ------------
    This is the "Hello World" for decision boundaries. It's small enough to plot and
    inspect by eye, but non-trivial enough to test perceptron/logistic classifiers.

    Parameters
    ----------
    n:
        Total number of points (split 50/50 between the classes).
    seed:
        RNG seed.
    centers:
        Cluster centers for class 0 and class 1.
    scale:
        Standard deviation of each blob.
    add_bias:
        If True, adds a bias column to X, so X has shape (n, 3) = [1, x1, x2].

    Returns
    -------
    (X, y):
        X shape:
            - (n, 3) if add_bias else (n, 2)
        y shape:
            - (n,), values in {0,1}
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    X0 = rng.normal(loc=centers[0], scale=scale, size=(n0, 2))
    X1 = rng.normal(loc=centers[1], scale=scale, size=(n1, 2))

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]

    if add_bias:
        X = add_bias_column(X)

    return X, y


# -----------------------------------------------------------------------------
# Loss functions (callables)
# -----------------------------------------------------------------------------

class MSE:
    """
    Mean Squared Error (MSE) loss + gradient w.r.t. predictions.

    What it does
    ------------
    For residual r = y_pred - y_true, MSE is:

        L = 0.5 * mean(r^2)

    The factor 0.5 is a convenience: the derivative becomes clean.

    Gradient (needed for gradient descent)
    --------------------------------------
    The gradient returned here is **dL/dy_pred**, because that's the most reusable
    interface for linear models and small nets:

        dL/dy_pred = r / n

    where n is number of samples.

    When to use
    -----------
    - Great for "nice" Gaussian noise.
    - Sensitive to outliers (because r^2 explodes).
    """
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        r = y_pred - y_true
        loss = 0.5 * float(np.mean(r ** 2))
        dL_dy = r / len(y_true)
        return loss, dL_dy


class MAE:
    """
    Mean Absolute Error (MAE) loss + (sub)gradient w.r.t. predictions.

    What it does
    ------------
    For residual r = y_pred - y_true:

        L = mean(|r|)

    Gradient (subgradient)
    ----------------------
    Absolute value is not differentiable at r=0. In optimisation we use a
    **subgradient**:

        d|r|/dr = sign(r) , and we define sign(0)=0

    Returned gradient is dL/dy_pred = sign(r) / n.

    When to use
    -----------
    - Robust to outliers.
    - Can be slower to optimise than MSE because the gradient is piecewise constant.
    """
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        r = y_pred - y_true
        loss = float(np.mean(np.abs(r)))
        d = np.sign(r)
        dL_dy = d / len(y_true)
        return loss, dL_dy


class Huber:
    """
    Huber loss: a robust compromise between MSE and MAE.

    What it does
    ------------
    For residual r and threshold δ:

        Lδ(r) = 0.5 r^2                    if |r| <= δ
                δ (|r| - 0.5 δ)            if |r| >  δ

    - Near zero: quadratic (smooth, like MSE).
    - Far from zero: linear (robust, like MAE).

    Gradient
    --------
        dL/dr = r                          if |r| <= δ
                δ * sign(r)                if |r| >  δ

    Returned gradient is dL/dy_pred = dL/dr / n.

    When to use
    -----------
    - You expect mostly Gaussian noise *plus* some outliers.
    """
    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        r = y_pred - y_true
        a = np.abs(r)
        quad = a <= self.delta

        loss = float(np.mean(np.where(quad, 0.5 * r ** 2, self.delta * (a - 0.5 * self.delta))))
        d = np.where(quad, r, self.delta * np.sign(r))
        dL_dy = d / len(y_true)
        return loss, dL_dy


class LogCosh:
    """
    Log-cosh loss: smooth robust alternative to Huber.

    What it does
    ------------
        L = mean( log(cosh(r)) )

    Properties
    ----------
    - For small r: log(cosh(r)) ≈ 0.5 r^2      (MSE-like)
    - For large r: log(cosh(r)) ≈ |r| - log(2) (MAE-like)

    Gradient
    --------
        d/dr log(cosh(r)) = tanh(r)

    Returned gradient is dL/dy_pred = tanh(r) / n.

    When to use
    -----------
    - You want robustness but also fully smooth derivatives (no kink).
    """
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        r = y_pred - y_true
        loss = float(np.mean(np.log(np.cosh(r))))
        d = np.tanh(r)
        dL_dy = d / len(y_true)
        return loss, dL_dy


class BCEWithLogits:
    """
    Binary Cross-Entropy computed on logits (stable), plus gradient.

    Setup
    -----
    - Labels: y in {0, 1}
    - Model outputs: h = logits (real numbers, not squashed)

    Loss
    ----
    A stable form of BCE is:

        L = mean( softplus(h) - y*h )

    where softplus(h) = log(1 + exp(h)).

    Why logits?
    -----------
    If you first apply sigmoid and then take logs, you can hit log(0) problems.
    This logits-form avoids that and is standard in modern ML libraries.

    Gradient
    --------
        dL/dh = sigmoid(h) - y

    Returned gradient is divided by n (mean-loss convention).

    Usage
    -----
    This is great for logistic regression with a linear model:

        h = X @ w
        loss, dL_dh = BCEWithLogits()(h, y)
        grad_w = X.T @ dL_dh
    """
    def __call__(self, h: np.ndarray, y01: np.ndarray) -> Tuple[float, np.ndarray]:
        h = np.asarray(h, dtype=float).reshape(-1)
        y01 = np.asarray(y01, dtype=float).reshape(-1)

        sp = softplus(h)
        loss = float(np.mean(sp - y01 * h))
        dL_dh = (stable_sigmoid(h) - y01) / len(y01)
        return loss, dL_dh


class Hinge:
    """
    Hinge loss (SVM-style) on logits, plus gradient.

    Setup
    -----
    - Labels: y in {-1, +1}
    - Model outputs: h = logits

    Loss
    ----
        L = mean( max(0, 1 - y*h) )

    Gradient
    --------
    For each sample:
    - If 1 - y*h > 0 (margin violated): dL/dh = -y
    - Else: dL/dh = 0

    Returned gradient is divided by n (mean-loss convention).
    """
    def __call__(self, h: np.ndarray, ypm1: np.ndarray) -> Tuple[float, np.ndarray]:
        h = np.asarray(h, dtype=float).reshape(-1)
        ypm1 = np.asarray(ypm1, dtype=float).reshape(-1)

        margin = 1.0 - ypm1 * h
        active = margin > 0
        loss = float(np.mean(np.maximum(0.0, margin)))
        dL_dh = np.where(active, -ypm1, 0.0) / len(ypm1)
        return loss, dL_dh


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def train_linear_model_gd(
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
    lr: float = 0.1,
    epochs: int = 2000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic gradient-descent trainer for linear models y_pred = X @ w.

    What it does
    ------------
    Implements the "minimal viable optimiser" you used in the tutorials:

        1) initialise weights w (small random values)
        2) for each epoch:
            - predict: y_pred = X @ w
            - compute loss + gradient wrt predictions: (loss, dL/dy_pred) = loss_fn(...)
            - chain rule to get gradient wrt weights:
                  y_pred = Xw  =>  dy_pred/dw = X
                  grad_w = X^T @ (dL/dy_pred)
            - update: w <- w - lr * grad_w

    Why this interface is nice
    --------------------------
    Any loss object that returns (loss, dL/dy_pred) plugs in. That means you can swap
    MSE/MAE/Huber/LogCosh without rewriting training code.

    Parameters
    ----------
    X:
        Design matrix, shape (n_samples, n_features). If you want a bias term, use
        `add_bias_column` first.
    y:
        Targets, shape (n_samples,).
    loss_fn:
        Callable returning (loss, dL/dy_pred).
    lr:
        Learning rate.
    epochs:
        Number of GD steps.
    seed:
        RNG seed for the initial weights.

    Returns
    -------
    (w, history):
        w shape (n_features,)
        history shape (epochs,) containing the loss curve
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    w = rng.normal(0.0, 0.01, size=X.shape[1])

    history = np.empty(epochs, dtype=float)
    for t in range(epochs):
        y_pred = X @ w
        loss, dL_dy = loss_fn(y_pred, y)
        grad_w = X.T @ dL_dy
        w -= lr * grad_w
        history[t] = loss

    return w, history


# -----------------------------------------------------------------------------
# ConnectionistNeuron (single neuron)
# -----------------------------------------------------------------------------

class ConnectionistNeuron:
    """
    A single "connectionist neuron" with configurable activation and simple training.

    Concept
    -------
    For input x in R^N, weights w in R^N, bias b in R:

        h = w^T x + b                (net input / pre-activation)
        y = f(h)                     (activation)

    Supported activations
    ---------------------
    - "step"    : Heaviside (perceptron) -> output 0/1
    - "sigmoid" : logistic               -> output in (0,1)
    - "tanh"    : tanh                   -> output in (-1,1)
    - "linear"  : identity               -> output is h

    Training support
    ----------------
    - For "step"   : classic online perceptron update.
    - For "sigmoid": batch gradient descent on BCE (logistic regression style).

    Notes
    -----
    This class is intentionally "transparent": it's good for learning and for toy models,
    not meant to compete with PyTorch.
    """

    def __init__(self, n_features: int, activation: str = "step", lr: float = 0.1, seed: Optional[int] = None):
        """
        Create a neuron with random small initial weights.

        Parameters
        ----------
        n_features:
            Input dimension N.
        activation:
            One of {"step","sigmoid","tanh","linear"}.
        lr:
            Learning rate η.
        seed:
            RNG seed for reproducible initialisation.
        """
        self.n_features = int(n_features)
        self.activation_name = str(activation)
        self.lr = float(lr)

        rng = np.random.default_rng(seed)
        self.w = rng.normal(loc=0.0, scale=0.01, size=self.n_features)
        self.b = 0.0

        self._set_activation(self.activation_name)

    def _set_activation(self, activation: str) -> None:
        """
        Internal activation dispatcher.

        What it does
        ------------
        Maps the string name to a callable `self.activation(z)`.

        Why internal?
        ------------
        You want `forward()` to be activation-agnostic: it always does `f(w^T x + b)`.
        So we configure f once during init.

        Side effect
        -----------
        Sets `self._predict_proba` only when the output can be interpreted as a probability
        (sigmoid). For other activations it is set to None.
        """
        if activation == "step":
            self.activation = lambda z: (np.asarray(z) >= 0).astype(int)
            self._predict_proba = None
        elif activation == "sigmoid":
            self.activation = stable_sigmoid
            self._predict_proba = stable_sigmoid
        elif activation == "tanh":
            self.activation = np.tanh
            self._predict_proba = None
        elif activation == "linear":
            self.activation = lambda z: np.asarray(z)
            self._predict_proba = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def net_input(self, X: ArrayLike) -> np.ndarray:
        """
        Compute the net input (a.k.a. logits / pre-activation).

        Formula
        -------
            h = X @ w + b

        Shapes
        ------
        - If X is shape (n_features,) -> returns scalar array shape ().
        - If X is shape (n_samples, n_features) -> returns vector shape (n_samples,).

        Why this exists
        --------------
        Splitting the linear part from the activation makes debugging easier:
        you can inspect h before any nonlinearity.
        """
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def forward(self, X: ArrayLike) -> np.ndarray:
        """
        Forward pass: apply activation to the net input.

        In symbols:
            y = f( w^T x + b )

        This returns the "raw" activation output, which may be continuous (sigmoid/tanh/linear)
        or discrete (step).
        """
        return self.activation(self.net_input(X))

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        """
        Produce a class prediction (mostly for binary classification).

        Behaviour depends on activation:
        - step: output is already 0/1, so we return it as-is.
        - sigmoid: output is in (0,1), so we threshold it:
              y_hat = 1[p >= threshold]
        - other activations: returns forward(X) (you can decide what "prediction" means).

        Parameters
        ----------
        threshold:
            Decision threshold τ for sigmoid outputs.

        Returns
        -------
        np.ndarray:
            Predictions (shape matches forward()).
        """
        y = self.forward(X)
        if self.activation_name == "sigmoid":
            return (y >= threshold).astype(int)
        return y

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Return a probability-like output p for sigmoid neurons.

        Why this is separate
        --------------------
        Only sigmoid outputs have the right semantics: values in (0,1) that you can
        interpret as probabilities.

        Raises
        ------
        RuntimeError:
            If activation is not sigmoid.
        """
        if self._predict_proba is None:
            raise RuntimeError("predict_proba is only available for activation='sigmoid'.")
        return self._predict_proba(self.net_input(X))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, shuffle: bool = True, verbose: bool = False) -> dict:
        """
        Train the neuron parameters (w, b).

        What it does
        ------------
        Routes to the appropriate training routine depending on activation:

        - "step"   -> perceptron learning rule (online updates)
        - "sigmoid"-> batch GD on logistic cross-entropy

        Parameters
        ----------
        X:
            Feature matrix shape (n_samples, n_features).
        y:
            Labels shape (n_samples,), expected in {0,1}.
        epochs:
            Number of passes over the dataset.
        shuffle:
            For perceptron only: shuffle sample order each epoch (often improves convergence).
        verbose:
            Print progress each epoch.

        Returns
        -------
        dict:
            Training history (loss curve or errors-per-epoch).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n_samples, {self.n_features}).")

        if self.activation_name == "step":
            return self._fit_perceptron(X, y, epochs, shuffle, verbose)

        if self.activation_name == "sigmoid":
            return self._fit_logistic_gd(X, y, epochs, verbose)

        raise RuntimeError("fit is implemented only for activation='step' or 'sigmoid'.")

    def _fit_perceptron(self, X: np.ndarray, y: np.ndarray, epochs: int, shuffle: bool, verbose: bool) -> dict:
        """
        Online perceptron training (classic 1950s but still charming).

        Update rule (per sample)
        ------------------------
        Let y_hat = step(w·x + b) and error = y - y_hat, where error ∈ {-1,0,+1}.

            w <- w + lr * error * x
            b <- b + lr * error

        Intuition
        ---------
        - If a positive example was predicted as 0 (error=+1): push w towards x.
        - If a negative example was predicted as 1 (error=-1): push w away from x.

        Returns
        -------
        dict:
            {"errors_per_epoch": [..]}
        """
        rng = np.random.default_rng()
        history = {"errors_per_epoch": []}

        for ep in range(epochs):
            idx = np.arange(len(X))
            if shuffle:
                rng.shuffle(idx)

            errors = 0
            for i in idx:
                y_hat = int(self.forward(X[i]))
                error = int(y[i]) - y_hat
                if error != 0:
                    self.w += self.lr * error * X[i]
                    self.b += self.lr * error
                    errors += 1

            history["errors_per_epoch"].append(errors)
            if verbose:
                print(f"epoch {ep + 1:>3}/{epochs}: misclassifications={errors}")

        return history

    def _fit_logistic_gd(self, X: np.ndarray, y: np.ndarray, epochs: int, verbose: bool) -> dict:
        """
        Batch gradient descent for a sigmoid neuron (logistic regression as a single neuron).

        Loss (Binary Cross-Entropy)
        ---------------------------
        With logits z = Xw + b and probabilities p = sigmoid(z):

            L = -mean( y*log(p) + (1-y)*log(1-p) )

        Gradients (the elegant part)
        ----------------------------
        The derivative collapses nicely:

            dL/dz = (p - y) / n

        Then:
            grad_w = X^T (p - y) / n
            grad_b = mean(p - y)

        Returns
        -------
        dict:
            {"loss": [..]}
        """
        history = {"loss": []}
        n = len(X)
        eps = 1e-12

        for ep in range(epochs):
            z = self.net_input(X)
            p = stable_sigmoid(z)

            p_clip = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
            history["loss"].append(float(loss))

            diff = (p - y)
            grad_w = (X.T @ diff) / n
            grad_b = float(np.mean(diff))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if verbose:
                print(f"epoch {ep + 1:>3}/{epochs}: loss={loss:.6f}")

        return history


# -----------------------------------------------------------------------------
# Evolutionary MLP (mutation-based optimisation)
# -----------------------------------------------------------------------------

def _activation_from_name(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Small internal mapper from activation name -> callable.

    Supported names:
    - "tanh"
    - "sigmoid"
    - "relu"
    - "linear"

    This exists so `EvoMLP` can stay lightweight and still be configurable.
    """
    name = name.lower()
    if name == "tanh":
        return np.tanh
    if name == "sigmoid":
        return stable_sigmoid
    if name == "relu":
        return lambda z: np.maximum(0.0, z)
    if name == "linear":
        return lambda z: z
    raise ValueError(f"Unknown activation: {name}")


class EvoMLP:
    """
    A small multi-layer perceptron designed for *mutation*, not backpropagation.

    Origin story
    ------------
    Your turtle projects use a "brain" that:
    - runs a forward pass to decide actions
    - mutates weights randomly to explore behaviours
    - gets selected by a fitness score

    This class formalises that into a reusable MLP:
    - arbitrary layer sizes
    - optional bias terms
    - clean clone() and mutate() utilities

    Important philosophical note
    ----------------------------
    This is **not** gradient-based learning. It's evolutionary search / hill-climbing.
    That makes it:
    - dead simple to implement
    - often slower than backprop for smooth problems
    - sometimes more robust for weird, non-differentiable environments (like discrete games)

    Parameters
    ----------
    layer_sizes:
        Example: [n_in, n_hidden, n_out]
    hidden_activation:
        Activation used for all hidden layers.
    output_activation:
        Activation used for output layer. Often "linear" for action scores.
    seed:
        RNG seed for reproducible initialisation.
    weight_scale:
        Standard deviation for initial weights.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "tanh",
        output_activation: str = "linear",
        seed: Optional[int] = None,
        weight_scale: float = 0.3,
        use_bias: bool = True,
    ):
        self.layer_sizes = [int(s) for s in layer_sizes]
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output size.")

        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.hidden_act = _activation_from_name(hidden_activation)
        self.out_act = _activation_from_name(output_activation)

        self.use_bias = bool(use_bias)

        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.W.append(rng.normal(0.0, weight_scale, size=(fan_in, fan_out)))
            if self.use_bias:
                self.b.append(rng.normal(0.0, weight_scale, size=(fan_out,)))
            else:
                self.b.append(np.zeros((fan_out,), dtype=float))

    def forward(self, x: ArrayLike) -> np.ndarray:
        """
        Forward pass through the MLP.

        What it does
        ------------
        Applies repeated affine + activation transformations:

            a0 = x
            z1 = a0 @ W0 + b0
            a1 = act(z1)
            ...
            zk = a_{k-1} @ W_{k-1} + b_{k-1}
            ak = out_act(zk)

        Supports both:
        - x shape (n_features,)          -> returns (n_out,)
        - x shape (n_samples, n_features)-> returns (n_samples, n_out)

        Returns
        -------
        np.ndarray:
            Network output (action scores, logits, etc.).
        """
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            is_last = (i == len(self.W) - 1)
            a = self.out_act(z) if is_last else self.hidden_act(z)

        return a.reshape(-1) if squeeze else a

    def clone(self) -> "EvoMLP":
        """
        Deep-copy the network.

        Why this exists
        --------------
        In evolutionary algorithms, you often keep the top-k performers and produce
        new children by cloning + mutating them. Shallow copies would share arrays,
        causing "spooky action at a distance" between individuals.

        Returns
        -------
        EvoMLP:
            A fully independent copy.
        """
        return copy.deepcopy(self)

    def mutate(self, scale: float = 0.1, p: float = 1.0, rng: Optional[np.random.Generator] = None) -> None:
        """
        Randomly perturb weights and biases in-place.

        What it does
        ------------
        For each parameter entry θ:
        - with probability p, add uniform noise in [-scale, +scale]

    (Yes, this is a blunt instrument. It's also the point.)

        Parameters
        ----------
        scale:
            Magnitude of the mutation (step size).
        p:
            Mutation probability per parameter entry.
            - p=1.0 means mutate *every* weight/bias.
            - smaller p means sparse mutations.
        rng:
            Optional NumPy RNG. If None, a default generator is used.
        """
        if rng is None:
            rng = np.random.default_rng()

        for i in range(len(self.W)):
            mask = rng.random(self.W[i].shape) < p
            self.W[i] = self.W[i] + mask * rng.uniform(-scale, scale, size=self.W[i].shape)

            if self.use_bias:
                mask_b = rng.random(self.b[i].shape) < p
                self.b[i] = self.b[i] + mask_b * rng.uniform(-scale, scale, size=self.b[i].shape)

    def act_argmax(self, x: ArrayLike) -> int:
        """
        Convenience: interpret the network output as action scores and pick argmax.

        Typical use
        -----------
        If your output is e.g. 4 numbers for actions [right, left, up, down],
        then `act_argmax(x)` returns the chosen action index.

        Returns
        -------
        int:
            Index of the maximum output component.
        """
        y = self.forward(x)
        return int(np.argmax(y))


def evolve_population(
    population: List[EvoMLP],
    fitness_fn: Callable[[EvoMLP], float],
    keep: int = 10,
    mutate_scale: float = 0.2,
    mutate_p: float = 1.0,
    elites: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> List[EvoMLP]:
    """
    One-step evolutionary update: select -> clone -> mutate.

    What it does
    ------------
    Given a population of networks:
    1) Evaluate fitness for each individual.
    2) Keep the top `keep` as parents.
    3) Refill the population size by cloning parents and mutating children.

    Options:
    - `elites`: number of top individuals that are carried over unchanged (no mutation),
      which stabilises progress when mutation is aggressive.

    This mirrors the "next_generation" pattern from your turtle experiments, but makes
    it reusable and decoupled from the environment.

    Parameters
    ----------
    population:
        List of EvoMLP instances.
    fitness_fn:
        Function mapping a network -> scalar fitness. Higher is better.
    keep:
        Number of parents to keep.
    mutate_scale:
        Mutation magnitude.
    mutate_p:
        Mutation probability per parameter entry.
    elites:
        Number of best individuals copied verbatim into the next generation.
    rng:
        Optional NumPy RNG.

    Returns
    -------
    List[EvoMLP]:
        New population (same size as input).
    """
    if rng is None:
        rng = np.random.default_rng()

    if keep <= 0 or keep > len(population):
        raise ValueError("keep must satisfy 1 <= keep <= len(population).")

    scored = [(fitness_fn(ind), ind) for ind in population]
    scored.sort(key=lambda t: t[0], reverse=True)

    parents = [ind for _, ind in scored[:keep]]
    next_pop: List[EvoMLP] = []

    # carry over elites unchanged
    for i in range(min(elites, len(parents))):
        next_pop.append(parents[i].clone())

    # fill remaining slots
    while len(next_pop) < len(population):
        parent = parents[rng.integers(0, len(parents))]
        child = parent.clone()
        child.mutate(scale=mutate_scale, p=mutate_p, rng=rng)
        next_pop.append(child)

    return next_pop


# -----------------------------------------------------------------------------
# Turtle maze utilities (Track + Ray)
# -----------------------------------------------------------------------------

class Track:
    """
    Grid maze generator + optional turtle drawing.

    What it is
    ----------
    A rectangular grid of cells. Each cell has 4 walls: N/S/E/W (booleans).
    The maze is generated via randomized depth-first search (DFS), producing a
    *perfect maze* (exactly one unique path between any two cells).

    Why it matters
    --------------
    This gives you a nice toy world for:
    - navigation
    - reinforcement learning experiments
    - evolutionary policies with lidar-like rays (see `Ray`)

    Notes on coordinates
    --------------------
    The maze lives in a continuous 2D plane (turtle coordinates) but is backed by a grid.
    - `cell` is the pixel size of one cell.
    - The maze is centered around (0,0).
    """
    def __init__(
        self,
        cols: int = 21,
        rows: int = 15,
        cell: int = 24,
        wall_thickness: int = 3,
        margin: int = 20,
        seed: Optional[int] = None,
    ):
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell = int(cell)
        self.wall_thickness = int(wall_thickness)
        self.margin = int(margin)
        self.rng = random.Random(seed)

        # Each cell: dict of wall booleans
        self.grid = [[{"N": True, "S": True, "E": True, "W": True} for _ in range(self.cols)]
                     for _ in range(self.rows)]
        self.generated = False

    def generate(self) -> None:
        """
        Generate a random perfect maze using DFS backtracking.

        Algorithm sketch
        ---------------
        - Pick a random start cell.
        - Mark it visited.
        - While stack not empty:
            - Look for unvisited neighbours.
            - If exists: choose one randomly, remove walls between cells, push neighbour.
            - Else: pop (backtrack).

        Side effects
        ------------
        - Updates `self.grid` in-place by removing walls.
        - Ensures an entrance on the left of (0,0) and an exit on the right of the
          bottom-right cell, matching your original projects.
        """
        visited = [[False] * self.cols for _ in range(self.rows)]
        stack: List[Tuple[int, int]] = []

        r = self.rng.randrange(self.rows)
        c = self.rng.randrange(self.cols)
        visited[r][c] = True
        stack.append((r, c))

        while stack:
            cr, cc = stack[-1]
            neighbors = []
            if cr > 0 and not visited[cr - 1][cc]:
                neighbors.append((cr - 1, cc, "N"))
            if cr < self.rows - 1 and not visited[cr + 1][cc]:
                neighbors.append((cr + 1, cc, "S"))
            if cc > 0 and not visited[cr][cc - 1]:
                neighbors.append((cr, cc - 1, "W"))
            if cc < self.cols - 1 and not visited[cr][cc + 1]:
                neighbors.append((cr, cc + 1, "E"))

            if neighbors:
                nr, nc, direction = self.rng.choice(neighbors)
                self._remove_wall(cr, cc, nr, nc, direction)
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        # entrance and exit
        self.grid[0][0]["W"] = False
        self.grid[self.rows - 1][self.cols - 1]["E"] = False
        self.generated = True

    def _remove_wall(self, r1: int, c1: int, r2: int, c2: int, direction: str) -> None:
        """
        Remove walls between two adjacent cells.

        This is an internal helper used by `generate()` and assumes the cells are neighbours.
        """
        if direction == "N":
            self.grid[r1][c1]["N"] = False
            self.grid[r2][c2]["S"] = False
        elif direction == "S":
            self.grid[r1][c1]["S"] = False
            self.grid[r2][c2]["N"] = False
        elif direction == "W":
            self.grid[r1][c1]["W"] = False
            self.grid[r2][c2]["E"] = False
        elif direction == "E":
            self.grid[r1][c1]["E"] = False
            self.grid[r2][c2]["W"] = False
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _cell_top_left(self, r: int, c: int) -> Tuple[float, float]:
        """
        Convert grid indices (r,c) -> top-left turtle coordinate of that cell.

        The maze is centered around (0,0), so the left edge is at -width/2 and the
        top edge at +height/2.
        """
        width = self.cols * self.cell
        height = self.rows * self.cell
        left = -width / 2
        top = height / 2
        x0 = left + c * self.cell
        y0 = top - r * self.cell
        return x0, y0

    def _dims(self) -> Tuple[float, float, int, int, int]:
        """
        Return core geometry parameters useful for coordinate transforms.

        Returns
        -------
        (left, top, cell, rows, cols)
        """
        width = self.cols * self.cell
        height = self.rows * self.cell
        left = -width / 2
        top = height / 2
        return left, top, self.cell, self.rows, self.cols

    def draw(self):
        """
        Draw the maze with turtle graphics.

        This is optional sugar for visual debugging. If turtle is not available, it raises
        a RuntimeError.

        Returns
        -------
        (screen, pen):
            The turtle Screen and the Turtle pen used for drawing. This lets your external
            simulation keep a single global screen and call `screen.update()` once per frame.
        """
        if turtle is None:
            raise RuntimeError("turtle is not available in this Python environment.")

        if not self.generated:
            self.generate()

        screen = turtle.Screen()
        width_px = int(self.cols * self.cell + 2 * self.margin)
        height_px = int(self.rows * self.cell + 2 * self.margin)
        screen.setup(width=width_px, height=height_px)
        screen.title("Turtle Track (Maze)")
        screen.tracer(False)

        pen = turtle.Turtle(visible=False)
        pen.speed(0)
        pen.color("black")
        pen.width(self.wall_thickness)

        def draw_line(x1, y1, x2, y2):
            pen.up()
            pen.goto(x1, y1)
            pen.down()
            pen.goto(x2, y2)

        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = self._cell_top_left(r, c)
                if self.grid[r][c]["N"]:
                    draw_line(x0, y0, x0 + self.cell, y0)
                if self.grid[r][c]["W"]:
                    draw_line(x0, y0, x0, y0 - self.cell)

        # bottom boundary
        r = self.rows - 1
        for c in range(self.cols):
            if self.grid[r][c]["S"]:
                x0, y0 = self._cell_top_left(r, c)
                draw_line(x0, y0 - self.cell, x0 + self.cell, y0 - self.cell)

        # right boundary
        c = self.cols - 1
        for r in range(self.rows):
            if self.grid[r][c]["E"]:
                x0, y0 = self._cell_top_left(r, c)
                draw_line(x0 + self.cell, y0, x0 + self.cell, y0 - self.cell)

        screen.update()
        return screen, pen


class Ray:
    """
    Grid ray-caster for `Track` (lidar-like distance to walls).

    What it does
    ------------
    Given a continuous position (x, y) and a heading angle, this walks cell-by-cell
    through the grid using a 2D DDA-style stepping method:
    - compute the next vertical grid line hit and next horizontal grid line hit
    - step to whichever is closer
    - check if that boundary is a wall
    - stop at the first wall hit (or outside the maze)

    Why you care
    ------------
    This is a low-cost way to get "sensors" for agents (cars, robots) without doing
    expensive geometric collision detection.

    Debug drawing
    -------------
    You can optionally draw the ray with turtle (for visual debugging), but the distance
    computation itself is purely numeric.
    """
    def __init__(
        self,
        track: Track,
        x: float,
        y: float,
        heading_deg: float,
        debug_draw: bool = False,
        color: str = "yellow",
        width: int = 1,
    ):
        self.track = track
        self.x, self.y = float(x), float(y)
        self.heading = float(heading_deg) % 360.0
        self.debug_draw = bool(debug_draw)

        self._pen = None
        if self.debug_draw:
            if turtle is None:
                raise RuntimeError("turtle is not available but debug_draw=True was requested.")
            self._pen = turtle.Turtle(visible=False)
            self._pen.hideturtle()
            self._pen.speed(0)
            self._pen.color(color)
            self._pen.width(width)
            self._pen.penup()

    def destroy(self) -> None:
        """
        Clean up debug-drawing turtle resources (optional).

        In most runs you don't need this, but when you create many Ray objects with
        debug_draw=True, clearing prevents visual clutter and resource buildup.
        """
        if self._pen is not None:
            try:
                self._pen.clear()
                self._pen.hideturtle()
            except Exception:
                pass

    def cast(self, max_iters: int = 10000) -> float:
        """
        Cast the ray and return the distance to the first wall hit.

        Parameters
        ----------
        max_iters:
            Safety cap to avoid infinite loops in degenerate numeric situations.

        Returns
        -------
        float:
            Distance from (x,y) to the collision point along the ray direction.
            If the start point is outside the maze, returns 0.0.
        """
        if not self.track.generated:
            # Ray casting needs a maze. Generate if missing.
            self.track.generate()

        left, top, cell, rows, cols = self.track._dims()
        dx = math.cos(math.radians(self.heading))
        dy = math.sin(math.radians(self.heading))

        x = self.x
        y = self.y

        r = int((top - y) // cell)
        c = int((x - left) // cell)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0.0

        length = 0.0
        end_x, end_y = x, y
        eps = 1e-12

        it = 0
        while it < max_iters:
            it += 1

            # param t to next vertical boundary
            if dx > 0:
                next_vx = left + (c + 1) * cell
                t_vert = (next_vx - x) / dx
            elif dx < 0:
                next_vx = left + c * cell
                t_vert = (next_vx - x) / dx
            else:
                t_vert = float("inf")

            # param t to next horizontal boundary
            if dy > 0:
                next_hy = top - r * cell
                t_horiz = (next_hy - y) / dy
            elif dy < 0:
                next_hy = top - (r + 1) * cell
                t_horiz = (next_hy - y) / dy
            else:
                t_horiz = float("inf")

            if t_vert <= eps and t_horiz <= eps:
                t_vert = t_horiz = 0.0

            if t_vert < t_horiz:
                step_t = t_vert
                x2 = x + dx * step_t
                y2 = y + dy * step_t

                wall = self.track.grid[r][c]["E"] if dx > 0 else self.track.grid[r][c]["W"]
                length += max(0.0, step_t)
                if wall:
                    end_x, end_y = x2, y2
                    break

                x, y = x2, y2
                c = c + 1 if dx > 0 else c - 1
                if c < 0 or c >= cols:
                    end_x, end_y = x, y
                    break

            elif t_horiz < t_vert:
                step_t = t_horiz
                x2 = x + dx * step_t
                y2 = y + dy * step_t

                wall = self.track.grid[r][c]["N"] if dy > 0 else self.track.grid[r][c]["S"]
                length += max(0.0, step_t)
                if wall:
                    end_x, end_y = x2, y2
                    break

                x, y = x2, y2
                r = r - 1 if dy > 0 else r + 1
                if r < 0 or r >= rows:
                    end_x, end_y = x, y
                    break

            else:
                # corner case: hits vertical and horizontal boundary at same time
                step_t = t_vert
                x2 = x + dx * step_t
                y2 = y + dy * step_t

                wall_v = self.track.grid[r][c]["E"] if dx > 0 else self.track.grid[r][c]["W"]
                wall_h = self.track.grid[r][c]["N"] if dy > 0 else self.track.grid[r][c]["S"]
                length += max(0.0, step_t)
                if wall_v or wall_h:
                    end_x, end_y = x2, y2
                    break

                x, y = x2, y2
                c = c + 1 if dx > 0 else c - 1
                r = r - 1 if dy > 0 else r + 1
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    end_x, end_y = x, y
                    break

        if self._pen is not None:
            try:
                self._pen.clear()
                self._pen.goto(self.x, self.y)
                self._pen.pendown()
                self._pen.goto(end_x, end_y)
                self._pen.penup()
            except Exception:
                pass

        return float(length)


__all__ = [
    # helpers
    "stable_sigmoid",
    "softplus",
    "add_bias_column",
    # datasets
    "make_regression_1d",
    "make_binary_classification_2d",
    # losses
    "MSE",
    "MAE",
    "Huber",
    "LogCosh",
    "BCEWithLogits",
    "Hinge",
    # training
    "train_linear_model_gd",
    # models
    "ConnectionistNeuron",
    "EvoMLP",
    "evolve_population",
    # turtle utilities
    "Track",
    "Ray",
]
