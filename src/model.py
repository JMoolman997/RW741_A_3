from __future__ import annotations

import copy
import numpy as np


class FeedForwardClassifier:
    """One-hidden-layer (or linear softmax) network with expandable parameters."""
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 0,
                 seed: int = 1, activation: str = "relu"):
        self.input_dim, self.output_dim, self.hidden_units = input_dim, output_dim, hidden_units
        if activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.activation = activation
        self.rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        # The direct softmax path remains when capacity grows. This residual
        # parameterisation makes the zero-hidden -> one-hidden transition exact.
        self.params["W"] = self._weight(input_dim, output_dim)
        self.params["b"] = np.zeros(output_dim)
        if hidden_units:
            self.params["W1"] = self._weight(input_dim, hidden_units)
            self.params["b1"] = np.zeros(hidden_units)
            self.params["W2"] = np.zeros((hidden_units, output_dim))

    def _weight(self, fan_in, fan_out):
        return self.rng.normal(0, np.sqrt(2 / max(1, fan_in)), (fan_in, fan_out))

    def copy(self):
        return copy.deepcopy(self)

    def forward(self, X, cache=False):
        if self.hidden_units:
            z = X @ self.params["W1"] + self.params["b1"]
            h = np.maximum(z, 0) if self.activation == "relu" else np.tanh(z)
            logits = X @ self.params["W"] + self.params["b"] + h @ self.params["W2"]
            state = (X, z, h)
        else:
            logits = X @ self.params["W"] + self.params["b"]
            state = (X,)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
        return (probs, state) if cache else probs

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def loss_and_gradients(self, X, y, l2_coefficient=0.0):
        probs, state = self.forward(X, cache=True)
        cross_entropy = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        weight_keys = [key for key in self.params if key.startswith("W")]
        penalty = .5 * l2_coefficient * sum(np.sum(self.params[key] ** 2) for key in weight_keys)
        loss = cross_entropy + penalty
        dlogits = probs.copy(); dlogits[np.arange(len(y)), y] -= 1; dlogits /= len(y)
        if self.hidden_units:
            _, z, h = state
            grads = {"W": X.T @ dlogits + l2_coefficient * self.params["W"],
                     "b": dlogits.sum(0),
                     "W2": h.T @ dlogits + l2_coefficient * self.params["W2"]}
            dh = dlogits @ self.params["W2"].T
            dz = dh * ((z > 0) if self.activation == "relu" else (1 - np.tanh(z) ** 2))
            grads.update(W1=X.T @ dz + l2_coefficient * self.params["W1"], b1=dz.sum(0))
        else:
            grads = {"W": X.T @ dlogits + l2_coefficient * self.params["W"],
                     "b": dlogits.sum(0)}
        return float(loss), grads

    def expand_output(self, new_output_dim: int):
        if new_output_dim <= self.output_dim:
            return
        extra = new_output_dim - self.output_dim
        if self.hidden_units:
            self.params["W2"] = np.hstack([self.params["W2"], np.zeros((self.hidden_units, extra))])
        self.params["W"] = np.hstack([self.params["W"], self._weight(self.input_dim, extra)])
        self.params["b"] = np.r_[self.params["b"], np.zeros(extra)]
        self.output_dim = new_output_dim

    def add_hidden_neuron(self):
        if self.hidden_units == 0:
            self.params["W1"] = self._weight(self.input_dim, 1)
            self.params["b1"] = np.zeros(1)
            self.params["W2"] = np.zeros((1, self.output_dim))
        else:
            self.params["W1"] = np.hstack([self.params["W1"], self._weight(self.input_dim, 1)])
            self.params["b1"] = np.r_[self.params["b1"], 0.0]
            self.params["W2"] = np.vstack([self.params["W2"], np.zeros((1, self.output_dim))])
        self.hidden_units += 1
