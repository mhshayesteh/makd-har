from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .reproducibility import set_global_seed


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training") from exc
    return tf


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = np.asarray(logits).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "per_class_f1": f1_score(
            y_true, predictions, average=None, zero_division=0
        ).astype(float).tolist(),
    }


def train_supervised(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    verbose: int = 0,
):
    tf = _tensorflow()
    set_global_seed(seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, {key: [float(v) for v in values] for key, values in history.history.items()}


def predict_logits(model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    return np.asarray(model.predict(inputs, batch_size=batch_size, verbose=0), dtype=np.float32)


def logits_to_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)


def aggregate_teacher_probabilities(
    teacher_probabilities: Sequence[np.ndarray], weights: np.ndarray | None = None
) -> np.ndarray:
    stacked = np.stack(teacher_probabilities, axis=0).astype(np.float64)
    if weights is None:
        weights = np.full(len(stacked), 1.0 / len(stacked), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(stacked),) or np.any(weights < 0):
        raise ValueError("weights have the wrong shape or contain negative values")
    weights = weights / weights.sum()
    result = np.tensordot(weights, stacked, axes=(0, 0))
    return (result / result.sum(axis=1, keepdims=True)).astype(np.float32)


def train_distilled_student(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_teacher_probabilities: np.ndarray | None,
    val_teacher_probabilities: np.ndarray | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    temperature: float,
    hard_label_weight: float,
) -> tuple[object, dict[str, list[float]]]:
    """Train from scratch with supervised loss and optional teacher probabilities."""

    tf = _tensorflow()
    set_global_seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    hard_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    history: dict[str, list[float]] = {"train_loss": [], "val_macro_f1": []}
    best_weights = None
    best_score = -np.inf
    stale_epochs = 0

    if train_teacher_probabilities is None:
        train_teacher_probabilities = np.zeros((len(y_train), 0), dtype=np.float32)
        val_teacher_probabilities = np.zeros((len(y_val), 0), dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, train_teacher_probabilities)
    )
    dataset = dataset.shuffle(len(y_train), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for _ in range(epochs):
        losses: list[float] = []
        for batch_x, batch_y, batch_teacher in dataset:
            with tf.GradientTape() as tape:
                student_logits = model(batch_x, training=True)
                hard_loss = hard_loss_fn(batch_y, student_logits)
                if int(batch_teacher.shape[-1]) > 0:
                    student_log_probs = tf.nn.log_softmax(student_logits / temperature, axis=1)
                    soft_loss = -tf.reduce_mean(
                        tf.reduce_sum(batch_teacher * student_log_probs, axis=1)
                    ) * (temperature**2)
                    loss = hard_label_weight * hard_loss + (1.0 - hard_label_weight) * soft_loss
                else:
                    loss = hard_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(float(loss.numpy()))

        val_logits = predict_logits(model, x_val, batch_size)
        val_f1 = float(classification_metrics(y_val, val_logits)["macro_f1"])
        history["train_loss"].append(float(np.mean(losses)))
        history["val_macro_f1"].append(val_f1)
        if val_f1 > best_score + 1e-7:
            best_score = val_f1
            best_weights = model.get_weights()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model, history


def model_footprint(model) -> dict[str, object]:
    weight_bytes = int(sum(np.asarray(weight).nbytes for weight in model.get_weights()))
    return {
        "parameters": int(model.count_params()),
        "weight_memory_mb": weight_bytes / (1024**2),
        "dtype_bytes_included": True,
    }


def serialized_model_size_mb(model, output_path: str | Path) -> float:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path, include_optimizer=False)
    return output_path.stat().st_size / (1024**2)


def benchmark_latency(
    model,
    sample: np.ndarray,
    warmups: int,
    repetitions: int,
) -> dict[str, object]:
    """Synchronous batch-size-one latency with warm-up and distributional reporting."""

    tf = _tensorflow()
    sample_tensor = tf.convert_to_tensor(sample[:1])
    for _ in range(warmups):
        model(sample_tensor, training=False).numpy()
    measurements: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        model(sample_tensor, training=False).numpy()
        measurements.append((time.perf_counter_ns() - started) / 1_000_000.0)
    values = np.asarray(measurements)
    return {
        "batch_size": 1,
        "warmups": warmups,
        "repetitions": repetitions,
        "median_ms": float(np.median(values)),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)) if repetitions > 1 else 0.0,
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "host": platform.node(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
    }
