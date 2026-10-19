from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    input_kind: str
    builder: Callable[[tuple[int, ...], int], object]


def _keras():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model construction. Install requirements-colab.txt."
        ) from exc
    return tf, keras


def build_student(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = keras.layers.Conv1D(24, 5, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(48, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="makd_student")


def build_hcf_mlp(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="handcrafted_features")
    x = keras.layers.Dense(512, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="teacher_hcf_mlp")


def build_raw_cnn(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = inputs
    for filters in (64, 128, 192):
        x = keras.layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="teacher_raw_cnn")


def build_lstm(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = keras.layers.Bidirectional(keras.layers.LSTM(96, return_sequences=True))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
    x = keras.layers.Dense(96, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="teacher_bilstm")


def _residual_tcn_block(x, filters: int, dilation: int, keras):
    residual = x
    x = keras.layers.Conv1D(
        filters, 3, padding="causal", dilation_rate=dilation, activation="relu"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(x)
    if int(residual.shape[-1]) != filters:
        residual = keras.layers.Conv1D(filters, 1, padding="same")(residual)
    return keras.layers.Activation("relu")(keras.layers.Add()([x, residual]))


def build_tcn(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = inputs
    for dilation, filters in ((1, 64), (2, 96), (4, 128), (8, 128)):
        x = _residual_tcn_block(x, filters, dilation, keras)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="teacher_tcn")


def build_transformer(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = keras.layers.Dense(64)(inputs)
    for _ in range(2):
        attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = keras.layers.LayerNormalization()(x + attention)
        feed_forward = keras.layers.Dense(128, activation="gelu")(x)
        feed_forward = keras.layers.Dense(64)(feed_forward)
        x = keras.layers.LayerNormalization()(x + feed_forward)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="teacher_transformer")


def build_cnn_baseline(input_shape: tuple[int, ...], num_classes: int):
    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_slfn_baseline(input_shape: tuple[int, ...], num_classes: int):
    """A genuine single-hidden-layer feedforward network."""

    _, keras = _keras()
    inputs = keras.Input(shape=input_shape, name="raw_window")
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(128, activation="tanh", name="single_hidden_layer")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="baseline_slfn")


def teacher_registry() -> dict[str, TeacherSpec]:
    return {
        "hcf_mlp": TeacherSpec("hcf_mlp", "features", build_hcf_mlp),
        "raw_cnn": TeacherSpec("raw_cnn", "raw", build_raw_cnn),
        "lstm": TeacherSpec("lstm", "raw", build_lstm),
        "tcn": TeacherSpec("tcn", "raw", build_tcn),
        "transformer": TeacherSpec("transformer", "raw", build_transformer),
    }


def baseline_registry() -> dict[str, Callable[[tuple[int, ...], int], object]]:
    return {
        "cnn": build_cnn_baseline,
        "slfn": build_slfn_baseline,
        "lstm": build_lstm,
    }
