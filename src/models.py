"""Selección adaptativa de estimadores y esquemas de validación cruzada según tamaño muestral."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, LeaveOneOut

from src.config import (
    ACEPTABILIDAD_THRESHOLD_HIGH,
    ACEPTABILIDAD_THRESHOLD_LOW,
    CV_THRESHOLD_KFOLD5,
    CV_THRESHOLD_LOOCV,
    SEED,
    SENSORIAL_THRESHOLD_HIGH,
    SENSORIAL_THRESHOLD_LOW,
    VIABILIDAD_THRESHOLD_HIGH,
    VIABILIDAD_THRESHOLD_LOW,
)


def select_sensory_model(n: int) -> tuple[Any, str]:
    """Selecciona el modelo de regresión sensorial en función de n."""
    if n < SENSORIAL_THRESHOLD_LOW:
        return Ridge(alpha=1.0), "Ridge(alpha=1.0)"
    if n < SENSORIAL_THRESHOLD_HIGH:
        return (
            RandomForestRegressor(
                n_estimators=30, max_depth=3, random_state=SEED
            ),
            "RandomForestRegressor(n_estimators=30, max_depth=3)",
        )
    return (
        RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=SEED
        ),
        "RandomForestRegressor(n_estimators=100, max_depth=10)",
    )


def select_viability_model(n: int) -> tuple[Any, str]:
    """Selecciona el clasificador binario de viabilidad en función de n."""
    if n < VIABILIDAD_THRESHOLD_LOW:
        return (
            LogisticRegression(
                C=0.5, class_weight="balanced", random_state=SEED, max_iter=1000
            ),
            'LogisticRegression(C=0.5, class_weight="balanced")',
        )
    if n < VIABILIDAD_THRESHOLD_HIGH:
        return (
            RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                class_weight="balanced",
                random_state=SEED,
            ),
            "RandomForestClassifier(n_estimators=50, max_depth=4)",
        )
    return (
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForestClassifier(n_estimators=100, max_depth=10)",
    )


def select_acceptability_model(n: int) -> tuple[Any, str]:
    """Selecciona el clasificador multiclase de aceptabilidad en función de n."""
    if n < ACEPTABILIDAD_THRESHOLD_LOW:
        return (
            RandomForestClassifier(
                n_estimators=30,
                max_depth=4,
                class_weight="balanced",
                random_state=SEED,
            ),
            "RandomForestClassifier(n_estimators=30, max_depth=4)",
        )
    if n < ACEPTABILIDAD_THRESHOLD_HIGH:
        return (
            RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                class_weight="balanced",
                random_state=SEED,
            ),
            "RandomForestClassifier(n_estimators=50, max_depth=4)",
        )
    return (
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForestClassifier(n_estimators=100, max_depth=10)",
    )


def select_cv_splitter(n: int) -> Any:
    """Selecciona el esquema de validación cruzada según el tamaño muestral n."""
    if n < CV_THRESHOLD_LOOCV:
        return LeaveOneOut()
    if n < CV_THRESHOLD_KFOLD5:
        return KFold(n_splits=min(5, n), shuffle=True, random_state=SEED)
    return KFold(n_splits=10, shuffle=True, random_state=SEED)

