"""Módulo de validación de entrada e ingeniería de características fisicoquímicas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ACCEPTABILITY_REQUIRED_COLUMNS,
    FIBRA_PROPERTY_COLUMNS,
    ID_FORMULACION_COL,
    ID_GOMITA_COL,
    INCLUDE_PROMEDIO_IN_ACCEPTABILITY,
    INGREDIENT_COLUMNS,
    TIPO_FIBRA_COL,
)


def fail(message: str) -> None:
    """Imprime un mensaje de error claro en stderr y finaliza la ejecución con código 1."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_and_validate_csv(
    filepath: Path,
    required_columns: list[str],
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Carga un archivo CSV verificando existencia, columnas obligatorias y cardinalidad."""
    if not filepath.exists():
        fail(
            f"No se encontró el archivo requerido '{filepath}'. "
            f"Columnas esperadas: {', '.join(required_columns)}."
        )

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except Exception as exc:
        fail(f"No se pudo leer el archivo '{filepath}': {exc}")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        fail(
            f"Al archivo '{filepath}' le faltan las siguientes columnas obligatorias: "
            f"{', '.join(missing_cols)}."
        )

    if expected_rows is not None and len(df) != expected_rows:
        fail(
            f"El archivo '{filepath}' debe contener exactamente {expected_rows} filas, "
            f"pero tiene {len(df)}."
        )

    return df


def validate_numeric_columns(
    df: pd.DataFrame, columns: list[str], filepath: Path
) -> pd.DataFrame:
    """Verifica que las columnas indicadas sean numéricas y sin valores faltantes."""
    out = df.copy()
    for col in columns:
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.isna().any():
            fail(
                f"El archivo '{filepath}' contiene valores nulos o no numéricos "
                f"en la columna '{col}'."
            )
        out[col] = converted.astype(float)
    return out


def engineer_features(
    df_form: pd.DataFrame,
    include_fiber_dummies: bool = True,
) -> pd.DataFrame:
    """Calcula variables físico-químicas derivadas y codificación one-hot del tipo de fibra.

    Variables derivadas generadas:
        - ratio_gelatina_goma_xantica: gelatina / goma_xantica
        - ratio_azucar_edulcorantes_totales: azucar / (stevia + sorbitol)
        - fraccion_fibra: fibra / peso_total
        - fraccion_agua: agua / peso_total
        - solidos_totales: peso_total - agua
        - humedad_estimada: (agua / peso_total) * 100
        - dummies para tipo_fibra
    """
    base = df_form[INGREDIENT_COLUMNS + FIBRA_PROPERTY_COLUMNS].astype(float).copy()

    peso_total = base[INGREDIENT_COLUMNS].sum(axis=1)
    peso_total_safe = peso_total.replace(0.0, np.nan)

    goma_safe = base["goma_xantica"].replace(0.0, np.nan)
    edulcorantes_totales = base["stevia"] + base["sorbitol"]
    edulcorantes_safe = edulcorantes_totales.replace(0.0, np.nan)

    base["ratio_gelatina_goma_xantica"] = (base["gelatina"] / goma_safe).fillna(0.0)
    base["ratio_azucar_edulcorantes_totales"] = (
        base["azucar"] / edulcorantes_safe
    ).fillna(0.0)
    base["fraccion_fibra"] = (base["fibra"] / peso_total_safe).fillna(0.0)
    base["fraccion_agua"] = (base["agua"] / peso_total_safe).fillna(0.0)
    base["solidos_totales"] = peso_total - base["agua"]
    base["humedad_estimada"] = (
        (base["agua"] / peso_total_safe) * 100.0
    ).fillna(0.0)

    if include_fiber_dummies:
        dummies = pd.get_dummies(
            df_form[TIPO_FIBRA_COL].astype(str),
            prefix=TIPO_FIBRA_COL,
            dtype=float,
        )
        base = pd.concat([base, dummies], axis=1)

    return base.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def encode_acceptability_features(
    df_accep: pd.DataFrame,
    df_formulaciones: pd.DataFrame,
) -> pd.DataFrame:
    """Codifica variables demográficas y hedónicas para el modelo de aceptabilidad."""
    likert_map = {
        "Me disgusta mucho": 1.0,
        "Me disgusta": 2.0,
        "Me es indiferente": 3.0,
        "Me gusta": 4.0,
        "Me gusta mucho": 5.0,
    }
    edad_map = {
        "6 a 12 años": 1.0,
        "De 13 a 17 años": 2.0,
        "18 a 24 años": 3.0,
        "25 a 34 años": 4.0,
        "35 a 44 años": 5.0,
        "45 a 54 años": 6.0,
        "55 a 64 años": 7.0,
        "Más de 65 años": 8.0,
    }
    sexo_map = {"Femenino": 0.0, "Masculino": 1.0, "Otro": 2.0}
    fuma_map = {"No": 0.0, "A veces": 1.0, "Si": 2.0, "Sí": 2.0}

    work = df_accep.copy()

    if set(work[ID_FORMULACION_COL].astype(str)).issubset(
        set(df_formulaciones[ID_GOMITA_COL].astype(str))
    ):
        form_feats = engineer_features(df_formulaciones, include_fiber_dummies=True)
        form_feats[ID_GOMITA_COL] = df_formulaciones[ID_GOMITA_COL].astype(str).values
        work[ID_FORMULACION_COL] = work[ID_FORMULACION_COL].astype(str)
        work = pd.merge(
            work,
            form_feats,
            left_on=ID_FORMULACION_COL,
            right_on=ID_GOMITA_COL,
            how="left",
        )

    feature_parts: list[pd.Series | pd.DataFrame] = []

    for col, mapping in [
        ("edad", edad_map),
        ("sexo", sexo_map),
        ("habito_fumar", fuma_map),
        ("apariencia", likert_map),
        ("color", likert_map),
        ("sabor", likert_map),
        ("masticar", likert_map),
    ]:
        numeric_col = pd.to_numeric(work[col], errors="coerce")
        if numeric_col.isna().all():
            mapped = work[col].astype(str).map(mapping)
            if mapped.isna().any():
                dummies = pd.get_dummies(work[col].astype(str), prefix=col, dtype=float)
                feature_parts.append(dummies)
            else:
                feature_parts.append(mapped.rename(col).astype(float))
        else:
            feature_parts.append(numeric_col.fillna(0.0).rename(col).astype(float))

    if INCLUDE_PROMEDIO_IN_ACCEPTABILITY:
        prom = pd.to_numeric(work["promedio_aceptacion"], errors="coerce").fillna(0.0)
        feature_parts.append(prom.rename("promedio_aceptacion").astype(float))

    extra_cols = [
        c
        for c in work.columns
        if c not in set(ACCEPTABILITY_REQUIRED_COLUMNS + [ID_GOMITA_COL])
    ]
    if extra_cols:
        feature_parts.append(work[extra_cols].astype(float))

    X_df = pd.concat(feature_parts, axis=1)
    return X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

