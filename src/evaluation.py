"""Cálculo de métricas, validación cruzada y generación de tablas del paper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from src.config import (
    ACCEPTABILITY_CLASSES,
    CATEGORIA_COL,
    EXPECTED_SENSORIAL_ROWS,
    FIBRA_PROPERTY_COLUMNS,
    FORMULACIONES_FILE,
    ID_FORMULACION_COL,
    ID_GOMITA_COL,
    INCLUDE_FIBER_DUMMIES_IN_VIABILITY,
    INGREDIENT_COLUMNS,
    OUTPUT_DIR,
    SEED,
    SENSORIAL_FILE,
    SENSORY_COLUMNS,
    SENSORY_DISPLAY_NAMES,
    SYNTHETIC_SAMPLE_SIZES,
    TIPO_FIBRA_COL,
    VIABLE_COL,
)
from src.features import encode_acceptability_features, engineer_features, fail
from src.models import (
    select_acceptability_model,
    select_cv_splitter,
    select_sensory_model,
    select_viability_model,
)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula RMSE usando mean_squared_error(squared=False) con fallback para sklearn >= 1.6."""
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fmt(value: Any) -> str:
    """Formatea un valor numérico a 2 decimales o 'N/A' si no es finito."""
    if value is None or isinstance(value, str):
        return str(value) if value is not None else "N/A"
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(value):
            return "N/A"
        return f"{float(value):.2f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla Markdown alineada."""
    headers = [str(col) for col in df.columns]
    rows = [[str(val) for val in row] for row in df.to_numpy()]
    widths = [
        max([len(headers[i])] + [len(r[i]) for r in rows])
        for i in range(len(headers))
    ]

    header_line = (
        "| "
        + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
        + " |"
    )
    separator_line = (
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    )
    body_lines = [
        "| "
        + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
        + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + body_lines)


def save_output_table(
    df: pd.DataFrame,
    filename_stem: str,
    title: str,
    summary_sections: list[tuple[str, str]],
    output_dir: Path = OUTPUT_DIR,
) -> str:
    """Exporta un DataFrame a CSV en output_dir y agrega su representación Markdown al resumen."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{filename_stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    md_text = dataframe_to_markdown(df)
    summary_sections.append((title, md_text))
    return md_text


def compute_table2(
    df_formulaciones: pd.DataFrame,
    df_sensorial: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula la Tabla 2: evaluación LOOCV pooled de los 10 atributos sensoriales."""
    merged = pd.merge(
        df_formulaciones,
        df_sensorial,
        on=ID_GOMITA_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_SENSORIAL_ROWS:
        fail(
            f"La unión entre '{FORMULACIONES_FILE}' y '{SENSORIAL_FILE}' por "
            f"'{ID_GOMITA_COL}' produjo {len(merged)} filas (se esperaban {EXPECTED_SENSORIAL_ROWS})."
        )

    if not set(merged[VIABLE_COL].astype(int).unique()).issubset({1}):
        fail(
            f"'{SENSORIAL_FILE}' debe contener únicamente formulaciones viables ({VIABLE_COL}=1)."
        )

    X_df = engineer_features(merged, include_fiber_dummies=True)
    X = X_df.to_numpy(dtype=float)
    n_samples = len(merged)

    base_model, _ = select_sensory_model(n_samples)
    loo = LeaveOneOut()

    rows: list[dict[str, Any]] = []
    numeric_metrics: list[dict[str, float]] = []

    for attr in SENSORY_COLUMNS:
        y = merged[attr].to_numpy(dtype=float)
        y_pred_pooled = np.empty(n_samples, dtype=float)
        y_base_pooled = np.empty(n_samples, dtype=float)

        for train_idx, test_idx in loo.split(X):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train_idx])
            X_test_scaled = scaler.transform(X[test_idx])

            model = clone(base_model)
            model.fit(X_train_scaled, y[train_idx])
            y_pred_pooled[test_idx] = model.predict(X_test_scaled)
            y_base_pooled[test_idx] = float(np.mean(y[train_idx]))

        mae_val = float(mean_absolute_error(y, y_pred_pooled))
        rmse_val = compute_rmse(y, y_pred_pooled)
        r2_val = float(r2_score(y, y_pred_pooled))
        base_mae_val = float(mean_absolute_error(y, y_base_pooled))
        base_rmse_val = compute_rmse(y, y_base_pooled)

        numeric_metrics.append(
            {
                "MAE": mae_val,
                "RMSE": rmse_val,
                "R²": r2_val,
                "Baseline MAE (media)": base_mae_val,
                "Baseline RMSE (media)": base_rmse_val,
            }
        )

        rows.append(
            {
                "Atributo sensorial": SENSORY_DISPLAY_NAMES[attr],
                "MAE": fmt(mae_val),
                "RMSE": fmt(rmse_val),
                "R²": fmt(r2_val),
                "Baseline MAE (media)": fmt(base_mae_val),
                "Baseline RMSE (media)": fmt(base_rmse_val),
            }
        )

    metrics_df = pd.DataFrame(numeric_metrics)
    rows.append(
        {
            "Atributo sensorial": "Promedio",
            "MAE": fmt(metrics_df["MAE"].mean()),
            "RMSE": fmt(metrics_df["RMSE"].mean()),
            "R²": fmt(metrics_df["R²"].mean()),
            "Baseline MAE (media)": fmt(metrics_df["Baseline MAE (media)"].mean()),
            "Baseline RMSE (media)": fmt(metrics_df["Baseline RMSE (media)"].mean()),
        }
    )

    return pd.DataFrame(rows)


def compute_table3(
    df_formulaciones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula la Tabla 3 (métricas y matriz de confusión) del modelo de viabilidad."""
    X_df = engineer_features(
        df_formulaciones,
        include_fiber_dummies=INCLUDE_FIBER_DUMMIES_IN_VIABILITY,
    )
    X = X_df.to_numpy(dtype=float)
    y = df_formulaciones[VIABLE_COL].astype(int).to_numpy()
    n_samples = len(df_formulaciones)

    base_model, _ = select_viability_model(n_samples)
    loo = LeaveOneOut()

    y_pred_pooled = np.full(n_samples, -1, dtype=int)
    y_prob_pooled = np.full(n_samples, np.nan, dtype=float)

    for train_idx, test_idx in loo.split(X):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])

        model = clone(base_model)
        model.fit(X_train_scaled, y_train)

        y_pred_pooled[test_idx] = model.predict(X_test_scaled).astype(int)
        y_prob_pooled[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

    valid_mask = (y_pred_pooled >= 0) & np.isfinite(y_prob_pooled)

    def safe_metric(fn: Any) -> str:
        """Evalúa una métrica capturando excepciones si falta alguna clase."""
        try:
            if not np.all(valid_mask):
                return "N/A"
            if len(np.unique(y)) < 2 or len(np.unique(y_pred_pooled)) < 2:
                return "N/A"
            return fmt(fn())
        except Exception:
            return "N/A"

    metrics_rows = [
        {
            "Métrica": "accuracy",
            "Valor": safe_metric(lambda: accuracy_score(y, y_pred_pooled)),
            "Baseline mayoritaria": fmt(17 / 25),
        },
        {
            "Métrica": "balanced_accuracy",
            "Valor": safe_metric(lambda: balanced_accuracy_score(y, y_pred_pooled)),
            "Baseline mayoritaria": fmt(0.50),
        },
        {
            "Métrica": "precision",
            "Valor": safe_metric(
                lambda: precision_score(y, y_pred_pooled, pos_label=1, zero_division=0)
            ),
            "Baseline mayoritaria": fmt(0.00),
        },
        {
            "Métrica": "recall",
            "Valor": safe_metric(
                lambda: recall_score(y, y_pred_pooled, pos_label=1, zero_division=0)
            ),
            "Baseline mayoritaria": fmt(0.00),
        },
        {
            "Métrica": "f1",
            "Valor": safe_metric(
                lambda: f1_score(y, y_pred_pooled, pos_label=1, zero_division=0)
            ),
            "Baseline mayoritaria": fmt(0.00),
        },
        {
            "Métrica": "roc_auc",
            "Valor": safe_metric(lambda: roc_auc_score(y, y_prob_pooled)),
            "Baseline mayoritaria": fmt(0.50),
        },
    ]
    df_metrics = pd.DataFrame(metrics_rows)

    if np.all(valid_mask):
        cm = confusion_matrix(y, y_pred_pooled, labels=[0, 1])
    else:
        cm = np.zeros((2, 2), dtype=int)

    df_confusion = pd.DataFrame(
        [
            {
                "": "Real: no viable",
                "Predicho: no viable": int(cm[0, 0]),
                "Predicho: viable": int(cm[0, 1]),
            },
            {
                "": "Real: viable",
                "Predicho: no viable": int(cm[1, 0]),
                "Predicho: viable": int(cm[1, 1]),
            },
        ]
    )

    return df_metrics, df_confusion


def generate_synthetic_dataset(
    n: int,
    df_formulaciones_ref: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Genera n muestras sintéticas dentro de los rangos de formulaciones.csv con ruido gaussiano."""
    synth_cols: dict[str, Any] = {
        ID_GOMITA_COL: [f"SYN_{i + 1}" for i in range(n)]
    }

    for col in INGREDIENT_COLUMNS + FIBRA_PROPERTY_COLUMNS:
        col_min = float(df_formulaciones_ref[col].min())
        col_max = float(df_formulaciones_ref[col].max())
        if np.isclose(col_min, col_max):
            synth_cols[col] = np.full(n, col_min, dtype=float)
        else:
            synth_cols[col] = rng.uniform(col_min, col_max, size=n)

    fiber_types = df_formulaciones_ref[TIPO_FIBRA_COL].astype(str).unique()
    synth_cols[TIPO_FIBRA_COL] = rng.choice(fiber_types, size=n)

    df_synth = pd.DataFrame(synth_cols)

    score_viab = (
        df_synth["gelatina"].to_numpy()
        + 0.5 * df_synth["fibra"].to_numpy()
        + rng.normal(0.0, 0.25, size=n)
    )
    y_viable = (score_viab >= float(np.median(score_viab))).astype(int)
    if len(np.unique(y_viable)) < 2:
        y_viable[0] = 0
        y_viable[-1] = 1
    df_synth[VIABLE_COL] = y_viable

    sensory_data: dict[str, np.ndarray] = {}
    for idx, attr in enumerate(SENSORY_COLUMNS):
        signal = (
            5.0
            + 0.1 * df_synth["azucar"].to_numpy()
            - 0.2 * df_synth["fibra"].to_numpy()
            + rng.normal(0.0, 0.35 + 0.02 * idx, size=n)
        )
        sensory_data[attr] = np.clip(signal, 1.0, 9.0)
    df_sensory_synth = pd.DataFrame(sensory_data)

    accept_score = score_viab + rng.normal(0.0, 0.3, size=n)
    quantiles = np.quantile(accept_score, [0.25, 0.50, 0.75])
    y_accept_idx = np.digitize(accept_score, quantiles)
    y_accept = np.array([ACCEPTABILITY_CLASSES[i] for i in y_accept_idx], dtype=object)

    return df_synth, df_sensory_synth, y_viable, y_accept


def compute_table5(df_formulaciones: pd.DataFrame) -> pd.DataFrame:
    """Calcula la Tabla 5: validación funcional de la lógica adaptativa con datos sintéticos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows: list[dict[str, Any]] = []
    previous_models: tuple[str, str, str] | None = None

    for n in SYNTHETIC_SAMPLE_SIZES:
        try:
            sens_model, sens_name = select_sensory_model(n)
            viab_model, viab_name = select_viability_model(n)
            accep_model, accep_name = select_acceptability_model(n)
            splitter = select_cv_splitter(n)

            df_synth, df_sens_synth, y_viab, y_accep = generate_synthetic_dataset(
                n, df_formulaciones, rng
            )

            X_df = engineer_features(df_synth, include_fiber_dummies=True)
            X = X_df.to_numpy(dtype=float)

            for train_idx, test_idx in splitter.split(X):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X[train_idx])
                X_test_scaled = scaler.transform(X[test_idx])

                m_sens = clone(sens_model)
                m_sens.fit(X_train_scaled, df_sens_synth["dulzura"].to_numpy()[train_idx])
                _ = m_sens.predict(X_test_scaled)

                if len(np.unique(y_viab[train_idx])) >= 2:
                    m_viab = clone(viab_model)
                    m_viab.fit(X_train_scaled, y_viab[train_idx])
                    _ = m_viab.predict(X_test_scaled)

                if len(np.unique(y_accep[train_idx])) >= 2:
                    m_accep = clone(accep_model)
                    m_accep.fit(X_train_scaled, y_accep[train_idx])
                    _ = m_accep.predict(X_test_scaled)

            full_pipeline = {
                "scaler": StandardScaler().fit(X),
                "sensorial": clone(sens_model).fit(
                    StandardScaler().fit_transform(X),
                    df_sens_synth["dulzura"].to_numpy(),
                ),
                "viabilidad": clone(viab_model).fit(
                    StandardScaler().fit_transform(X), y_viab
                ),
                "aceptabilidad": clone(accep_model).fit(
                    StandardScaler().fit_transform(X), y_accep
                ),
            }
            pkl_path = OUTPUT_DIR / f"pipeline_sintetico_n{n}.pkl"
            with open(pkl_path, "wb") as f_out:
                pickle.dump(full_pipeline, f_out)
            with open(pkl_path, "rb") as f_in:
                loaded_pipeline = pickle.load(f_in)
            _ = loaded_pipeline["sensorial"].predict(
                loaded_pipeline["scaler"].transform(X[:1])
            )
            pkl_path.unlink(missing_ok=True)

            current_models = (sens_name, viab_name, accep_name)
            if previous_models is not None and current_models != previous_models:
                resultado = "OK, transición correcta"
            else:
                resultado = "OK, sin errores"
            previous_models = current_models

        except Exception as exc:
            sens_name = "N/A"
            viab_name = "N/A"
            accep_name = "N/A"
            resultado = f"ERROR: {type(exc).__name__}: {exc}"

        rows.append(
            {
                "Tamaño muestral (n)": int(n),
                "Modelo sensorial seleccionado": sens_name,
                "Modelo de viabilidad seleccionado": viab_name,
                "Modelo de aceptabilidad seleccionado": accep_name,
                "Resultado": resultado,
            }
        )

    return pd.DataFrame(rows)


def compute_table4(
    df_formulaciones: pd.DataFrame,
    df_aceptabilidad: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calcula las subtablas de la Tabla 4 usando LeaveOneGroupOut por id_formulacion."""
    X_df = encode_acceptability_features(df_aceptabilidad, df_formulaciones)
    X = X_df.to_numpy(dtype=float)
    y = df_aceptabilidad[CATEGORIA_COL].astype(str).to_numpy()
    groups = df_aceptabilidad[ID_FORMULACION_COL].astype(str).to_numpy()

    n_total = len(df_aceptabilidad)
    base_model, _ = select_acceptability_model(n_total)
    logo = LeaveOneGroupOut()

    y_pred_pooled = np.empty(n_total, dtype=object)
    fold_rows: list[dict[str, Any]] = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])

        model = clone(base_model)
        model.fit(X_train_scaled, y[train_idx])
        y_pred_fold = model.predict(X_test_scaled)
        y_pred_pooled[test_idx] = y_pred_fold

        y_true_fold = y[test_idx]
        left_out_group = str(groups[test_idx[0]])

        all_classes_present = set(ACCEPTABILITY_CLASSES).issubset(
            set(y_true_fold)
        ) and set(ACCEPTABILITY_CLASSES).issubset(set(y_pred_fold))

        acc_fold = fmt(accuracy_score(y_true_fold, y_pred_fold))
        if all_classes_present:
            bal_acc_fold = fmt(balanced_accuracy_score(y_true_fold, y_pred_fold))
            macro_f1_fold = fmt(
                f1_score(
                    y_true_fold,
                    y_pred_fold,
                    labels=ACCEPTABILITY_CLASSES,
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            bal_acc_fold = "N/A"
            macro_f1_fold = "N/A"

        fold_rows.append(
            {
                "Formulación dejada fuera": left_out_group,
                "n test": int(len(test_idx)),
                "accuracy": acc_fold,
                "balanced_accuracy": bal_acc_fold,
                "macro_f1": macro_f1_fold,
            }
        )

    df_folds = pd.DataFrame(fold_rows)

    all_classes_global = set(ACCEPTABILITY_CLASSES).issubset(set(y)) and set(
        ACCEPTABILITY_CLASSES
    ).issubset(set(y_pred_pooled))

    df_global = pd.DataFrame(
        [
            {
                "Métrica": "accuracy",
                "Valor": fmt(accuracy_score(y, y_pred_pooled)),
            },
            {
                "Métrica": "balanced_accuracy",
                "Valor": (
                    fmt(balanced_accuracy_score(y, y_pred_pooled))
                    if all_classes_global
                    else "N/A"
                ),
            },
            {
                "Métrica": "macro_f1",
                "Valor": (
                    fmt(
                        f1_score(
                            y,
                            y_pred_pooled,
                            labels=ACCEPTABILITY_CLASSES,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if all_classes_global
                    else "N/A"
                ),
            },
        ]
    )

    counts = (
        pd.Series(y)
        .value_counts()
        .reindex(ACCEPTABILITY_CLASSES, fill_value=0)
    )
    df_dist = pd.DataFrame(
        {
            "Categoría": ACCEPTABILITY_CLASSES,
            "Cantidad": [int(counts[c]) for c in ACCEPTABILITY_CLASSES],
            "Porcentaje (%)": [
                fmt((counts[c] / n_total) * 100.0) for c in ACCEPTABILITY_CLASSES
            ],
        }
    )

    cm = confusion_matrix(y, y_pred_pooled, labels=ACCEPTABILITY_CLASSES)
    cm_rows: list[dict[str, Any]] = []
    for idx, real_cls in enumerate(ACCEPTABILITY_CLASSES):
        row_dict: dict[str, Any] = {"Real \\ Predicho": f"Real: {real_cls}"}
        for jdx, pred_cls in enumerate(ACCEPTABILITY_CLASSES):
            row_dict[f"Predicho: {pred_cls}"] = int(cm[idx, jdx])
        cm_rows.append(row_dict)
    df_confusion = pd.DataFrame(cm_rows)

    return {
        "tabla4_folds": df_folds,
        "tabla4_global": df_global,
        "tabla4_distribucion": df_dist,
        "tabla4_confusion": df_confusion,
    }

