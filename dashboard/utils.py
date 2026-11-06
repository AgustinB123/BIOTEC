"""Utilidades compartidas para los módulos del dashboard interactivo en Streamlit."""

from __future__ import annotations

import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SENSORY_TARGETS: list[str] = [
    "brillo",
    "claridad",
    "dulzura",
    "acidez",
    "adhesividad",
    "elasticidad",
    "arenosidad",
    "dureza",
    "gomosidad",
    "masticabilidad",
]


def get_project_paths() -> dict[str, Path]:
    """Devuelve las rutas estándar de datos, modelos e informes del proyecto."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    reports_dir = data_dir / "informes"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "backup_dir": data_dir / "backups",
        "formulaciones": data_dir / "formulaciones.xlsx",
        "sensorial": data_dir / "perfil_sensorial.xlsx",
        "propiedades": data_dir / "propiedades_fibra.xlsx",
        "aceptabilidad": data_dir / "aceptabilidad.xlsx",
        "dataset_procesado": data_dir / "dataset_procesado.xlsx",
        "modelos_sensoriales": models_dir / "modelos_sensoriales.pkl",
        "modelo_aceptabilidad": models_dir / "modelo_aceptabilidad.pkl",
        "escaladores": models_dir / "escaladores.pkl",
        "features": models_dir / "features.pkl",
    }


def crear_backup_si_existe(filepath: Path) -> Path | None:
    """Respalda un archivo existente con marca temporal antes de sobrescribirlo."""
    if filepath.exists():
        backup_dir = filepath.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        shutil.move(str(filepath), str(backup_path))
        return backup_path
    return None


def cargar_modelos_sensoriales() -> tuple[Any, Any, Any, str | None]:
    """Carga los modelos sensoriales, escaladores y lista de variables desde disco."""
    paths = get_project_paths()
    try:
        with open(paths["modelos_sensoriales"], "rb") as f_mod:
            modelos = pickle.load(f_mod)
        with open(paths["escaladores"], "rb") as f_esc:
            escaladores = pickle.load(f_esc)
        with open(paths["features"], "rb") as f_feat:
            features = pickle.load(f_feat)
        return modelos, escaladores, features, None
    except Exception as exc:
        return None, None, None, str(exc)


def cargar_todos_los_modelos() -> tuple[Any, Any, Any, Any, str | None]:
    """Carga el modelo de aceptabilidad junto con los modelos sensoriales y escaladores."""
    paths = get_project_paths()
    modelos, escaladores, features, err = cargar_modelos_sensoriales()
    if err is not None:
        return None, None, None, None, err
    try:
        with open(paths["modelo_aceptabilidad"], "rb") as f_accep:
            modelo_aceptabilidad = pickle.load(f_accep)
        return modelo_aceptabilidad, modelos, escaladores, features, None
    except Exception as exc:
        return None, None, None, None, str(exc)


def get_ingredient_limits() -> dict[str, tuple[float, float]]:
    """Retorna los rangos experimentales válidos (mínimo, máximo) de cada ingrediente en gramos."""
    return {
        "azucar_g": (0.2, 11.4),
        "jarabe_glucosa_g": (0.3, 7.8),
        "agua_g": (0.5, 8.922),
        "gelatina_g": (0.0, 1.2),
        "goma_xantica_g": (0.0, 1.2),
        "acido_citrico_g": (0.05, 0.6),
        "stevia_g": (0.0, 1.4),
        "esencia_g": (0.05, 0.6),
        "sorbitol_g": (0.0, 5.5),
        "fibra_g": (0.0, 2.6),
    }


def get_ingredient_descriptions(short: bool = False) -> dict[str, str]:
    """Retorna las descripciones funcionales de cada ingrediente de la formulación."""
    if short:
        return {
            "azucar_g": "Azúcar",
            "jarabe_glucosa_g": "Jarabe de Glucosa",
            "agua_g": "Agua",
            "gelatina_g": "Gelatina",
            "goma_xantica_g": "Goma Xántica",
            "acido_citrico_g": "Ácido Cítrico",
            "stevia_g": "Stevia",
            "esencia_g": "Esencia",
            "sorbitol_g": "Sorbitol",
            "fibra_g": "Fibra",
        }
    return {
        "azucar_g": "Azúcar: aporta dulzura y cuerpo a la matriz.",
        "jarabe_glucosa_g": "Jarabe de glucosa: controla la cristalización y aporta humedad.",
        "agua_g": "Agua: medio de disolución e hidratación de hidrocoloides.",
        "gelatina_g": "Gelatina: agente gelificante principal responsable de la elasticidad.",
        "goma_xantica_g": "Goma xántica: hidrocoloide estabilizante y retenedor de agua.",
        "acido_citrico_g": "Ácido cítrico: regulador de pH y potenciador del perfil frutal.",
        "stevia_g": "Stevia: edulcorante no calórico de alta intensidad.",
        "esencia_g": "Esencia: componente aromático y saborizante.",
        "sorbitol_g": "Sorbitol: poliol humectante y edulcorante secundario.",
        "fibra_g": "Fibra dietaria: ingrediente funcional antioxidante y estructurante.",
    }


def predecir_perfil(
    receta: dict[str, Any],
    modelos: dict[str, Any],
    escaladores: dict[str, Any],
    targets: list[str],
    features: list[str],
) -> dict[str, float]:
    """Predice los atributos sensoriales (escala 1-9) para una formulación dada."""
    receta_df = pd.DataFrame([receta])
    for col in features:
        if col not in receta_df.columns:
            receta_df[col] = 0.0
    receta_df = receta_df[features]

    predicciones: dict[str, float] = {}
    for target in targets:
        try:
            X_scaled = escaladores[target].transform(receta_df)
            predicciones[target] = float(modelos[target].predict(X_scaled)[0])
        except Exception:
            predicciones[target] = 0.0
    return predicciones

