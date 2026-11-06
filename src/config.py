"""Configuración centralizada de rutas, esquemas de datos y umbrales adaptativos."""

from __future__ import annotations

from pathlib import Path

# Semilla global de reproducibilidad
SEED: int = 42

# Directorios del proyecto
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = (
    BASE_DIR / "datosPruba"
    if (BASE_DIR / "datosPruba" / "formulaciones.csv").exists()
    else BASE_DIR
)
OUTPUT_DIR: Path = BASE_DIR / "outputs"

# Archivos de entrada del paper
FORMULACIONES_FILE: Path = DATA_DIR / "formulaciones.csv"
SENSORIAL_FILE: Path = DATA_DIR / "perfil_sensorial.csv"
ACEPTABILIDAD_FILE: Path = DATA_DIR / "aceptabilidad.csv"

# Dimensiones esperadas en los conjuntos experimentales
EXPECTED_FORMULACIONES_ROWS: int = 25
EXPECTED_SENSORIAL_ROWS: int = 8
EXPECTED_ACEPTABILIDAD_ROWS: int = 243

# Umbrales de selección adaptativa de modelos
SENSORIAL_THRESHOLD_LOW: int = 15
SENSORIAL_THRESHOLD_HIGH: int = 50

VIABILIDAD_THRESHOLD_LOW: int = 30
VIABILIDAD_THRESHOLD_HIGH: int = 100

ACEPTABILIDAD_THRESHOLD_LOW: int = 30
ACEPTABILIDAD_THRESHOLD_HIGH: int = 100

# Umbrales de validación cruzada
CV_THRESHOLD_LOOCV: int = 15
CV_THRESHOLD_KFOLD5: int = 100

# Tamaños muestrales para validación funcional con datos sintéticos (Tabla 5)
SYNTHETIC_SAMPLE_SIZES: tuple[int, ...] = (10, 30, 60, 100, 200)

# Columnas clave y esquemas
ID_GOMITA_COL: str = "id_gomita"
TIPO_FIBRA_COL: str = "tipo_fibra"
VIABLE_COL: str = "viable"

INGREDIENT_COLUMNS: list[str] = [
    "azucar",
    "jarabe_glucosa",
    "agua",
    "gelatina",
    "goma_xantica",
    "acido_citrico",
    "stevia",
    "esencia",
    "sorbitol",
    "fibra",
]

FIBRA_PROPERTY_COLUMNS: list[str] = [
    "WHC",
    "SC",
    "OHC",
    "FRAP",
    "DPPH",
    "polifenoles",
]

FORMULACIONES_REQUIRED_COLUMNS: list[str] = (
    [ID_GOMITA_COL]
    + INGREDIENT_COLUMNS
    + [TIPO_FIBRA_COL, VIABLE_COL]
    + FIBRA_PROPERTY_COLUMNS
)

SENSORY_COLUMNS: list[str] = [
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

SENSORY_DISPLAY_NAMES: dict[str, str] = {
    "brillo": "Brillo",
    "claridad": "Claridad",
    "dulzura": "Dulzura",
    "acidez": "Acidez",
    "adhesividad": "Adhesividad",
    "elasticidad": "Elasticidad",
    "arenosidad": "Arenosidad",
    "dureza": "Dureza",
    "gomosidad": "Gomosidad",
    "masticabilidad": "Masticabilidad",
}

SENSORIAL_REQUIRED_COLUMNS: list[str] = [ID_GOMITA_COL] + SENSORY_COLUMNS

ID_FORMULACION_COL: str = "id_formulacion"
CATEGORIA_COL: str = "categoria"

ACCEPTABILITY_REQUIRED_COLUMNS: list[str] = [
    ID_FORMULACION_COL,
    "edad",
    "sexo",
    "habito_fumar",
    "apariencia",
    "color",
    "sabor",
    "masticar",
    "promedio_aceptacion",
    CATEGORIA_COL,
]

ACCEPTABILITY_CLASSES: list[str] = ["Baja", "Media", "Alta", "Muy Alta"]

# Opciones de ingeniería de características
INCLUDE_FIBER_DUMMIES_IN_VIABILITY: bool = True
INCLUDE_PROMEDIO_IN_ACCEPTABILITY: bool = False

