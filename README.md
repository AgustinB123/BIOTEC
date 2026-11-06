# BIOTEC MOQUEADOS — Sistema Adaptativo de Formulación y Evaluación Sensorial

Repositorio de ingeniería de características, modelado adaptativo de machine learning, optimización de formulaciones y reproducción de métricas experimentales para gomitas funcionales enriquecidas con fibra de batata (_Ipomoea batatas_).

---

## Estructura del Repositorio

```text
.
├── calcular_metricas_paper.py   # Script reproducible principal para generar las tablas del paper
├── src/                         # Paquete modular de evaluación científica y ML
│   ├── __init__.py
│   ├── config.py                # Rutas, esquemas de datos, semillas y umbrales adaptativos
│   ├── features.py              # Validación de entradas e ingeniería de características
│   ├── models.py                # Lógica adaptativa de selección de modelos y validación cruzada
│   └── evaluation.py            # Cálculo de métricas (Tablas 2, 3, 4 y 5) y exportación
├── dashboard/                   # Aplicación interactiva en Streamlit
│   ├── pipeline_app.py          # Orquestador principal de los 6 módulos
│   ├── utils.py                 # Funciones compartidas de rutas, modelos y predicción
│   ├── cargar_datos.py          # Módulo 1: Carga y validación de datasets
│   ├── entrenar_modelos.py      # Módulo 2: Entrenamiento y evaluación de modelos
│   ├── recomendador_recetas.py  # Módulo 3: Optimización mediante algoritmos genéticos
│   ├── perfil_sensorial.py      # Módulo 4: Simulador de perfil sensorial (escala 1–9)
│   ├── prediccion_aceptabilidad.py # Módulo 5: Predicción de aceptabilidad del consumidor
│   └── comparacion_formulaciones.py # Módulo 6: Comparación multidimensional de recetas
├── datosPruba/                  # Conjuntos de datos de prueba y validación del paper
│   ├── formulaciones.csv        # 25 formulaciones experimentales (8 viables, 17 no viables)
│   ├── perfil_sensorial.csv     # 8 formulaciones viables evaluadas en 10 atributos
│   ├── ingredientes.xlsx
│   ├── Perfil_Sensorial.xlsx
│   ├── propiedades_fibra.xlsx
│   └── aceptabilidad.xlsx
├── data/processed/              # Datasets procesados e informes del dashboard
├── models/                      # Modelos y escaladores serializados (.pkl)
├── outputs/                     # Tablas generadas en CSV y resumen.md del paper
└── requirements.txt             # Dependencias del proyecto
```

---

## Instalación y Configuración

1. **Crear y activar el entorno virtual:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Reproducir las Tablas del Paper Científico

Ejecutar el script principal desde la raíz del proyecto:

### 2. Ejecutar el Dashboard Interactivo

```bash
streamlit run dashboard/pipeline_app.py
```
