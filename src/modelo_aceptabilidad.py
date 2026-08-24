"""
MODELO DE PREDICCIÓN DE ACEPTABILIDAD
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Entrena un modelo para predecir la aceptación del consumidor
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR = os.path.join(BASE_DIR, 'data', 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Crear carpetas
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'dataset_aceptabilidad_completo.xlsx')
OUTPUT_MODEL = os.path.join(MODELS_DIR, 'modelo_aceptabilidad.pkl')

# ==================== 1. CARGAR DATOS ====================

def load_data():
    print("📂 Cargando datos de aceptabilidad...")
    df = pd.read_excel(INPUT_FILE)
    print(f"   ✅ {len(df)} muestras cargadas")
    print(f"   📋 Columnas: {', '.join(df.columns.tolist())}")
    return df

# ==================== 2. PREPARAR DATOS ====================

def prepare_data(df):
    print("\n🔧 Preparando datos para el modelo...")
    
    # Variables de entrada (características demográficas y sensoriales)
    feature_cols = ['edad_num', 'sexo_num', 'fuma_num', 
                   'apariencia_num', 'color_num', 'sabor_num', 'sensacion_num']
    
    X = df[feature_cols]
    
    # Variable de salida (aceptación global)
    # Convertir a numérico: Baja=0, Media=1, Alta=2, Muy Alta=3
    aceptacion_map = {
        'Muy Baja': 0,
        'Baja': 0,
        'Media': 1,
        'Alta': 2,
        'Muy Alta': 3
    }
    y = df['aceptacion_global'].map(aceptacion_map)
    
    print(f"   ✅ Features: {len(feature_cols)}")
    print(f"   ✅ Target: aceptacion_global")
    print(f"   📋 Distribución de clases:")
    for clase, count in y.value_counts().items():
        nombre = {0: 'Baja', 1: 'Media', 2: 'Alta', 3: 'Muy Alta'}[clase]
        print(f"      {nombre}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols

# ==================== 3. ENTRENAR MODELO ====================

def train_model(X, y):
    print("\n🌲 Entrenando Random Forest Classifier...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✅ Entrenamiento: {len(X_train)} muestras")
    print(f"   ✅ Prueba: {len(X_test)} muestras")
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Rendimiento del modelo:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"\n   Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n   Validación Cruzada (5 folds):")
    print(f"   Accuracy promedio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Importancia de atributos
    importance = pd.DataFrame({
        'atributo': ['Edad', 'Sexo', 'Fuma', 'Apariencia', 'Color', 'Sabor', 'Sensación'],
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    print(f"\n   Importancia de atributos:")
    for _, row in importance.iterrows():
        print(f"     {row['atributo']}: {row['importancia']:.3f}")
    
    return model, importance, X_test, y_test, y_pred

# ==================== 4. VISUALIZACIONES ====================

def create_plots(y_test, y_pred, importance):
    print("\n📊 Generando visualizaciones...")
    
    # Mapeo de clases
    clase_map = {0: 'Baja', 1: 'Media', 2: 'Alta', 3: 'Muy Alta'}
    
    # 1. Matriz de Confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Baja', 'Media', 'Alta', 'Muy Alta'],
                yticklabels=['Baja', 'Media', 'Alta', 'Muy Alta'],
                ax=ax)
    ax.set_title('Matriz de Confusión - Aceptabilidad')
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'aceptabilidad_confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"   ✅ Matriz de Confusión guardada")
    
    # 2. Importancia de Atributos
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
    bars = ax.barh(importance['atributo'], importance['importancia'], color=colors)
    
    for bar, val in zip(bars, importance['importancia']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Importancia')
    ax.set_title('Importancia de Atributos para la Aceptación')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'aceptabilidad_importancia.png'), dpi=300)
    plt.close()
    print(f"   ✅ Importancia de atributos guardada")
    
    # 3. Distribución de predicciones vs reales
    fig, ax = plt.subplots(figsize=(10, 6))
    
    real_counts = pd.Series(y_test).map(clase_map).value_counts()
    pred_counts = pd.Series(y_pred).map(clase_map).value_counts()
    
    df_plot = pd.DataFrame({
        'Real': real_counts,
        'Predicho': pred_counts
    }).fillna(0)
    
    df_plot.plot(kind='bar', ax=ax, color=['#2E86AB', '#A2D5C6'])
    ax.set_title('Distribución de Clases: Real vs Predicho')
    ax.set_xlabel('Categoría de Aceptación')
    ax.set_ylabel('Cantidad')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'aceptabilidad_distribucion.png'), dpi=300)
    plt.close()
    print(f"   ✅ Distribución guardada")
    
    print("   ✅ Todas las visualizaciones generadas")

# ==================== 5. GUARDAR MODELO ====================

def save_model(model, importance):
    print("\n💾 Guardando modelo...")
    
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump({
            'model': model,
            'importance': importance,
            'features': ['edad_num', 'sexo_num', 'fuma_num', 
                        'apariencia_num', 'color_num', 'sabor_num', 'sensacion_num']
        }, f)
    print(f"   ✅ Modelo guardado: {OUTPUT_MODEL}")

# ==================== 6. FUNCIÓN PRINCIPAL ====================

def main():
    print("\n" + "="*60)
    print("🧪 MODELO DE ACEPTABILIDAD DEL CONSUMIDOR")
    print("="*60 + "\n")
    
    try:
        # 1. Cargar datos
        df = load_data()
        
        # 2. Preparar datos
        X, y, feature_cols = prepare_data(df)
        
        # 3. Entrenar modelo
        model, importance, X_test, y_test, y_pred = train_model(X, y)
        
        # 4. Visualizaciones
        create_plots(y_test, y_pred, importance)
        
        # 5. Guardar modelo
        save_model(model, importance)
        
        print("\n" + "="*60)
        print("✅ MODELO COMPLETADO CON ÉXITO")
        print("="*60)
        print(f"\n📊 Resumen:")
        print(f"   - Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"   - Atributo más importante: {importance.iloc[0]['atributo']}")
        print(f"\n📁 Archivos generados:")
        print(f"   - Modelo: {OUTPUT_MODEL}")
        print(f"   - Gráficos: {PLOTS_DIR}")
        print("="*60)
        
        return model
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    model = main()