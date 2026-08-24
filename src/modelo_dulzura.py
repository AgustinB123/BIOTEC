"""
MODELO DE PREDICCIÓN DE DULZURA - SEMANA 2
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Entrena un modelo Random Forest para predecir la dulzura
de una gomita en función de su receta.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR = os.path.join(BASE_DIR, 'data', 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Crear carpetas si no existen
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Archivos
INPUT_FILE = os.path.join(DATA_DIR, 'dataset_maestro_limpio.xlsx')
MODEL_FILE = os.path.join(MODELS_DIR, 'modelo_dulzura_rf.pkl')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler_dulzura.pkl')
FEATURES_FILE = os.path.join(MODELS_DIR, 'features_dulzura.pkl')

# Gráficos
PLOT_PRED_VS_REAL = os.path.join(PLOTS_DIR, 'dulzura_prediccion_vs_real.png')
PLOT_IMPORTANCIA = os.path.join(PLOTS_DIR, 'dulzura_importancia_variables.png')
PLOT_RESIDUOS = os.path.join(PLOTS_DIR, 'dulzura_residuos.png')
PLOT_TOP_10 = os.path.join(PLOTS_DIR, 'dulzura_top_10_variables.png')

# ==================== 1. CARGAR DATOS ====================

def load_data():
    """
    Carga el Dataset Maestro
    """
    print("📂 Cargando Dataset Maestro...")
    df = pd.read_excel(INPUT_FILE)
    print(f"   ✅ {len(df)} formulaciones cargadas")
    print(f"   📋 Columnas disponibles: {len(df.columns)}")
    return df

# ==================== 2. SELECCIONAR VARIABLES ====================

def select_variables(df):
    """
    Selecciona las variables de entrada (X) y salida (Y)
    """
    print("\n🔍 Seleccionando variables...")
    
    # Variable de salida
    target = 'dulzura'
    
    # Variables de entrada (excluyendo identificadores y otras variables sensoriales)
    exclude_cols = [
        'codigo_gomita', 'fibra_tipo',  # Identificadores
        'brillo', 'claridad', 'acidez', 'adhesividad',  # Otros atributos sensoriales
        'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad'
    ]
    
    # Seleccionar todas las columnas excepto las excluidas
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Asegurar que el target no esté en features
    if target in feature_cols:
        feature_cols.remove(target)
    
    # Crear X e Y
    X = df[feature_cols]
    y = df[target]
    
    print(f"   ✅ Variables de entrada (X): {len(feature_cols)}")
    print(f"   ✅ Variable de salida (Y): {target}")
    print(f"   📋 Primeras variables: {', '.join(feature_cols[:5])}...")
    
    return X, y, feature_cols

# ==================== 3. DIVIDIR DATOS ====================

def split_data(X, y):
    """
    Divide los datos en entrenamiento (75%) y prueba (25%)
    """
    print("\n📊 Dividiendo datos...")
    
    # Como tenemos pocos datos, usamos 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"   ✅ Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   ✅ Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   📋 Rango de dulzura en entrenamiento: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"   📋 Rango de dulzura en prueba: {y_test.min():.2f} - {y_test.max():.2f}")
    
    return X_train, X_test, y_train, y_test

# ==================== 4. ENTRENAR MODELO ====================

def train_model(X_train, y_train):
    """
    Entrena el modelo Random Forest Regressor
    """
    print("\n🌲 Entrenando Random Forest Regressor...")
    
    # Crear modelo con hiperparámetros específicos
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Entrenar
    model.fit(X_train, y_train)
    
    print(f"   ✅ Modelo entrenado")
    print(f"   📋 Número de árboles: {model.n_estimators}")
    print(f"   📋 Profundidad máxima: {model.max_depth}")
    
    return model

# ==================== 5. EVALUAR MODELO ====================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evalúa el modelo con múltiples métricas
    """
    print("\n📈 Evaluando modelo...")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas de entrenamiento
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = mean_squared_error(y_train, y_train_pred) ** 0.5
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    # Métricas de prueba
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred) ** 0.5
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    # Validación cruzada (con pocos datos, usar 3 folds)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("📊 RESULTADOS DEL MODELO")
    print("="*60)
    print(f"\nENTRENAMIENTO:")
    print(f"  R²:  {r2_train:.3f}")
    print(f"  RMSE: {rmse_train:.3f}")
    print(f"  MAE:  {mae_train:.3f}")
    
    print(f"\nPRUEBA:")
    print(f"  R²:  {r2_test:.3f}")
    print(f"  RMSE: {rmse_test:.3f}")
    print(f"  MAE:  {mae_test:.3f}")
    
    print(f"\nVALIDACIÓN CRUZADA (3 folds):")
    print(f"  R² promedio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("="*60)
    
    # Guardar métricas en diccionario
    metrics = {
        'r2_train': r2_train,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return y_train_pred, y_test_pred, metrics

# ==================== 6. IMPORTANCIA DE VARIABLES ====================

def get_feature_importance(model, feature_names):
    """
    Obtiene la importancia de cada variable
    """
    importance = pd.DataFrame({
        'variable': feature_names,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    print("\n📊 TOP 10 VARIABLES MÁS IMPORTANTES:")
    print("="*50)
    for i, row in importance.head(10).iterrows():
        print(f"  {row['variable']}: {row['importancia']:.3f}")
    
    return importance

# ==================== 7. VISUALIZACIONES ====================

def create_plots(y_test, y_test_pred, y_train, y_train_pred, importance, metrics):
    """
    Crea todos los gráficos del modelo
    """
    print("\n📊 Generando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Predicción vs Real
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Puntos de entrenamiento y prueba
    ax1.scatter(y_train, y_train_pred, alpha=0.7, s=100, label='Entrenamiento', color='blue')
    ax1.scatter(y_test, y_test_pred, alpha=0.9, s=150, label='Prueba', color='red', marker='s')
    
    # Línea diagonal (predicción perfecta)
    min_val = min(y_train.min(), y_test.min()) - 0.5
    max_val = max(y_train.max(), y_test.max()) + 0.5
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Predicción perfecta')
    
    ax1.set_xlabel('Dulzura Real', fontsize=12)
    ax1.set_ylabel('Dulzura Predicha', fontsize=12)
    ax1.set_title(f'Predicción vs Real - Dulzura\nR² Prueba: {metrics["r2_test"]:.3f} | RMSE: {metrics["rmse_test"]:.3f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PRED_VS_REAL, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 1: {PLOT_PRED_VS_REAL}")
    plt.close()
    
    # 2. Importancia de variables (todas)
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    top_vars = importance.head(15)
    bars = ax2.barh(top_vars['variable'], top_vars['importancia'], color=sns.color_palette("viridis", len(top_vars)))
    
    # Colorear las 5 más importantes
    for i, bar in enumerate(bars):
        if i < 5:
            bar.set_color('#2E86AB')
        else:
            bar.set_color('#A2D5C6')
    
    ax2.set_xlabel('Importancia', fontsize=12)
    ax2.set_title('Importancia de Variables - Dulzura', fontsize=14)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(PLOT_IMPORTANCIA, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 2: {PLOT_IMPORTANCIA}")
    plt.close()
    
    # 3. Top 10 variables (gráfico más limpio)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    top10 = importance.head(10)
    colors = sns.color_palette("viridis", len(top10))
    bars = ax3.barh(top10['variable'], top10['importancia'], color=colors)
    
    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, top10['importancia'])):
        ax3.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    ax3.set_xlabel('Importancia', fontsize=12)
    ax3.set_title('Top 10 Variables Más Importantes - Dulzura', fontsize=14)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(PLOT_TOP_10, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 3: {PLOT_TOP_10}")
    plt.close()
    
    # 4. Gráfico de Residuos
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # Residuos de entrenamiento y prueba
    residuos_train = y_train - y_train_pred
    residuos_test = y_test - y_test_pred
    
    ax4.scatter(y_train_pred, residuos_train, alpha=0.7, s=100, label='Entrenamiento', color='blue')
    ax4.scatter(y_test_pred, residuos_test, alpha=0.9, s=150, label='Prueba', color='red', marker='s')
    
    # Línea en 0 (residuo perfecto)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Dulzura Predicha', fontsize=12)
    ax4.set_ylabel('Residuos (Real - Predicho)', fontsize=12)
    ax4.set_title('Gráfico de Residuos - Dulzura', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_RESIDUOS, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 4: {PLOT_RESIDUOS}")
    plt.close()
    
    print("   ✅ Todas las visualizaciones generadas")

# ==================== 8. GUARDAR MODELO ====================

def save_model(model, scaler, feature_names):
    """
    Guarda el modelo entrenado y sus componentes
    """
    print("\n💾 Guardando modelo...")
    
    # Guardar modelo
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ Modelo guardado: {MODEL_FILE}")
    
    # Guardar scaler (aunque no se usó en Random Forest, puede ser útil)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler guardado: {SCALER_FILE}")
    
    # Guardar nombres de features
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"   ✅ Features guardados: {FEATURES_FILE}")

# ==================== 9. GENERAR INFORME ====================

def generate_report(metrics, importance, feature_names):
    """
    Genera un informe de resultados en formato texto
    """
    print("\n📝 Generando informe...")
    
    report_file = os.path.join(DATA_DIR, 'informe_modelo_dulzura.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("INFORME MODELO DE PREDICCIÓN DE DULZURA\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Fecha: 2026-08-24\n")
        f.write(f"Total de muestras: 8\n\n")
        
        f.write("MÉTRICAS DE EVALUACIÓN:\n")
        f.write("-"*40 + "\n")
        f.write(f"\nENTRENAMIENTO:\n")
        f.write(f"  R²:  {metrics['r2_train']:.3f}\n")
        f.write(f"  RMSE: {metrics['rmse_train']:.3f}\n")
        f.write(f"  MAE:  {metrics['mae_train']:.3f}\n")
        
        f.write(f"\nPRUEBA:\n")
        f.write(f"  R²:  {metrics['r2_test']:.3f}\n")
        f.write(f"  RMSE: {metrics['rmse_test']:.3f}\n")
        f.write(f"  MAE:  {metrics['mae_test']:.3f}\n")
        
        f.write(f"\nVALIDACIÓN CRUZADA (3 folds):\n")
        f.write(f"  R² promedio: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}\n\n")
        
        f.write("TOP 10 VARIABLES MÁS IMPORTANTES:\n")
        f.write("-"*40 + "\n")
        for i, row in importance.head(10).iterrows():
            f.write(f"  {row['variable']}: {row['importancia']:.3f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"Total de variables: {len(feature_names)}\n")
        f.write(f"Archivo del modelo: {MODEL_FILE}\n")
        f.write(f"Gráficos generados en: {PLOTS_DIR}\n")
        f.write("="*60 + "\n")
    
    print(f"   ✅ Informe guardado: {report_file}")

# ==================== 10. PREDECIR NUEVAS MUESTRAS ====================

def predict_sample(model, features, sample_data):
    """
    Función para predecir la dulzura de una nueva formulación
    """
    # Asegurar que los datos tengan el orden correcto de features
    sample_df = pd.DataFrame([sample_data])[features]
    prediction = model.predict(sample_df)[0]
    return prediction

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal que ejecuta todo el flujo
    """
    print("\n" + "="*60)
    print("🍬 MODELO DE PREDICCIÓN DE DULZURA - SEMANA 2")
    print("="*60 + "\n")
    
    try:
        # 1. Cargar datos
        df = load_data()
        
        # 2. Seleccionar variables
        X, y, feature_names = select_variables(df)
        
        # 3. Dividir datos
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 4. Entrenar modelo
        model = train_model(X_train, y_train)
        
        # 5. Evaluar modelo
        y_train_pred, y_test_pred, metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        
        # 6. Importancia de variables
        importance = get_feature_importance(model, feature_names)
        
        # 7. Visualizaciones
        create_plots(y_test, y_test_pred, y_train, y_train_pred, importance, metrics)
        
        # 8. Guardar modelo
        scaler = StandardScaler()  # No se usa pero se guarda por compatibilidad
        save_model(model, scaler, feature_names)
        
        # 9. Generar informe
        generate_report(metrics, importance, feature_names)
        
        # 10. Resumen final
        print("\n" + "="*60)
        print("✅ MODELO COMPLETADO CON ÉXITO")
        print("="*60)
        print(f"\n📊 Resumen de resultados:")
        print(f"   - R² (Prueba): {metrics['r2_test']:.3f}")
        print(f"   - RMSE (Prueba): {metrics['rmse_test']:.3f}")
        print(f"   - MAE (Prueba): {metrics['mae_test']:.3f}")
        print(f"\n📁 Archivos generados:")
        print(f"   - Modelo: {MODEL_FILE}")
        print(f"   - Informe: {os.path.join(DATA_DIR, 'informe_modelo_dulzura.txt')}")
        print(f"   - Gráficos: {PLOTS_DIR}")
        print("\n" + "="*60)
        
        return model, metrics, importance
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    model, metrics, importance = main()
    
    if model is not None:
        print("\n💡 El modelo está listo para usar en el Dashboard:")
        print(f"   - Cargar con: pickle.load(open('{MODEL_FILE}', 'rb'))")
        print(f"   - Predecir: model.predict(X_nuevo)")
        print(f"   - Ver gráficos en: {PLOTS_DIR}")