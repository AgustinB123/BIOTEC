"""
MODELO DE PREDICCIÓN DE PERFIL SENSORIAL COMPLETO - SEMANA 3
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Entrena modelos Random Forest para predecir los 10 atributos sensoriales
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
    Selecciona las variables de entrada (X) y las variables de salida (Y)
    """
    print("\n🔍 Seleccionando variables...")
    
    # Variables de salida (atributos sensoriales)
    targets = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
               'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
    
    # Variables de entrada (excluyendo identificadores y variables sensoriales)
    exclude_cols = [
        'codigo_gomita', 'fibra_tipo',  # Identificadores
        'brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad',  # Atributos sensoriales
        'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad'
    ]
    
    # Seleccionar todas las columnas excepto las excluidas
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Crear X e Y
    X = df[feature_cols]
    Y = df[targets]
    
    print(f"   ✅ Variables de entrada (X): {len(feature_cols)}")
    print(f"   ✅ Variables de salida (Y): {len(targets)}")
    print(f"   📋 Atributos a predecir: {', '.join(targets)}")
    
    return X, Y, feature_cols, targets

# ==================== 3. DIVIDIR DATOS ====================

def split_data(X, y):
    """
    Divide los datos en entrenamiento (75%) y prueba (25%)
    """
    print("\n📊 Dividiendo datos...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"   ✅ Entrenamiento: {len(X_train)} muestras (75.0%)")
    print(f"   ✅ Prueba: {len(X_test)} muestras (25.0%)")
    
    return X_train, X_test, y_train, y_test

# ==================== 4. ENTRENAR MODELOS ====================

def train_models(X_train, y_train, targets):
    """
    Entrena un modelo Random Forest para cada atributo sensorial
    """
    print("\n🌲 Entrenando modelos Random Forest...")
    print("="*60)
    
    models = {}
    scalers = {}
    
    for i, target in enumerate(targets):
        print(f"\n📈 Entrenando modelo para: {target.upper()}")
        
        # Escalar datos (opcional, pero mejora estabilidad)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Entrenar modelo
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train[target])
        
        # Guardar modelo y scaler
        models[target] = model
        scalers[target] = scaler
        
        # Evaluación rápida en entrenamiento
        y_pred = model.predict(X_train_scaled)
        r2 = r2_score(y_train[target], y_pred)
        rmse = mean_squared_error(y_train[target], y_pred) ** 0.5
        mae = mean_absolute_error(y_train[target], y_pred)
        
        print(f"   ✅ Modelo entrenado - R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    print("\n" + "="*60)
    print(f"   ✅ {len(models)} modelos entrenados exitosamente")
    
    return models, scalers

# ==================== 5. EVALUAR MODELOS ====================

def evaluate_models(models, scalers, X_test, y_test, targets):
    """
    Evalúa todos los modelos en el conjunto de prueba
    """
    print("\n📈 Evaluando modelos en conjunto de prueba...")
    print("="*60)
    
    results = {}
    predictions = {}
    
    for target in targets:
        # Escalar datos de prueba
        X_test_scaled = scalers[target].transform(X_test)
        
        # Predecir
        y_pred = models[target].predict(X_test_scaled)
        predictions[target] = y_pred
        
        # Calcular métricas
        r2 = r2_score(y_test[target], y_pred)
        rmse = mean_squared_error(y_test[target], y_pred) ** 0.5
        mae = mean_absolute_error(y_test[target], y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(models[target], X_test_scaled, y_test[target], cv=3, scoring='r2')
        
        results[target] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"\n📊 {target.upper()}:")
        print(f"   R²:  {r2:.3f}")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE:  {mae:.3f}")
        print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    print("\n" + "="*60)
    
    return results, predictions

# ==================== 6. IMPORTANCIA DE VARIABLES ====================

def get_feature_importance(models, feature_names, targets):
    """
    Obtiene la importancia de variables para cada modelo
    """
    print("\n📊 Importancia de variables por atributo...")
    print("="*60)
    
    importance_dict = {}
    
    for target in targets:
        importance = pd.DataFrame({
            'variable': feature_names,
            'importancia': models[target].feature_importances_
        }).sort_values('importancia', ascending=False)
        
        importance_dict[target] = importance
        
        print(f"\n📈 TOP 5 variables para {target.upper()}:")
        for i, row in importance.head(5).iterrows():
            print(f"   {row['variable']}: {row['importancia']:.3f}")
    
    return importance_dict

# ==================== 7. VISUALIZACIONES ====================

def create_plots(y_test, predictions, targets, results):
    """
    Crea visualizaciones para todos los modelos
    """
    print("\n📊 Generando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Gráfico de Predicción vs Real para cada atributo
    n_targets = len(targets)
    cols = 3
    rows = (n_targets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        # Datos
        y_true = y_test[target]
        y_pred = predictions[target]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.7, s=50, color='blue')
        
        # Línea diagonal (predicción perfecta)
        min_val = min(y_true.min(), y_pred.min()) - 0.5
        max_val = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Métricas
        r2 = results[target]['r2']
        rmse = results[target]['rmse']
        mae = results[target]['mae']
        
        ax.set_xlabel('Valor Real', fontsize=10)
        ax.set_ylabel('Valor Predicho', fontsize=10)
        ax.set_title(f'{target.upper()}\nR²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Ocultar ejes vacíos
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Guardar
    output_file = os.path.join(PLOTS_DIR, 'sensorial_prediccion_vs_real.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 1: {output_file}")
    plt.close()
    
    # 2. Gráfico de Residuos
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        y_true = y_test[target]
        y_pred = predictions[target]
        residuos = y_true - y_pred
        
        ax.scatter(y_pred, residuos, alpha=0.7, s=50, color='green')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Valor Predicho', fontsize=10)
        ax.set_ylabel('Residuos', fontsize=10)
        ax.set_title(f'{target.upper()} - Residuos', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    output_file = os.path.join(PLOTS_DIR, 'sensorial_residuos.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 2: {output_file}")
    plt.close()
    
    # 3. Gráfico de Barras - R² de todos los modelos
    fig, ax = plt.subplots(figsize=(12, 6))
    
    r2_values = [results[target]['r2'] for target in targets]
    colors = ['#2E86AB' if r2 >= 0.80 else '#A2D5C6' for r2 in r2_values]
    
    bars = ax.bar(targets, r2_values, color=colors)
    
    # Añadir valores
    for bar, r2 in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=0.80, color='r', linestyle='--', label='Objetivo R² > 0.80')
    ax.set_xlabel('Atributo Sensorial', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Rendimiento de Modelos por Atributo Sensorial', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    output_file = os.path.join(PLOTS_DIR, 'sensorial_r2_comparativo.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 3: {output_file}")
    plt.close()
    
    # 4. Mapa de Calor de Importancia de Variables (Top 10)
    print("   ✅ Gráfico 4: Top variables")
    # (Se genera en la función separada)
    
    print("   ✅ Todas las visualizaciones generadas")

def create_importance_heatmap(importance_dict, targets):
    """
    Crea un mapa de calor con las variables más importantes
    """
    # Obtener top 10 variables de todos los modelos combinados
    all_vars = []
    for target in targets:
        top_vars = importance_dict[target].head(10)['variable'].tolist()
        all_vars.extend(top_vars)
    
    # Variables únicas más importantes
    top_vars = list(dict.fromkeys(all_vars))[:12]
    
    # Crear matriz de importancias
    importance_matrix = []
    for target in targets:
        row = []
        for var in top_vars:
            if var in importance_dict[target]['variable'].values:
                val = importance_dict[target][importance_dict[target]['variable'] == var]['importancia'].values[0]
            else:
                val = 0
            row.append(val)
        importance_matrix.append(row)
    
    # Crear heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Configurar ejes
    ax.set_xticks(np.arange(len(top_vars)))
    ax.set_yticks(np.arange(len(targets)))
    ax.set_xticklabels(top_vars, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([t.upper() for t in targets], fontsize=10)
    
    # Añadir valores
    for i in range(len(targets)):
        for j in range(len(top_vars)):
            text = ax.text(j, i, f'{importance_matrix[i][j]:.2f}',
                          ha='center', va='center', color='black', fontsize=7)
    
    ax.set_title('Importancia de Variables por Atributo Sensorial', fontsize=14)
    plt.tight_layout()
    
    output_file = os.path.join(PLOTS_DIR, 'sensorial_importancia_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico 4: {output_file}")
    plt.close()

# ==================== 8. GUARDAR MODELOS ====================

def save_models(models, scalers, feature_names, targets):
    """
    Guarda todos los modelos entrenados
    """
    print("\n💾 Guardando modelos...")
    
    # Guardar cada modelo individualmente
    for target in targets:
        model_file = os.path.join(MODELS_DIR, f'modelo_{target}_rf.pkl')
        scaler_file = os.path.join(MODELS_DIR, f'scaler_{target}.pkl')
        
        with open(model_file, 'wb') as f:
            pickle.dump(models[target], f)
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(scalers[target], f)
        
        print(f"   ✅ Modelo {target}: {model_file}")
    
    # Guardar diccionario completo
    all_models_file = os.path.join(MODELS_DIR, 'modelos_sensoriales_completos.pkl')
    with open(all_models_file, 'wb') as f:
        pickle.dump({
            'models': models,
            'scalers': scalers,
            'features': feature_names,
            'targets': targets
        }, f)
    print(f"   ✅ Modelos completos: {all_models_file}")
    
    # Guardar features
    features_file = os.path.join(MODELS_DIR, 'features_sensoriales.pkl')
    with open(features_file, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"   ✅ Features: {features_file}")

# ==================== 9. GENERAR INFORME ====================

def generate_report(results, importance_dict, targets, feature_names):
    """
    Genera un informe completo de resultados
    """
    print("\n📝 Generando informe...")
    
    report_file = os.path.join(DATA_DIR, 'informe_modelos_sensoriales.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("INFORME MODELOS DE PREDICCIÓN SENSORIAL\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Fecha: 2026-08-24\n")
        f.write(f"Total de muestras: 232\n")
        f.write(f"Variables de entrada: {len(feature_names)}\n")
        f.write(f"Variables de salida: {len(targets)}\n\n")
        
        f.write("RESUMEN DE RENDIMIENTO\n")
        f.write("-"*40 + "\n\n")
        f.write(f"{'Atributo':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'CV R²':<10}\n")
        f.write("-"*60 + "\n")
        
        for target in targets:
            r2 = results[target]['r2']
            rmse = results[target]['rmse']
            mae = results[target]['mae']
            cv = results[target]['cv_mean']
            f.write(f"{target:<15} {r2:.3f}{' '*6} {rmse:.3f}{' '*6} {mae:.3f}{' '*6} {cv:.3f}\n")
        
        f.write("\n\nTOP 5 VARIABLES MÁS IMPORTANTES POR ATRIBUTO\n")
        f.write("-"*40 + "\n")
        
        for target in targets:
            f.write(f"\n{target.upper()}:\n")
            top_vars = importance_dict[target].head(5)
            for i, row in top_vars.iterrows():
                f.write(f"  {row['variable']}: {row['importancia']:.3f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ARCHIVOS GENERADOS\n")
        f.write("-"*40 + "\n")
        f.write(f"  Modelos: {MODELS_DIR}\n")
        f.write(f"  Gráficos: {PLOTS_DIR}\n")
        f.write("="*60 + "\n")
    
    print(f"   ✅ Informe guardado: {report_file}")

# ==================== 10. FUNCIÓN DE PREDICCIÓN ====================

def predict_sensorial(formulacion, models_dir=MODELS_DIR):
    """
    Función para predecir todos los atributos sensoriales de una nueva formulación
    
    Args:
        formulacion: dict con los valores de ingredientes
        models_dir: directorio donde están los modelos
    
    Returns:
        dict con las predicciones de los 10 atributos sensoriales
    """
    # Cargar modelos
    models_file = os.path.join(models_dir, 'modelos_sensoriales_completos.pkl')
    with open(models_file, 'rb') as f:
        data = pickle.load(f)
    
    models = data['models']
    scalers = data['scalers']
    features = data['features']
    targets = data['targets']
    
    # Crear DataFrame con la formulación
    X_new = pd.DataFrame([formulacion])[features]
    
    # Predecir cada atributo
    predicciones = {}
    for target in targets:
        X_scaled = scalers[target].transform(X_new)
        predicciones[target] = models[target].predict(X_scaled)[0]
    
    return predicciones

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal
    """
    print("\n" + "="*60)
    print("🧪 MODELOS DE PREDICCIÓN SENSORIAL COMPLETA - SEMANA 3")
    print("="*60 + "\n")
    
    try:
        # 1. Cargar datos
        df = load_data()
        
        # 2. Seleccionar variables
        X, y, feature_names, targets = select_variables(df)
        
        # 3. Dividir datos
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 4. Entrenar modelos
        models, scalers = train_models(X_train, y_train, targets)
        
        # 5. Evaluar modelos
        results, predictions = evaluate_models(models, scalers, X_test, y_test, targets)
        
        # 6. Importancia de variables
        importance_dict = get_feature_importance(models, feature_names, targets)
        
        # 7. Visualizaciones
        create_plots(y_test, predictions, targets, results)
        create_importance_heatmap(importance_dict, targets)
        
        # 8. Guardar modelos
        save_models(models, scalers, feature_names, targets)
        
        # 9. Generar informe
        generate_report(results, importance_dict, targets, feature_names)
        
        # 10. Resumen final
        print("\n" + "="*60)
        print("✅ MODELOS COMPLETADOS CON ÉXITO")
        print("="*60)
        
        print("\n📊 RESULTADOS POR ATRIBUTO:")
        print("-"*40)
        for target in targets:
            r2 = results[target]['r2']
            mae = results[target]['mae']
            status = "✅" if r2 > 0.80 else "⚠️" if r2 > 0.60 else "❌"
            print(f"  {status} {target.upper()}: R²={r2:.3f}, MAE={mae:.3f}")
        
        print("\n📁 Archivos generados:")
        print(f"   - Modelos: {MODELS_DIR}")
        print(f"   - Gráficos: {PLOTS_DIR}")
        print(f"   - Informe: {os.path.join(DATA_DIR, 'informe_modelos_sensoriales.txt')}")
        print("\n" + "="*60)
        
        return models, results, importance_dict
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== EJEMPLO DE USO ====================

def ejemplo_prediccion():
    """
    Ejemplo de cómo usar los modelos para predecir una nueva formulación
    """
    print("\n" + "="*60)
    print("🔮 EJEMPLO DE PREDICCIÓN")
    print("="*60)
    
    # Nueva formulación (ejemplo)
    nueva_formulacion = {
        'azucar_g': 5.0,
        'jarabe_glucosa_g': 3.0,
        'agua_g': 4.0,
        'gelatina_g': 0.8,
        'goma_xantica_g': 0.3,
        'acido_citrico_g': 0.4,
        'stevia_g': 0.5,
        'esencia_g': 0.2,
        'sorbitol_g': 2.0,
        'fibra_g': 1.2,
        'WHC': 4.2,
        'SC': 3.5,
        'OHC': 1.5,
        'FRAP': 52.3,
        'DPPH': 44.1,
        'Polifenoles_totales': 92.7,
        'ratio_gelatina_goma': 2.67,
        'ratio_azucar_edulcorante': 2.0,
        'peso_total_g': 17.4,
        'porcentaje_fibra': 6.9,
        'proporcion_gelatina': 0.73,
        'edulcorantes_totales_g': 2.5,
        'dulzor_total_g': 7.5,
        'proporcion_agua': 0.23,
        'ratio_fibra_agua': 0.3,
        'solidos_totales_g': 13.4,
        'humedad_estimada': 23.0,
        'fibra_Batata_Cáscara': 0,
        'fibra_Batata_Cáscara_3%': 1,
        'fibra_Batata_Cáscara_5%': 0,
        'fibra_Sin_Fibra': 0
    }
    
    try:
        predicciones = predict_sensorial(nueva_formulacion)
        
        print("\n📊 Predicciones para la nueva formulación:")
        print("-"*40)
        for attr, value in predicciones.items():
            print(f"  {attr.upper()}: {value:.3f}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    models, results, importance = main()
    
    if models is not None:
        # Mostrar ejemplo de predicción
        ejemplo_prediccion()
        
        print("\n💡 Los modelos están listos para usar en el Dashboard:")
        print(f"   - Cargar con: pickle.load(open('{os.path.join(MODELS_DIR, 'modelos_sensoriales_completos.pkl')}', 'rb'))")
        print(f"   - Predecir todos los atributos con: predict_sensorial(formulacion)")
        print(f"   - Ver gráficos en: {PLOTS_DIR}")