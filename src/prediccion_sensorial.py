"""
SCRIPT PARA PREDECIR PERFIL SENSORIAL DE TODAS LAS FORMULACIONES
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Entrena modelos de IA con los datos sensoriales existentes
y predice para todas las formulaciones generadas
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'formulaciones_generadas.xlsx')
SENSORIAL_FILE = os.path.join(DATA_DIR, 'datos_sensoriales.xlsx')
OUTPUT_FILE = os.path.join(DATA_DIR, 'formulaciones_con_sensorial.xlsx')

# Crear carpeta si no existe
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== DATOS SENSORIALES ====================
def get_sensorial_data():
    """
    Define los datos sensoriales de las muestras existentes
    """
    sensorial_data = {
        'codigo_gomita': ['4B', '5I', '5K', '10T', '10W', '10Y', '10X', 'Referencia'],
        'fibra_tipo': ['Mandioca', 'Mandioca', 'Batata', 'Batata', 'Mandioca', 'Mandioca', 'Batata', 'Batata'],
        'brillo': [2.500, 3.580, 2.200, 1.380, 1.714, 2.900, 2.150, 1.760],
        'claridad': [2.014, 3.540, 1.433, 1.420, 2.043, 3.420, 3.183, 1.700],
        'dulzura': [2.071, 3.320, 1.617, 1.320, 2.286, 2.560, 1.683, 1.660],
        'acidez': [2.943, 3.900, 2.050, 1.680, 3.357, 2.920, 3.783, 3.600],
        'adhesividad': [6.914, 7.100, 4.967, 3.400, 3.429, 5.400, 3.583, 2.860],
        'elasticidad': [2.100, 4.160, 2.117, 1.200, 6.286, 6.780, 3.283, 3.240],
        'arenosidad': [5.114, 4.600, 6.233, 5.140, 5.486, 4.800, 5.100, 5.940],
        'dureza': [1.514, 1.840, 1.467, 1.120, 3.229, 3.360, 1.700, 1.440],
        'gomosidad': [2.029, 2.480, 1.850, 1.120, 4.486, 4.280, 1.717, 1.520],
        'masticabilidad': [2.057, 3.000, 1.800, 1.240, 5.500, 3.820, 1.967, 1.660]
    }
    
    df = pd.DataFrame(sensorial_data)
    print(f"📊 Datos sensoriales cargados: {len(df)} muestras")
    return df

# ==================== FUNCIONES DE PREDICCIÓN ====================

def prepare_data(df_formulaciones, df_sensorial):
    """
    Prepara los datos para entrenamiento
    """
    # Seleccionar características relevantes (ingredientes)
    feature_cols = ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
                   'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 
                   'esencia_g', 'sorbitol_g', 'fibra_g']
    
    # Codificar fibra_tipo (categórica)
    le = LabelEncoder()
    df_formulaciones['fibra_tipo_encoded'] = le.fit_transform(df_formulaciones['fibra_tipo'])
    
    # Features completas
    all_features = feature_cols + ['fibra_tipo_encoded']
    
    # Para las muestras sensoriales, necesitamos extraer sus formulaciones correspondientes.
    # El código de muestra es el identificador compartido; los nombres de fibra no coinciden
    # exactamente entre ambos conjuntos, así que no deben usarse para la unión.
    df_sensorial_base = df_sensorial.drop(columns=['fibra_tipo'])
    df_combined = df_formulaciones[df_formulaciones['codigo_gomita'].isin(df_sensorial_base['codigo_gomita'])].copy()

    # Unir con datos sensoriales por código de muestra
    df_combined = df_combined.merge(df_sensorial_base, on='codigo_gomita', how='inner')
    
    print(f"🔬 Muestras para entrenamiento: {len(df_combined)}")
    
    return df_combined, all_features, le

def train_models(df_train, all_features, sensorial_cols):
    """
    Entrena modelos para cada atributo sensorial
    """
    X = df_train[all_features]
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    results = {}
    
    print("\n" + "="*60)
    print("🚀 ENTRENANDO MODELOS PARA CADA ATRIBUTO SENSORIAL")
    print("="*60)
    
    for col in sensorial_cols:
        y = df_train[col]
        
        # División entrenamiento/prueba (como tenemos pocos datos, usamos 80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Entrenar Random Forest (bueno para datos pequeños)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)**0.5
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
        
        models[col] = {
            'model': model,
            'scaler': scaler,
            'feature_names': all_features
        }
        
        results[col] = {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'CV_R²_mean': cv_scores.mean(),
            'CV_R²_std': cv_scores.std()
        }
        
        print(f"\n📈 {col.upper()}:")
        print(f"   R²: {r2:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return models, results

def predict_all_formulations(df_formulaciones, models, le):
    """
    Predice todos los atributos sensoriales para todas las formulaciones
    """
    print("\n" + "="*60)
    print("🔮 PREDICIENDO PERFIL SENSORIAL PARA TODAS LAS FORMULACIONES")
    print("="*60)
    
    # Preparar features para predicción
    feature_cols = ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
                   'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 
                   'esencia_g', 'sorbitol_g', 'fibra_g']
    
    # Codificar fibra_tipo
    df_pred = df_formulaciones.copy()
    df_pred['fibra_tipo_encoded'] = le.transform(df_pred['fibra_tipo'])
    
    # Seleccionar features
    all_features = feature_cols + ['fibra_tipo_encoded']
    X_pred = df_pred[all_features]
    
    # Hacer predicciones para cada atributo
    sensorial_cols = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                     'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
    
    for col in sensorial_cols:
        if col in models:
            # Escalar con el scaler del modelo
            model_info = models[col]
            X_scaled = model_info['scaler'].transform(X_pred)
            
            # Predecir
            predictions = model_info['model'].predict(X_scaled)
            df_pred[col + '_pred'] = predictions
    
    # Renombrar columnas de predicción
    for col in sensorial_cols:
        if col + '_pred' in df_pred.columns:
            df_pred[col] = df_pred[col + '_pred']
            df_pred = df_pred.drop(columns=[col + '_pred'])
    
    print(f"✅ Predicciones generadas para {len(df_pred)} formulaciones")
    return df_pred

def add_confidence_intervals(df_pred, models):
    """
    Agrega intervalos de confianza a las predicciones
    """
    print("\n📊 Agregando intervalos de confianza...")
    
    # Calcular error estándar para cada modelo
    for col, model_info in models.items():
        if col in df_pred.columns:
            # Estimar desviación estándar de las predicciones
            # (usando la variabilidad de los árboles en Random Forest)
            model = model_info['model']
            
            # Obtener predicciones de cada árbol
            tree_preds = np.array([tree.predict(model_info['scaler'].transform(
                df_pred[model_info['feature_names']]
            )) for tree in model.estimators_])
            
            # Calcular desviación estándar
            std_pred = tree_preds.std(axis=0)
            
            # Agregar intervalo de confianza (95%)
            df_pred[col + '_lower'] = df_pred[col] - 1.96 * std_pred
            df_pred[col + '_upper'] = df_pred[col] + 1.96 * std_pred
            
            # Clip a rango 1-7 (escala Likert)
            for bound in [col, col + '_lower', col + '_upper']:
                df_pred[bound] = df_pred[bound].clip(1, 7)
    
    return df_pred

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def generate_summary_stats(df_pred):
    """
    Genera estadísticas resumen de las predicciones
    """
    sensorial_cols = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                     'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DE LAS PREDICCIONES")
    print("="*60)
    
    stats = {}
    for col in sensorial_cols:
        if col in df_pred.columns:
            stats[col] = {
                'mean': df_pred[col].mean(),
                'std': df_pred[col].std(),
                'min': df_pred[col].min(),
                'max': df_pred[col].max(),
                'q25': df_pred[col].quantile(0.25),
                'q75': df_pred[col].quantile(0.75)
            }
            print(f"\n{col.upper()}:")
            print(f"   Media: {stats[col]['mean']:.3f}")
            print(f"   Std: {stats[col]['std']:.3f}")
            print(f"   Rango: {stats[col]['min']:.3f} - {stats[col]['max']:.3f}")
    
    return stats

def highlight_extreme_formulations(df_pred):
    """
    Identifica formulaciones con valores extremos
    """
    sensorial_cols = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                     'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
    
    print("\n" + "="*60)
    print("🎯 FORMULACIONES DESTACADAS")
    print("="*60)
    
    # Formulación con mayor dulzura
    max_dulzura = df_pred.loc[df_pred['dulzura'].idxmax()]
    print(f"\n🍬 Mayor dulzura: Código {max_dulzura['codigo_gomita']}")
    print(f"   Dulzura: {max_dulzura['dulzura']:.3f}")
    print(f"   Fibra: {max_dulzura['fibra_tipo']}")
    
    # Formulación con mayor dureza
    max_dureza = df_pred.loc[df_pred['dureza'].idxmax()]
    print(f"\n💪 Mayor dureza: Código {max_dureza['codigo_gomita']}")
    print(f"   Dureza: {max_dureza['dureza']:.3f}")
    print(f"   Fibra: {max_dureza['fibra_tipo']}")
    
    # Formulación más elástica
    max_elasticidad = df_pred.loc[df_pred['elasticidad'].idxmax()]
    print(f"\n🔮 Más elástica: Código {max_elasticidad['codigo_gomita']}")
    print(f"   Elasticidad: {max_elasticidad['elasticidad']:.3f}")
    print(f"   Fibra: {max_elasticidad['fibra_tipo']}")
    
    # Formulación menos adhesiva
    min_adhesividad = df_pred.loc[df_pred['adhesividad'].idxmin()]
    print(f"\n✨ Menos adhesiva: Código {min_adhesividad['codigo_gomita']}")
    print(f"   Adhesividad: {min_adhesividad['adhesividad']:.3f}")
    print(f"   Fibra: {min_adhesividad['fibra_tipo']}")

# ==================== EXPORTACIÓN ====================

def export_data(df_pred):
    """
    Exporta los datos con predicciones
    """
    print("\n💾 Exportando datos...")
    
    # Separar originales y nuevas
    original_codes = ['Referencia', '4B', '5I', '5K', '10T', '10W', '10X', '10Y']
    df_original = df_pred[df_pred['codigo_gomita'].isin(original_codes)]
    df_new = df_pred[~df_pred['codigo_gomita'].isin(original_codes)]
    
    # Exportar archivo completo
    df_pred.to_excel(OUTPUT_FILE, index=False)
    print(f"   ✅ Archivo completo guardado en: {OUTPUT_FILE}")
    
    # Exportar solo originales (para verificar)
    original_file = os.path.join(DATA_DIR, 'originales_con_sensorial.xlsx')
    df_original.to_excel(original_file, index=False)
    print(f"   ✅ Originales guardados en: {original_file}")
    
    # Exportar solo nuevas predicciones
    new_file = os.path.join(DATA_DIR, 'nuevas_predicciones.xlsx')
    df_new.to_excel(new_file, index=False)
    print(f"   ✅ Nuevas predicciones guardadas en: {new_file}")
    
    # Guardar estadísticas
    stats_file = os.path.join(DATA_DIR, 'estadisticas_predicciones.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ESTADÍSTICAS DE PREDICCIONES SENSORIALES\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total formulaciones: {len(df_pred)}\n")
        f.write(f"Originales con datos reales: {len(df_original)}\n")
        f.write(f"Nuevas predicciones: {len(df_new)}\n\n")
        
        sensorial_cols = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                         'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
        
        f.write("Estadísticas de predicciones:\n")
        for col in sensorial_cols:
            if col in df_new.columns:
                f.write(f"\n{col.upper()}:\n")
                f.write(f"  Media: {df_new[col].mean():.3f}\n")
                f.write(f"  Std: {df_new[col].std():.3f}\n")
                f.write(f"  Rango: {df_new[col].min():.3f} - {df_new[col].max():.3f}\n")
    
    print(f"   ✅ Estadísticas guardadas en: {stats_file}")
    
    return df_original, df_new

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal
    """
    print("\n" + "="*60)
    print("🧪 PREDICCIÓN DE PERFIL SENSORIAL CON IA")
    print("="*60 + "\n")
    
    try:
        # 1. Cargar datos
        print("📂 Cargando datos...")
        df_formulaciones = pd.read_excel(INPUT_FILE)
        print(f"   ✅ {len(df_formulaciones)} formulaciones cargadas")
        
        df_sensorial = get_sensorial_data()
        
        # 2. Preparar datos para entrenamiento
        df_train, all_features, le = prepare_data(df_formulaciones, df_sensorial)
        
        # 3. Entrenar modelos
        sensorial_cols = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                         'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
        
        models, results = train_models(df_train, all_features, sensorial_cols)
        
        # 4. Predecir para todas las formulaciones
        df_pred = predict_all_formulations(df_formulaciones, models, le)
        
        # 5. Agregar intervalos de confianza
        df_pred = add_confidence_intervals(df_pred, models)
        
        # 6. Generar estadísticas
        stats = generate_summary_stats(df_pred)
        highlight_extreme_formulations(df_pred)
        
        # 7. Exportar resultados
        df_original, df_new = export_data(df_pred)
        
        # 8. Mostrar resumen final
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO CON ÉXITO")
        print("="*60)
        print(f"\n📊 Resumen final:")
        print(f"   - Total formulaciones: {len(df_pred)}")
        print(f"   - Con datos reales: {len(df_original)}")
        print(f"   - Con predicciones: {len(df_new)}")
        print(f"\n📁 Archivos generados:")
        print(f"   - formulaciones_con_sensorial.xlsx (todas)")
        print(f"   - originales_con_sensorial.xlsx (solo reales)")
        print(f"   - nuevas_predicciones.xlsx (solo predicciones)")
        
        return df_pred, df_original, df_new
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    df_pred, df_original, df_new = main()
    
    if df_pred is not None:
        print("\n💡 Datos listos para usar:")
        print(f"   - Abre 'formulaciones_con_sensorial.xlsx' para ver todas")
        print(f"   - Las columnas agregadas son: brillo, claridad, dulzura, acidez,")
        print(f"     adhesividad, elasticidad, arenosidad, dureza, gomosidad, masticabilidad")
        print(f"   - También incluye intervalos de confianza (lower/upper)")