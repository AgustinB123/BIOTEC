"""
SCRIPT PARA CREAR EL DATASET MAESTRO
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Consolida formulaciones, perfil sensorial, propiedades de fibra
y aplica Feature Engineering + One-Hot Encoding
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Archivos de entrada
FORMULACIONES_FILE = os.path.join(DATA_DIR, 'formulaciones_con_sensorial.xlsx')
PROPIEDADES_FIBRA_FILE = os.path.join(DATA_DIR, 'propiedades_fibra.xlsx')

# Archivos de salida
OUTPUT_EXCEL = os.path.join(DATA_DIR, 'dataset_maestro_limpio.xlsx')
OUTPUT_CSV = os.path.join(DATA_DIR, 'dataset_maestro_limpio.csv')
OUTPUT_JSON = os.path.join(DATA_DIR, 'dataset_maestro_limpio.json')

# Crear carpetas si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# ==================== TABLA 1: FORMULACIONES ====================

def load_formulaciones():
    """
    Carga la tabla de formulaciones con perfil sensorial
    """
    print("📂 Cargando Tabla 1: Formulaciones...")
    
    # Cargar archivo
    df = pd.read_excel(FORMULACIONES_FILE)
    
    # Seleccionar columnas relevantes (eliminar intervalos de confianza)
    cols_to_keep = ['codigo_gomita', 'fibra_tipo', 'azucar_g', 'jarabe_glucosa_g', 
                   'agua_g', 'gelatina_g', 'goma_xantica_g', 'acido_citrico_g', 
                   'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g',
                   'brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                   'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
    
    # Filtrar columnas existentes
    available_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[available_cols]
    
    print(f"   ✅ {len(df)} formulaciones cargadas")
    print(f"   📋 Columnas: {', '.join(df.columns.tolist())}")
    return df

# ==================== TABLA 2: PROPIEDADES DE LA FIBRA ====================

def load_propiedades_fibra():
    """
    Carga la tabla de propiedades de la fibra
    """
    print("\n📂 Cargando Tabla 2: Propiedades de Fibra...")
    
    # Datos de propiedades de fibra (según la literatura)
    propiedades_data = {
        'fibra_tipo': ['Sin_Fibra', 'Batata_Cáscara', 'Batata_Cáscara_3%', 'Batata_Cáscara_5%'],
        'WHC': [0, 3.5, 4.2, 4.8],  # Capacidad retención agua (g/g)
        'SC': [0, 2.8, 3.5, 4.0],    # Capacidad hinchamiento (mL/g)
        'OHC': [0, 1.2, 1.5, 1.8],   # Capacidad retención aceite (g/g)
        'FRAP': [0, 45.2, 52.3, 58.7],  # Capacidad antioxidante (mg FeSO4·7H2O/100g)
        'DPPH': [0, 38.5, 44.1, 50.3],  # Capacidad antioxidante (mg Trolox/100g)
        'Polifenoles_totales': [0, 85.3, 92.7, 98.5]  # mg ác. Gal./100g
    }
    
    df = pd.DataFrame(propiedades_data)
    
    # Guardar archivo para referencia
    df.to_excel(PROPIEDADES_FIBRA_FILE, index=False)
    print(f"   ✅ Archivo guardado en: {PROPIEDADES_FIBRA_FILE}")
    print(f"   ✅ {len(df)} tipos de fibra cargados")
    print(f"   📋 Columnas: {', '.join(df.columns.tolist())}")
    return df

# ==================== UNIÓN DE TABLAS ====================

def merge_tables(df_formulaciones, df_propiedades):
    """
    Une las tablas de formulaciones y propiedades de fibra
    """
    print("\n🔗 Uniendo tablas...")
    
    # Unir por 'fibra_tipo'
    df_merged = df_formulaciones.merge(df_propiedades, on='fibra_tipo', how='left')
    
    print(f"   ✅ Dataset combinado: {len(df_merged)} filas, {len(df_merged.columns)} columnas")
    return df_merged

# ==================== FEATURE ENGINEERING ====================

def create_features(df):
    """
    Crea nuevas variables (Feature Engineering)
    ⭐ ELIMINA fibra_g y NO la vuelve a agregar (se agregará después del One-Hot)
    """
    print("\n🧬 Aplicando Feature Engineering...")
    
    df_features = df.copy()
    
    # ⭐ Guardar la columna fibra_g original
    fibra_g_original = df_features['fibra_g'].copy()
    
    # ⭐ ELIMINAR la columna fibra_g original para evitar duplicados
    df_features = df_features.drop(columns=['fibra_g'])
    
    # 1. ratio_gelatina_goma
    df_features['ratio_gelatina_goma'] = df_features['gelatina_g'] / (df_features['goma_xantica_g'] + 1e-9)
    
    # 2. ratio_azucar_edulcorante
    df_features['ratio_azucar_edulcorante'] = df_features['azucar_g'] / (df_features['stevia_g'] + df_features['sorbitol_g'] + 1e-9)
    
    # 3. peso_total_g (usando fibra_g original)
    ingredientes = ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
                   'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 
                   'esencia_g', 'sorbitol_g']
    df_features['peso_total_g'] = df_features[ingredientes].sum(axis=1) + fibra_g_original
    
    # 4. porcentaje_fibra (usando fibra_g original)
    df_features['porcentaje_fibra'] = (fibra_g_original / df_features['peso_total_g']) * 100
    
    # 5. proporcion_gelatina
    df_features['proporcion_gelatina'] = df_features['gelatina_g'] / (df_features['gelatina_g'] + df_features['goma_xantica_g'] + 1e-9)
    
    # 6. edulcorantes_totales_g
    df_features['edulcorantes_totales_g'] = df_features['stevia_g'] + df_features['sorbitol_g']
    
    # 7. dulzor_total_g
    df_features['dulzor_total_g'] = df_features['azucar_g'] + df_features['stevia_g'] + df_features['sorbitol_g']
    
    # 8. proporcion_agua
    df_features['proporcion_agua'] = df_features['agua_g'] / df_features['peso_total_g']
    
    # 9. ratio_fibra_agua (usando fibra_g original)
    df_features['ratio_fibra_agua'] = fibra_g_original / (df_features['agua_g'] + 1e-9)
    
    # 10. solidos_totales_g (usando fibra_g original)
    solidos = ['azucar_g', 'jarabe_glucosa_g', 'gelatina_g', 'goma_xantica_g', 
               'acido_citrico_g', 'stevia_g', 'esencia_g', 'sorbitol_g']
    df_features['solidos_totales_g'] = df_features[solidos].sum(axis=1) + fibra_g_original
    
    # 11. humedad_estimada
    df_features['humedad_estimada'] = (df_features['agua_g'] / df_features['peso_total_g']) * 100
    
    # ⭐ NO volver a agregar fibra_g aquí (se agregará después del One-Hot)
    
    print(f"   ✅ Creadas {len(df_features.columns) - len(df.columns) + 1} nuevas variables")
    print(f"   📋 Total columnas: {len(df_features.columns)}")
    
    return df_features, fibra_g_original

# ==================== ONE-HOT ENCODING ====================

def apply_one_hot_encoding(df):
    """
    Aplica One-Hot Encoding a la variable fibra_tipo
    ⭐ CORREGIDO: NO incluye fibra_g (ya fue eliminada)
    """
    print("\n🔢 Aplicando One-Hot Encoding...")
    
    df_encoded = df.copy()
    
    # Obtener dummies para fibra_tipo
    dummies = pd.get_dummies(df_encoded['fibra_tipo'], prefix='fibra')
    
    # Unir con el DataFrame original
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    print(f"   ✅ Creadas {len(dummies.columns)} columnas dummy")
    print(f"   📋 Columnas dummy: {', '.join(dummies.columns.tolist())}")
    
    return df_encoded

# ==================== LIMPIEZA Y VALIDACIÓN ====================

def clean_dataset(df, fibra_g_original):
    """
    Limpia y valida el dataset maestro
    ⭐ AGREGA fibra_g después del One-Hot Encoding
    """
    print("\n🧹 Limpiando dataset...")
    
    df_clean = df.copy()
    
    # 1. Reemplazar valores infinitos por NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # 2. Manejar ratios extremos
    mask_goma_cero = df_clean['goma_xantica_g'] == 0
    df_clean.loc[mask_goma_cero, 'ratio_gelatina_goma'] = 9999
    
    mask_edulcorante_cero = (df_clean['stevia_g'] + df_clean['sorbitol_g']) == 0
    df_clean.loc[mask_edulcorante_cero, 'ratio_azucar_edulcorante'] = 9999
    
    # ⭐ 3. ELIMINAR TODAS LAS COLUMNAS DUPLICADAS
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep='first')]
    
    # ⭐ 4. AGREGAR fibra_g DESPUÉS del One-Hot Encoding
    df_clean['fibra_g'] = fibra_g_original
    
    # 5. Ordenar columnas
    column_order = [
        'codigo_gomita', 'fibra_tipo',
        'azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
        'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 'esencia_g', 
        'sorbitol_g', 'fibra_g',
        'brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
        'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad',
        'WHC', 'SC', 'OHC', 'FRAP', 'DPPH', 'Polifenoles_totales',
        'ratio_gelatina_goma', 'ratio_azucar_edulcorante', 'peso_total_g',
        'porcentaje_fibra', 'proporcion_gelatina', 'edulcorantes_totales_g',
        'dulzor_total_g', 'proporcion_agua', 'ratio_fibra_agua',
        'solidos_totales_g', 'humedad_estimada',
    ]
    
    # Agregar columnas dummy (excluyendo fibra_g)
    dummy_cols = [col for col in df_clean.columns if col.startswith('fibra_') and col != 'fibra_tipo' and col != 'fibra_g']
    column_order.extend(dummy_cols)
    
    # Seleccionar solo columnas existentes
    existing_cols = [col for col in column_order if col in df_clean.columns]
    df_clean = df_clean[existing_cols]
    
    # ⭐ 6. VERIFICACIÓN FINAL
    print(f"   ✅ Dataset limpio: {len(df_clean)} filas, {len(df_clean.columns)} columnas")
    print(f"   📋 Columnas finales: {', '.join(df_clean.columns.tolist())}")
    
    # ⭐ 7. CONTAR 'fibra_g' (debe ser 1)
    fibra_g_count_final = sum(df_clean.columns == 'fibra_g')
    if fibra_g_count_final == 1:
        print(f"   ✅ 'fibra_g' aparece 1 vez (correcto)")
    else:
        print(f"   ⚠️ 'fibra_g' aparece {fibra_g_count_final} veces (¡ERROR!)")
    
    return df_clean

# ==================== VISUALIZACIÓN ====================

def show_summary(df):
    """
    Muestra un resumen del dataset maestro
    """
    print("\n" + "="*60)
    print("📊 RESUMEN DEL DATASET MAESTRO")
    print("="*60)
    print(f"Total de filas: {len(df)}")
    print(f"Total de columnas: {len(df.columns)}")
    
    print("\nDistribución por tipo de fibra:")
    if 'fibra_tipo' in df.columns:
        fibra_counts = df['fibra_tipo'].value_counts()
        for fibra, count in fibra_counts.items():
            print(f"  {fibra}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  ⚠️ Columna fibra_tipo no encontrada")
    
    print("\nColumnas por grupo:")
    grupos = {
        'Identificación': ['codigo_gomita', 'fibra_tipo'],
        'Ingredientes': ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
                        'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 
                        'esencia_g', 'sorbitol_g', 'fibra_g'],
        'Perfil Sensorial': ['brillo', 'claridad', 'dulzura', 'acidez', 
                           'adhesividad', 'elasticidad', 'arenosidad', 
                           'dureza', 'gomosidad', 'masticabilidad'],
        'Propiedades Fibra': ['WHC', 'SC', 'OHC', 'FRAP', 'DPPH', 'Polifenoles_totales'],
        'Feature Engineering': ['ratio_gelatina_goma', 'ratio_azucar_edulcorante', 
                              'peso_total_g', 'porcentaje_fibra', 'proporcion_gelatina',
                              'edulcorantes_totales_g', 'dulzor_total_g', 
                              'proporcion_agua', 'ratio_fibra_agua', 
                              'solidos_totales_g', 'humedad_estimada'],
        'One-Hot Encoding': [col for col in df.columns if col.startswith('fibra_') and col != 'fibra_tipo' and col != 'fibra_g']
    }
    
    for grupo, columnas in grupos.items():
        existing = [col for col in columnas if col in df.columns]
        print(f"  {grupo}: {len(existing)} columnas")
    
    print("\n" + "="*60)
    
    # Mostrar primeras filas (solo columnas principales)
    print("\n📋 Primeras 3 filas del dataset (columnas principales):")
    main_cols = ['codigo_gomita', 'fibra_tipo', 'azucar_g', 'fibra_g', 
                 'porcentaje_fibra', 'brillo', 'dulzura', 'adhesividad']
    existing_main = [col for col in main_cols if col in df.columns]
    if existing_main:
        print(df[existing_main].head(3).to_string())
    else:
        print("  ⚠️ No se encontraron columnas principales")
    
    print("\n" + "="*60)

# ==================== EXPORTACIÓN ====================

def export_dataset(df):
    """
    Exporta el dataset maestro en múltiples formatos
    """
    print("\n💾 Exportando Dataset Maestro...")
    
    # Exportar a Excel
    try:
        df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
        print(f"   ✅ Excel guardado en: {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"   ⚠️ Error al guardar Excel: {e}")
    
    # Exportar a CSV
    try:
        df.to_csv(OUTPUT_CSV, index=False, decimal=',')
        print(f"   ✅ CSV guardado en: {OUTPUT_CSV}")
    except Exception as e:
        print(f"   ⚠️ Error al guardar CSV: {e}")
    
    # Exportar a JSON
    try:
        df_json = df.copy()
        for col in df_json.columns:
            if df_json[col].dtype == 'object':
                df_json[col] = df_json[col].astype(str)
        df_json.to_json(OUTPUT_JSON, orient='records', indent=2, force_ascii=False)
        print(f"   ✅ JSON guardado en: {OUTPUT_JSON}")
    except Exception as e:
        print(f"   ⚠️ Error al guardar JSON: {e}")
    
    # Guardar documentación
    doc_file = os.path.join(DATA_DIR, 'dataset_maestro_documentacion.txt')
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DOCUMENTACIÓN DEL DATASET MAESTRO\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Fecha de creación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de filas: {len(df)}\n")
        f.write(f"Total de columnas: {len(df.columns)}\n\n")
        
        f.write("ESTRUCTURA DEL DATASET:\n")
        f.write("-"*40 + "\n\n")
        
        grupos = {
            'Identificación': ['codigo_gomita', 'fibra_tipo'],
            'Ingredientes': ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
                           'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 
                           'esencia_g', 'sorbitol_g', 'fibra_g'],
            'Perfil Sensorial': ['brillo', 'claridad', 'dulzura', 'acidez', 
                               'adhesividad', 'elasticidad', 'arenosidad', 
                               'dureza', 'gomosidad', 'masticabilidad'],
            'Propiedades Fibra': ['WHC', 'SC', 'OHC', 'FRAP', 'DPPH', 'Polifenoles_totales'],
            'Feature Engineering': ['ratio_gelatina_goma', 'ratio_azucar_edulcorante', 
                                  'peso_total_g', 'porcentaje_fibra', 'proporcion_gelatina',
                                  'edulcorantes_totales_g', 'dulzor_total_g', 
                                  'proporcion_agua', 'ratio_fibra_agua', 
                                  'solidos_totales_g', 'humedad_estimada'],
            'One-Hot Encoding': [col for col in df.columns if col.startswith('fibra_') and col != 'fibra_tipo' and col != 'fibra_g']
        }
        
        for grupo, columnas in grupos.items():
            f.write(f"\nGrupo {grupo}:\n")
            for col in columnas:
                if col in df.columns:
                    dtype = df[col].dtypes
                    f.write(f"  - {col} ({dtype})\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ESTADÍSTICAS DESCRIPTIVAS\n")
        f.write("="*60 + "\n\n")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            f.write(df[numeric_cols].describe().to_string())
        else:
            f.write("No hay columnas numéricas para mostrar estadísticas.\n")
        
        f.write("\n\n" + "="*60 + "\n")
        f.write("DISTRIBUCIÓN DE FIBRA\n")
        f.write("="*60 + "\n\n")
        
        if 'fibra_tipo' in df.columns:
            fibra_counts = df['fibra_tipo'].value_counts()
            for fibra, count in fibra_counts.items():
                f.write(f"  {fibra}: {count} filas ({count/len(df)*100:.1f}%)\n")
    
    print(f"   ✅ Documentación guardada en: {doc_file}")

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal
    """
    print("\n" + "="*60)
    print("🚀 CREANDO DATASET MAESTRO")
    print("="*60 + "\n")
    
    try:
        # 1. Cargar datos
        df_formulaciones = load_formulaciones()
        df_propiedades = load_propiedades_fibra()
        
        # 2. Unir tablas
        df_merged = merge_tables(df_formulaciones, df_propiedades)
        
        # 3. Feature Engineering (devuelve df_features y fibra_g_original)
        df_features, fibra_g_original = create_features(df_merged)
        
        # 4. One-Hot Encoding (NO incluye fibra_g)
        df_encoded = apply_one_hot_encoding(df_features)
        
        # 5. Limpiar dataset y agregar fibra_g
        df_clean = clean_dataset(df_encoded, fibra_g_original)
        
        # 6. Mostrar resumen
        show_summary(df_clean)
        
        # 7. Exportar
        export_dataset(df_clean)
        
        print("\n✅ ¡DATASET MAESTRO CREADO CON ÉXITO!")
        print(f"   Archivo: {OUTPUT_EXCEL}")
        print(f"   Filas: {len(df_clean)}")
        print(f"   Columnas: {len(df_clean.columns)}")
        
        return df_clean
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    df_maestro = main()
    
    if df_maestro is not None:
        print("\n💡 Dataset Maestro listo para usar:")
        print(f"   - {len(df_maestro)} formulaciones")
        print(f"   - {len(df_maestro.columns)} variables")
        print(f"   - Incluye: ingredientes, perfil sensorial, propiedades de fibra")
        print(f"   - Incluye: {len([col for col in df_maestro.columns if col.startswith('fibra_') and col != 'fibra_tipo' and col != 'fibra_g'])} variables dummy")
        print(f"   - Incluye: 11 variables de feature engineering")
        
        # Verificación final
        duplicadas = df_maestro.columns[df_maestro.columns.duplicated()]
        if len(duplicadas) > 0:
            print(f"\n⚠️ ADVERTENCIA: Aún hay columnas duplicadas: {duplicadas.tolist()}")
        else:
            print(f"\n✅ VERIFICACIÓN: No hay columnas duplicadas")
        
        fibra_g_count = sum(df_maestro.columns == 'fibra_g')
        if fibra_g_count == 1:
            print(f"✅ 'fibra_g' aparece 1 vez (correcto)")
        else:
            print(f"⚠️ ERROR: 'fibra_g' aparece {fibra_g_count} veces")