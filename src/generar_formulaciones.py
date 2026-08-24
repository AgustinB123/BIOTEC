"""
Script para generar formulaciones de gomitas usando diseño de experimentos
Mantiene la estructura exacta del archivo original
CORREGIDO: Solo la Referencia tiene Sin_Fibra
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
"""

import os
import numpy as np
import pandas as pd
from itertools import product
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_FILE = os.path.join(DATA_DIR, 'formulaciones_generadas.xlsx')
OUTPUT_CSV = os.path.join(DATA_DIR, 'formulaciones_generadas.csv')

# Crear carpeta si no existe
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== FORMULACIONES ORIGINALES ====================
def get_original_formulations():
    """
    Retorna las formulaciones originales exactas
    """
    original_data = {
        'codigo_gomita': ['Referencia', '4B', '5I', '5K', '10T', '10W', '10X', '10Y'],
        'fibra_tipo': ['Sin_Fibra', 'Batata_Cáscara', 'Batata_Cáscara', 'Batata_Cáscara', 
                      'Batata_Cáscara_3%', 'Batata_Cáscara_3%', 'Batata_Cáscara_5%', 'Batata_Cáscara_5%'],
        'azucar_g': [11.4, 0.2, 0.4, 0.4, 5.7, 4.56, 4.56, 5.7],
        'jarabe_glucosa_g': [7.8, 1.1, 0.3, 0.3, 7.8, 7.8, 7.8, 7.8],
        'agua_g': [8.922, 1.0, 6.0, 6.0, 8.922, 8.922, 8.922, 8.922],
        'gelatina_g': [1.14, 0, 1.2, 0.4, 1.14, 1.14, 1.14, 1.14],
        'goma_xantica_g': [0, 0.4, 0.4, 1.2, 0, 0, 0, 0],
        'acido_citrico_g': [0.522, 0.1, 0.2, 0.2, 0.522, 0.522, 0.522, 0.522],
        'stevia_g': [0, 0.1, 0.1, 0.1, 1.14, 1.368, 1.368, 1.14],
        'esencia_g': [0.09, 0.1, 0.6, 0.6, 0.09, 0.09, 0.09, 0.09],
        'sorbitol_g': [0, 0, 0, 0, 4.56, 5.472, 5.472, 4.56],
        'fibra_g': [0, 1.25, 2.5, 2.6, 0.9, 0.9, 1.5, 1.5]
    }
    
    return pd.DataFrame(original_data)

# ==================== GENERACIÓN DE NUEVAS FORMULACIONES ====================

def generate_new_formulations(n_formulations=100):
    """
    Genera nuevas formulaciones manteniendo la estructura original
    CORREGIDO: TODAS las nuevas tienen fibra (nunca Sin_Fibra)
    """
    print(f"🔬 Generando {n_formulations} nuevas formulaciones...")
    
    # Definir rangos para cada ingrediente basados en los datos originales
    ranges = {
        'azucar_g': (0.1, 11.4),
        'jarabe_glucosa_g': (0.1, 7.8),
        'agua_g': (0.5, 8.922),
        'gelatina_g': (0, 1.2),
        'goma_xantica_g': (0, 1.2),
        'acido_citrico_g': (0.05, 0.6),
        'stevia_g': (0, 1.4),
        'esencia_g': (0.05, 0.6),
        'sorbitol_g': (0, 5.5),
        'fibra_g': (0.1, 2.6)  # <--- AHORA MÍNIMO 0.1 (nunca 0)
    }
    
    # Tipos de fibra disponibles (TODAS con fibra)
    fibra_tipos = ['Batata_Cáscara', 'Batata_Cáscara_3%', 'Batata_Cáscara_5%']
    
    new_formulations = []
    
    for i in range(n_formulations):
        # Generar ID numérico (1 a n)
        codigo = str(i + 1)
        
        # Seleccionar tipo de fibra aleatoriamente
        fibra_tipo = random.choice(fibra_tipos)
        
        # Asegurar que fibra_g > 0
        fibra_g = round(random.uniform(0.1, 2.6), 2)
        
        # Generar valores aleatorios dentro de los rangos
        formulacion = {
            'codigo_gomita': codigo,
            'fibra_tipo': fibra_tipo,
            'azucar_g': round(random.uniform(ranges['azucar_g'][0], ranges['azucar_g'][1]), 2),
            'jarabe_glucosa_g': round(random.uniform(ranges['jarabe_glucosa_g'][0], ranges['jarabe_glucosa_g'][1]), 2),
            'agua_g': round(random.uniform(ranges['agua_g'][0], ranges['agua_g'][1]), 3),
            'gelatina_g': round(random.uniform(ranges['gelatina_g'][0], ranges['gelatina_g'][1]), 2),
            'goma_xantica_g': round(random.uniform(ranges['goma_xantica_g'][0], ranges['goma_xantica_g'][1]), 2),
            'acido_citrico_g': round(random.uniform(ranges['acido_citrico_g'][0], ranges['acido_citrico_g'][1]), 3),
            'stevia_g': round(random.uniform(ranges['stevia_g'][0], ranges['stevia_g'][1]), 3),
            'esencia_g': round(random.uniform(ranges['esencia_g'][0], ranges['esencia_g'][1]), 2),
            'sorbitol_g': round(random.uniform(ranges['sorbitol_g'][0], ranges['sorbitol_g'][1]), 3),
            'fibra_g': fibra_g
        }
        
        new_formulations.append(formulacion)
    
    df_new = pd.DataFrame(new_formulations)
    print(f"   ✅ Generadas {len(df_new)} nuevas formulaciones (TODAS con fibra)")
    return df_new

def generate_design_experiment(n_formulations=80):
    """
    Genera formulaciones usando diseño de experimentos (factorial parcial)
    CORREGIDO: TODAS las nuevas tienen fibra (nunca Sin_Fibra)
    """
    print(f"🔬 Generando {n_formulations} formulaciones con diseño experimental...")
    
    # Definir niveles para cada ingrediente
    levels = {
        'azucar_g': [0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.4],
        'jarabe_glucosa_g': [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.8],
        'fibra_g': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 2.6]  # <--- AHORA MÍNIMO 0.1
    }
    
    fibra_tipos = ['Batata_Cáscara', 'Batata_Cáscara_3%', 'Batata_Cáscara_5%']
    
    # Generar combinaciones de diseño
    design_points = []
    
    # Tomar muestras representativas
    azucar_sample = random.sample(levels['azucar_g'], min(4, len(levels['azucar_g'])))
    jarabe_sample = random.sample(levels['jarabe_glucosa_g'], min(4, len(levels['jarabe_glucosa_g'])))
    fibra_sample = random.sample(levels['fibra_g'], min(4, len(levels['fibra_g'])))
    
    # Crear combinaciones
    for azucar in azucar_sample:
        for jarabe in jarabe_sample:
            for fibra in fibra_sample:
                # TODAS tienen fibra > 0, así que siempre elegir un tipo con fibra
                fibra_tipo = random.choice(fibra_tipos)
                
                design_points.append({
                    'azucar_g': azucar,
                    'jarabe_glucosa_g': jarabe,
                    'fibra_g': fibra,
                    'fibra_tipo': fibra_tipo
                })
    
    # Limitar al número deseado
    design_points = random.sample(design_points, min(n_formulations, len(design_points)))
    
    # Completar con ingredientes fijos o variables
    new_formulations = []
    for i, point in enumerate(design_points):
        formulacion = {
            'codigo_gomita': str(100 + i + 1),  # IDs 101 en adelante
            'fibra_tipo': point['fibra_tipo'],
            'azucar_g': point['azucar_g'],
            'jarabe_glucosa_g': point['jarabe_glucosa_g'],
            'agua_g': round(random.uniform(0.5, 8.922), 3),
            'gelatina_g': round(random.uniform(0, 1.2), 2),
            'goma_xantica_g': round(random.uniform(0, 1.2), 2),
            'acido_citrico_g': round(random.uniform(0.05, 0.6), 3),
            'stevia_g': round(random.uniform(0, 1.4), 3),
            'esencia_g': round(random.uniform(0.05, 0.6), 2),
            'sorbitol_g': round(random.uniform(0, 5.5), 3),
            'fibra_g': point['fibra_g']
        }
        new_formulations.append(formulacion)
    
    df_design = pd.DataFrame(new_formulations)
    print(f"   ✅ Generadas {len(df_design)} formulaciones con diseño experimental (TODAS con fibra)")
    return df_design

def generate_specialized_formulations():
    """
    Genera formulaciones especializadas por tipo de fibra
    CORREGIDO: TODAS las nuevas tienen fibra (nunca Sin_Fibra)
    """
    print("🔬 Generando formulaciones especializadas por tipo de fibra...")
    
    specialized = []
    
    # Configuraciones específicas para cada tipo de fibra
    fibra_configs = [
        ('Batata_Cáscara', {'fibra_g': (0.5, 2.6), 'azucar_g': (0.1, 2.0), 'gelatina_g': (0.2, 1.2)}),
        ('Batata_Cáscara_3%', {'fibra_g': (0.5, 1.5), 'azucar_g': (2.0, 8.0), 'gelatina_g': (0.2, 1.2)}),
        ('Batata_Cáscara_5%', {'fibra_g': (0.5, 1.5), 'azucar_g': (2.0, 8.0), 'gelatina_g': (0.2, 1.2)})
    ]
    
    for fibra_tipo, config in fibra_configs:
        for _ in range(20):
            # Asegurar que fibra_g > 0
            fibra_g = round(random.uniform(config['fibra_g'][0], config['fibra_g'][1]), 2)
            
            formulacion = {
                'codigo_gomita': str(200 + len(specialized) + 1),
                'fibra_tipo': fibra_tipo,
                'azucar_g': round(random.uniform(config['azucar_g'][0], config['azucar_g'][1]), 2),
                'jarabe_glucosa_g': round(random.uniform(0.3, 7.8), 2),
                'agua_g': round(random.uniform(1.0, 8.922), 3),
                'gelatina_g': round(random.uniform(config['gelatina_g'][0], config['gelatina_g'][1]), 2),
                'goma_xantica_g': round(random.uniform(0, 1.2), 2),
                'acido_citrico_g': round(random.uniform(0.05, 0.6), 3),
                'stevia_g': round(random.uniform(0, 1.4), 3),
                'esencia_g': round(random.uniform(0.05, 0.6), 2),
                'sorbitol_g': round(random.uniform(0, 5.5), 3),
                'fibra_g': fibra_g
            }
            specialized.append(formulacion)
    
    df_specialized = pd.DataFrame(specialized)
    print(f"   ✅ Generadas {len(df_specialized)} formulaciones especializadas (TODAS con fibra)")
    return df_specialized

def generate_smart_formulations():
    """
    Genera formulaciones inteligentes combinando diferentes estrategias
    """
    print("\n" + "="*60)
    print("🚀 GENERANDO FORMULACIONES INTELIGENTES PARA GOMITAS")
    print("="*60 + "\n")
    
    # 1. Obtener formulaciones originales (incluye la única Sin_Fibra)
    df_original = get_original_formulations()
    print(f"📋 Formulaciones originales: {len(df_original)} (1 Sin_Fibra + 7 con fibra)")
    
    # 2. Generar formulaciones aleatorias (TODAS con fibra)
    df_random = generate_new_formulations(100)
    
    # 3. Generar formulaciones con diseño experimental (TODAS con fibra)
    df_design = generate_design_experiment(80)
    
    # 4. Generar formulaciones especializadas (TODAS con fibra)
    df_specialized = generate_specialized_formulations()
    
    # 5. Combinar todo
    df_combined = pd.concat([df_original, df_random, df_design, df_specialized], ignore_index=True)
    
    # 6. Limpiar y asegurar estructura
    column_order = ['codigo_gomita', 'fibra_tipo', 'azucar_g', 'jarabe_glucosa_g', 
                   'agua_g', 'gelatina_g', 'goma_xantica_g', 'acido_citrico_g', 
                   'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g']
    
    df_combined = df_combined[column_order]
    
    # 7. ELIMINAR cualquier formulación que no sea Referencia con fibra_g = 0
    # Esto es CRÍTICO: solo la Referencia puede tener fibra_g = 0
    mask_referencia = df_combined['codigo_gomita'] == 'Referencia'
    df_combined = df_combined[(df_combined['fibra_g'] > 0) | mask_referencia]
    
    # 8. Asegurar que solo la Referencia tenga Sin_Fibra
    df_combined.loc[df_combined['codigo_gomita'] != 'Referencia', 'fibra_tipo'] = \
        df_combined.loc[df_combined['codigo_gomita'] != 'Referencia', 'fibra_tipo'].apply(
            lambda x: x if x != 'Sin_Fibra' else random.choice(['Batata_Cáscara', 'Batata_Cáscara_3%', 'Batata_Cáscara_5%'])
        )
    
    # 9. Eliminar duplicados
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['azucar_g', 'jarabe_glucosa_g', 'fibra_g'])
    print(f"   ✅ Eliminados {initial_len - len(df_combined)} duplicados")
    
    # 10. Verificar que solo la Referencia tenga Sin_Fibra
    sin_fibra_count = len(df_combined[df_combined['fibra_tipo'] == 'Sin_Fibra'])
    if sin_fibra_count > 1:
        print(f"   ⚠️ ERROR: Hay {sin_fibra_count} formulaciones con Sin_Fibra. Solo debe haber 1.")
        # Forzar corrección: todas las que no son Referencia y tienen Sin_Fibra, se cambian
        df_combined.loc[(df_combined['codigo_gomita'] != 'Referencia') & 
                       (df_combined['fibra_tipo'] == 'Sin_Fibra'), 'fibra_tipo'] = 'Batata_Cáscara'
    
    print("\n" + "="*60)
    print(f"✅ TOTAL FORMULACIONES GENERADAS: {len(df_combined)}")
    print(f"   - Referencia (Sin_Fibra): 1")
    print(f"   - Con fibra: {len(df_combined) - 1}")
    print("="*60 + "\n")
    
    return df_combined

# ==================== FUNCIONES DE EXPORTACIÓN ====================

def export_data(df):
    """
    Exporta los datos manteniendo el formato original
    """
    print("💾 Exportando datos...")
    
    # Exportar a Excel (formato original)
    try:
        df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"   ✅ Excel guardado en: {OUTPUT_FILE}")
    except Exception as e:
        print(f"   ⚠️ Error al guardar Excel: {e}")
    
    # Exportar a CSV
    try:
        df.to_csv(OUTPUT_CSV, index=False, decimal=',')
        print(f"   ✅ CSV guardado en: {OUTPUT_CSV}")
    except Exception as e:
        print(f"   ⚠️ Error al guardar CSV: {e}")
    
    # Guardar estadísticas
    stats_file = os.path.join(DATA_DIR, 'estadisticas_formulaciones.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ESTADÍSTICAS DE FORMULACIONES GENERADAS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de formulaciones: {len(df)}\n\n")
        
        f.write("Distribución por tipo de fibra:\n")
        fibra_counts = df['fibra_tipo'].value_counts()
        for fibra, count in fibra_counts.items():
            f.write(f"  {fibra}: {count} ({count/len(df)*100:.1f}%)\n")
        
        f.write("\nEstadísticas descriptivas:\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        f.write(df[numeric_cols].describe().to_string())
        
        f.write("\n\nPrimeras 10 formulaciones:\n")
        f.write(df.head(10).to_string())
    
    print(f"   ✅ Estadísticas guardadas en: {stats_file}")

def generate_summary(df):
    """
    Genera un resumen del diseño generado
    """
    print("\n" + "="*60)
    print("📊 RESUMEN DE FORMULACIONES GENERADAS")
    print("="*60)
    print(f"Total de formulaciones: {len(df)}")
    
    # Distribución por tipo de fibra
    print("\nDistribución por tipo de fibra:")
    fibra_counts = df['fibra_tipo'].value_counts()
    for fibra, count in fibra_counts.items():
        print(f"  {fibra}: {count} ({count/len(df)*100:.1f}%)")
    
    # Verificar que solo hay 1 Sin_Fibra
    sin_fibra_count = len(df[df['fibra_tipo'] == 'Sin_Fibra'])
    if sin_fibra_count == 1:
        print(f"\n✅ CORRECTO: Solo 1 formulación Sin_Fibra (la Referencia)")
    else:
        print(f"\n⚠️ ERROR: Hay {sin_fibra_count} formulaciones Sin_Fibra. Debe haber 1.")
    
    # Rangos de variables principales
    print("\nRangos de variables principales:")
    for col in ['azucar_g', 'jarabe_glucosa_g', 'fibra_g']:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f} (media: {df[col].mean():.2f})")
    
    # Separación entre originales y nuevas
    original_codes = ['Referencia', '4B', '5I', '5K', '10T', '10W', '10X', '10Y']
    df_original = df[df['codigo_gomita'].isin(original_codes)]
    df_new = df[~df['codigo_gomita'].isin(original_codes)]
    
    print(f"\nFormulaciones originales: {len(df_original)}")
    print(f"Nuevas formulaciones: {len(df_new)}")
    
    print("\nArchivos generados:")
    print(f"  - Excel: {OUTPUT_FILE}")
    print(f"  - CSV: {OUTPUT_CSV}")
    print("="*60 + "\n")

# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal
    """
    try:
        # Generar formulaciones inteligentes
        df_final = generate_smart_formulations()
        
        # Mostrar resumen
        generate_summary(df_final)
        
        # Exportar datos
        export_data(df_final)
        
        # Mostrar primeras filas
        print("\n📋 Primeras 5 formulaciones (originales + nuevas):")
        print(df_final.head())
        
        print("\n📋 Últimas 5 formulaciones generadas:")
        print(df_final.tail())
        
        # Mostrar verificación final
        sin_fibra = df_final[df_final['fibra_tipo'] == 'Sin_Fibra']
        print(f"\n✅ Verificación final: {len(sin_fibra)} formulaciones con Sin_Fibra")
        if len(sin_fibra) == 1:
            print(f"   ✅ Solo la Referencia: {sin_fibra['codigo_gomita'].iloc[0]}")
        
        print("\n✅ ¡PROCESO COMPLETADO CON ÉXITO!")
        
        return df_final
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    df_result = main()
    
    if df_result is not None:
        print("\n💡 Datos listos para entrenar tu modelo de IA:")
        print(f"   - {len(df_result)} formulaciones totales")
        print(f"   - {len(df_result[df_result['codigo_gomita'].str.isdigit()])} nuevas formulaciones")
        sin_fibra = df_result[df_result['fibra_tipo'] == 'Sin_Fibra']
        print(f"   - {len(sin_fibra)} formulación con Sin_Fibra (solo la Referencia)")
        print(f"   - Columnas: {', '.join(df_result.columns.tolist())}")