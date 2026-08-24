"""
GENERADOR DE DATOS DE ACEPTABILIDAD
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Genera 300 muestras sintéticas de aceptabilidad basadas en datos reales
"""

import os
import numpy as np
import pandas as pd
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Archivos
INPUT_FILE = os.path.join(DATA_DIR, 'Analisis_sensorial_de_Gummies.xlsx')
OUTPUT_FILE = os.path.join(DATA_DIR, 'dataset_aceptabilidad_completo.xlsx')
OUTPUT_REAL = os.path.join(DATA_DIR, 'dataset_aceptabilidad_reales.xlsx')
OUTPUT_SINTETICO = os.path.join(DATA_DIR, 'dataset_aceptabilidad_sintetico.xlsx')

# ==================== 1. CARGAR DATOS REALES ====================

def load_real_data():
    """
    Carga los datos reales de la encuesta
    """
    print("📂 Cargando datos reales...")
    
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name='Respuestas de formulario 1')
        print(f"   ✅ {len(df)} respuestas reales cargadas")
        return df
    except Exception as e:
        print(f"   ❌ Error al cargar: {e}")
        return None

# ==================== 2. LIMPIAR DATOS ====================

def clean_data(df):
    """
    Limpia los datos y elimina la columna de comentarios
    """
    print("\n🧹 Limpiando datos...")
    
    # Crear copia
    df_clean = df.copy()
    
    # Renombrar columnas
    column_mapping = {
        'Marca temporal': 'timestamp',
        'Edad': 'edad',
        'Sexo': 'sexo',
        'Usted Fuma?': 'fuma',
        'Apariencia general': 'apariencia',
        'Color': 'color',
        'Sabor': 'sabor',
        'Sensación al masticar': 'sensacion'
    }
    
    # Solo renombrar columnas que existen
    for old, new in column_mapping.items():
        if old in df_clean.columns:
            df_clean = df_clean.rename(columns={old: new})
    
    # Eliminar columna de comentarios si existe
    if 'Comentarios' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Comentarios'])
    
    # También eliminar cualquier otra columna de comentarios
    comentarios_cols = [col for col in df_clean.columns if 'comentario' in col.lower()]
    if comentarios_cols:
        df_clean = df_clean.drop(columns=comentarios_cols)
    
    print(f"   ✅ Datos limpios: {len(df_clean)} filas, {len(df_clean.columns)} columnas")
    print(f"   📋 Columnas: {', '.join(df_clean.columns.tolist())}")
    
    return df_clean

# ==================== 3. CALCULAR DISTRIBUCIONES ====================

def calculate_distributions(df):
    """
    Calcula las distribuciones de los datos reales
    """
    print("\n📊 Calculando distribuciones...")
    
    distributions = {}
    
    # Distribución de edades
    distributions['edad'] = df['edad'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Edades: {len(distributions['edad'])} categorías")
    
    # Distribución de sexo
    distributions['sexo'] = df['sexo'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Sexo: {len(distributions['sexo'])} categorías")
    
    # Distribución de fumadores
    distributions['fuma'] = df['fuma'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Fuma: {len(distributions['fuma'])} categorías")
    
    # Distribución de apariencia
    distributions['apariencia'] = df['apariencia'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Apariencia: {len(distributions['apariencia'])} categorías")
    
    # Distribución de color
    distributions['color'] = df['color'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Color: {len(distributions['color'])} categorías")
    
    # Distribución de sabor
    distributions['sabor'] = df['sabor'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Sabor: {len(distributions['sabor'])} categorías")
    
    # Distribución de sensación
    distributions['sensacion'] = df['sensacion'].value_counts(normalize=True).to_dict()
    print(f"   ✅ Sensación: {len(distributions['sensacion'])} categorías")
    
    return distributions

# ==================== 4. GENERAR DATOS SINTÉTICOS ====================

def generate_synthetic_data(distributions, n_samples=300):
    """
    Genera datos sintéticos basados en las distribuciones reales
    """
    print(f"\n🧬 Generando {n_samples} muestras sintéticas...")
    
    synthetic_data = []
    
    # Lista de comentarios posibles (no se usarán, pero se guardan por si acaso)
    comentarios_posibles = [
        "Muy rico", "Muy dulce", "Buena textura", "Saludable",
        "Me gustó", "Un poco blando", "Sabor intenso", "Recomendable",
        "Buen producto", "Agradable", "Podría mejorar", "Excelente"
    ]
    
    for i in range(n_samples):
        # Seleccionar demografía basada en distribuciones reales
        edad = np.random.choice(
            list(distributions['edad'].keys()),
            p=list(distributions['edad'].values())
        )
        
        sexo = np.random.choice(
            list(distributions['sexo'].keys()),
            p=list(distributions['sexo'].values())
        )
        
        fuma = np.random.choice(
            list(distributions['fuma'].keys()),
            p=list(distributions['fuma'].values())
        )
        
        # Generar respuestas con ruido controlado
        # (basado en distribuciones reales pero con pequeña variación)
        
        def sample_with_noise(dist, noise=0.05):
            """Muestrea de una distribución con ruido"""
            items = list(dist.keys())
            probs = np.array(list(dist.values()))
            
            # Añadir ruido
            probs = probs + np.random.normal(0, noise, len(probs))
            probs = np.maximum(probs, 0)  # No negativos
            probs = probs / probs.sum()   # Normalizar
            
            return np.random.choice(items, p=probs)
        
        apariencia = sample_with_noise(distributions['apariencia'])
        color = sample_with_noise(distributions['color'])
        sabor = sample_with_noise(distributions['sabor'])
        sensacion = sample_with_noise(distributions['sensacion'])
        
        # Comentario aleatorio (no se usa en el dataset final pero se guarda para referencia)
        comentario = np.random.choice(comentarios_posibles)
        
        # Timestamp sintético
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        synthetic_data.append({
            'timestamp': timestamp,
            'edad': edad,
            'sexo': sexo,
            'fuma': fuma,
            'apariencia': apariencia,
            'color': color,
            'sabor': sabor,
            'sensacion': sensacion
        })
    
    df_synthetic = pd.DataFrame(synthetic_data)
    print(f"   ✅ {len(df_synthetic)} muestras sintéticas generadas")
    
    return df_synthetic

# ==================== 5. CREAR VERSIONES ENTERAS DE CATEGORÍAS ====================

def integer_encode(df):
    """
    Convierte categorías a valores enteros para el modelo
    """
    print("\n🔢 Codificando variables categóricas...")
    
    df_encoded = df.copy()
    
    # Mapeos para codificación
    edad_map = {
        '6 a 12 años': 1,
        'De 13 a 17 años': 2,
        '18 a 24 años': 3,
        '25 a 34 años': 4,
        '35 a 44 años': 5,
        '45 a 54 años': 6,
        '55 a 64 años': 7,
        'Más de 65 años': 8
    }
    
    respuesta_map = {
        'Me disgusta mucho': 1,
        'Me disgusta': 2,
        'Me es indiferente': 3,
        'Me gusta': 4,
        'Me gusta mucho': 5
    }
    
    sexo_map = {'Femenino': 0, 'Masculino': 1, 'Otro': 2}
    fuma_map = {'No': 0, 'A veces': 1, 'Si': 2}
    
    # Aplicar mapeos
    if 'edad' in df_encoded.columns:
        df_encoded['edad_num'] = df_encoded['edad'].map(edad_map)
    
    if 'sexo' in df_encoded.columns:
        df_encoded['sexo_num'] = df_encoded['sexo'].map(sexo_map)
    
    if 'fuma' in df_encoded.columns:
        df_encoded['fuma_num'] = df_encoded['fuma'].map(fuma_map)
    
    if 'apariencia' in df_encoded.columns:
        df_encoded['apariencia_num'] = df_encoded['apariencia'].map(respuesta_map)
    
    if 'color' in df_encoded.columns:
        df_encoded['color_num'] = df_encoded['color'].map(respuesta_map)
    
    if 'sabor' in df_encoded.columns:
        df_encoded['sabor_num'] = df_encoded['sabor'].map(respuesta_map)
    
    if 'sensacion' in df_encoded.columns:
        df_encoded['sensacion_num'] = df_encoded['sensacion'].map(respuesta_map)
    
    # Calcular aceptación promedio
    num_cols = ['apariencia_num', 'color_num', 'sabor_num', 'sensacion_num']
    existing_num_cols = [col for col in num_cols if col in df_encoded.columns]
    if existing_num_cols:
        df_encoded['aceptacion_promedio'] = df_encoded[existing_num_cols].mean(axis=1)
    
    # Categorizar aceptación global
    def categorizar_aceptacion(x):
        if x >= 4.5:
            return 'Muy Alta'
        elif x >= 4:
            return 'Alta'
        elif x >= 3:
            return 'Media'
        elif x >= 2:
            return 'Baja'
        else:
            return 'Muy Baja'
    
    if 'aceptacion_promedio' in df_encoded.columns:
        df_encoded['aceptacion_global'] = df_encoded['aceptacion_promedio'].apply(categorizar_aceptacion)
    
    print(f"   ✅ Variables codificadas")
    print(f"   📋 Columnas: {', '.join(df_encoded.columns.tolist())}")
    
    return df_encoded

# ==================== 6. FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal
    """
    print("\n" + "="*60)
    print("🧬 GENERADOR DE DATOS DE ACEPTABILIDAD")
    print("="*60 + "\n")
    
    # 1. Cargar datos reales
    df_real_raw = load_real_data()
    if df_real_raw is None:
        print("❌ No se pudieron cargar los datos reales")
        return None, None, None
    
    # 2. Limpiar datos (eliminar comentarios)
    df_real = clean_data(df_real_raw)
    
    # 3. Calcular distribuciones
    distributions = calculate_distributions(df_real)
    
    # 4. Generar datos sintéticos
    df_sintetico = generate_synthetic_data(distributions, n_samples=300)
    
    # 5. Combinar datos
    df_combinado = pd.concat([df_real, df_sintetico], ignore_index=True)
    
    # 6. Codificar variables categóricas
    df_real_encoded = integer_encode(df_real)
    df_sintetico_encoded = integer_encode(df_sintetico)
    df_combinado_encoded = integer_encode(df_combinado)
    
    # 7. Guardar archivos
    print("\n💾 Guardando archivos...")
    
    # Guardar solo reales
    df_real_encoded.to_excel(OUTPUT_REAL, index=False)
    print(f"   ✅ Reales guardados: {OUTPUT_REAL}")
    print(f"      {len(df_real_encoded)} muestras reales")
    
    # Guardar solo sintéticos
    df_sintetico_encoded.to_excel(OUTPUT_SINTETICO, index=False)
    print(f"   ✅ Sintéticos guardados: {OUTPUT_SINTETICO}")
    print(f"      {len(df_sintetico_encoded)} muestras sintéticas")
    
    # Guardar combinados
    df_combinado_encoded.to_excel(OUTPUT_FILE, index=False)
    print(f"   ✅ Combinado guardado: {OUTPUT_FILE}")
    print(f"      {len(df_combinado_encoded)} muestras totales")
    
    # 8. Estadísticas finales
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS FINALES")
    print("="*60)
    print(f"\n📋 Distribución de aceptación global (combinado):")
    if 'aceptacion_global' in df_combinado_encoded.columns:
        for cat, count in df_combinado_encoded['aceptacion_global'].value_counts().items():
            print(f"   {cat}: {count} ({count/len(df_combinado_encoded)*100:.1f}%)")
    
    print(f"\n📋 Distribución de edades (combinado):")
    if 'edad' in df_combinado_encoded.columns:
        for edad, count in df_combinado_encoded['edad'].value_counts().items():
            print(f"   {edad}: {count} ({count/len(df_combinado_encoded)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    print(f"\n📁 Archivos generados:")
    print(f"   - dataset_aceptabilidad_reales.xlsx ({len(df_real)} muestras)")
    print(f"   - dataset_aceptabilidad_sintetico.xlsx ({len(df_sintetico)} muestras)")
    print(f"   - dataset_aceptabilidad_completo.xlsx ({len(df_combinado)} muestras)")
    
    return df_real, df_sintetico, df_combinado

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    df_real, df_sintetico, df_combinado = main()
    
    if df_combinado is not None:
        print("\n💡 Datos listos para usar:")
        print(f"   - Reales: {len(df_real)} muestras")
        print(f"   - Sintéticos: {len(df_sintetico)} muestras")
        print(f"   - Total: {len(df_combinado)} muestras")
        print(f"   - Columnas: {', '.join(df_combinado.columns.tolist())}")
        print("\n✅ ¡Listo para entrenar el modelo de aceptabilidad!")