"""Módulo 2: Procesamiento de variables, entrenamiento y evaluación de modelos."""

from datetime import datetime
from pathlib import Path
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

try:
    from dashboard.utils import crear_backup_si_existe, get_project_paths as get_paths
except ImportError:
    from utils import crear_backup_si_existe, get_project_paths as get_paths

warnings.filterwarnings("ignore")

def guardar_con_versionado(df, filepath, nombre_archivo):
    """Guarda un archivo con versionado automático"""
    try:
        backup_path = crear_backup_si_existe(filepath)
        
        if filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(df, f)
        else:
            df.to_excel(filepath, index=False)
        
        mensaje = f"✅ {nombre_archivo} guardado"
        if backup_path:
            mensaje += f" (backup: {backup_path.name})"
        
        return True, mensaje, backup_path
    except Exception as e:
        return False, f"❌ Error: {str(e)}", None

# ==================== FUNCIONES DE PROCESAMIENTO ====================

def procesar_datos(df_formulaciones, df_sensorial, df_propiedades):
    """Procesa y combina los datos"""
    
    # 1. Unir formulaciones con perfil sensorial
    df_combinado = df_formulaciones.merge(df_sensorial, on=['codigo_gomita', 'fibra_tipo'], how='inner')
    
    if len(df_combinado) == 0:
        raise ValueError("❌ No hay coincidencias entre formulaciones y perfil sensorial")
    
    # 2. Unir con propiedades de fibra
    df_final = df_combinado.merge(df_propiedades, on='fibra_tipo', how='left')
    
    # 3. Feature Engineering
    # 3.1 ratio_gelatina_goma
    df_final['ratio_gelatina_goma'] = df_final['gelatina_g'] / (df_final['goma_xantica_g'] + 1e-9)
    
    # 3.2 ratio_azucar_edulcorante
    df_final['ratio_azucar_edulcorante'] = df_final['azucar_g'] / (df_final['stevia_g'] + df_final['sorbitol_g'] + 1e-9)
    
    # 3.3 peso_total_g
    ingredientes = ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 
                   'goma_xantica_g', 'acido_citrico_g', 'stevia_g', 
                   'esencia_g', 'sorbitol_g', 'fibra_g']
    df_final['peso_total_g'] = df_final[ingredientes].sum(axis=1)
    
    # 3.4 porcentaje_fibra
    df_final['porcentaje_fibra'] = (df_final['fibra_g'] / (df_final['peso_total_g'] + 1e-9)) * 100
    
    # 3.5 proporcion_gelatina
    df_final['proporcion_gelatina'] = df_final['gelatina_g'] / (df_final['gelatina_g'] + df_final['goma_xantica_g'] + 1e-9)
    
    # 3.6 edulcorantes_totales_g
    df_final['edulcorantes_totales_g'] = df_final['stevia_g'] + df_final['sorbitol_g']
    
    # 3.7 dulzor_total_g
    df_final['dulzor_total_g'] = df_final['azucar_g'] + df_final['stevia_g'] + df_final['sorbitol_g']
    
    # 3.8 proporcion_agua
    df_final['proporcion_agua'] = df_final['agua_g'] / (df_final['peso_total_g'] + 1e-9)
    
    # 3.9 ratio_fibra_agua
    df_final['ratio_fibra_agua'] = df_final['fibra_g'] / (df_final['agua_g'] + 1e-9)
    
    # 3.10 solidos_totales_g
    solidos = ['azucar_g', 'jarabe_glucosa_g', 'gelatina_g', 'goma_xantica_g', 
               'acido_citrico_g', 'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g']
    df_final['solidos_totales_g'] = df_final[solidos].sum(axis=1)
    
    # 3.11 humedad_estimada
    df_final['humedad_estimada'] = (df_final['agua_g'] / (df_final['peso_total_g'] + 1e-9)) * 100
    
    # 4. One-Hot Encoding
    dummies = pd.get_dummies(df_final['fibra_tipo'], prefix='fibra')
    df_final = pd.concat([df_final, dummies], axis=1)
    
    # 5. Eliminar duplicados
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    # 6. Manejar ratios extremos
    df_final.loc[df_final['goma_xantica_g'] == 0, 'ratio_gelatina_goma'] = 9999
    df_final.loc[(df_final['stevia_g'] + df_final['sorbitol_g']) == 0, 'ratio_azucar_edulcorante'] = 9999
    df_final = df_final.replace([np.inf, -np.inf], 9999)
    
    return df_final

def preparar_aceptabilidad(df_aceptabilidad):
    """Prepara los datos de aceptabilidad"""
    respuesta_map = {
        'Me disgusta mucho': 1, 'Me disgusta': 2, 'Me es indiferente': 3,
        'Me gusta': 4, 'Me gusta mucho': 5
    }
    
    edad_map = {
        '6 a 12 años': 1, 'De 13 a 17 años': 2, '18 a 24 años': 3,
        '25 a 34 años': 4, '35 a 44 años': 5, '45 a 54 años': 6,
        '55 a 64 años': 7, 'Más de 65 años': 8
    }
    sexo_map = {'Femenino': 0, 'Masculino': 1, 'Otro': 2}
    fuma_map = {'No': 0, 'A veces': 1, 'Si': 2}
    
    df = df_aceptabilidad.copy()
    df['edad_num'] = df['edad'].map(edad_map)
    df['sexo_num'] = df['sexo'].map(sexo_map)
    df['fuma_num'] = df['fuma'].map(fuma_map)
    df['apariencia_num'] = df['apariencia'].map(respuesta_map)
    df['color_num'] = df['color'].map(respuesta_map)
    df['sabor_num'] = df['sabor'].map(respuesta_map)
    df['sensacion_num'] = df['sensacion'].map(respuesta_map)
    
    # Codificar aceptacion_global
    aceptacion_map = {'Baja': 0, 'Media': 1, 'Alta': 2, 'Muy Alta': 3}
    df['aceptacion_num'] = df['aceptacion_global'].map(aceptacion_map)
    
    return df

# ==================== ENTRENAMIENTO DE MODELOS ====================

def entrenar_modelos_sensoriales(df, targets, params):
    """Entrena modelos para cada atributo sensorial"""
    resultados = {}
    modelos = {}
    escaladores = {}
    
    # Variables de entrada
    exclude_cols = ['codigo_gomita', 'fibra_tipo', 'brillo', 'claridad', 'dulzura', 
                   'acidez', 'adhesividad', 'elasticidad', 'arenosidad', 'dureza', 
                   'gomosidad', 'masticabilidad']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    for target in targets:
        X = df[feature_cols]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Usar parámetros personalizados
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(3, len(X_train)), scoring='r2')
        
        importance = pd.DataFrame({
            'variable': feature_cols,
            'importancia': model.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        resultados[target] = {
            'r2': r2, 'rmse': rmse, 'mae': mae,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
        modelos[target] = model
        escaladores[target] = scaler
    
    return modelos, escaladores, resultados, feature_cols

def entrenar_modelo_aceptabilidad(df_aceptabilidad, params):
    """Entrena modelo de aceptabilidad"""
    feature_cols = ['edad_num', 'sexo_num', 'fuma_num', 'apariencia_num', 'color_num', 'sabor_num', 'sensacion_num']
    
    X = df_aceptabilidad[feature_cols]
    y = df_aceptabilidad['aceptacion_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 10),
        min_samples_split=params.get('min_samples_split', 5),
        min_samples_leaf=params.get('min_samples_leaf', 2),
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Reporte de clasificación
    target_names = ['Baja', 'Media', 'Alta', 'Muy Alta']
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    # Importancia
    importance = pd.DataFrame({
        'variable': ['Edad', 'Sexo', 'Fuma', 'Apariencia', 'Color', 'Sabor', 'Sensación'],
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return model, accuracy, importance, report, feature_cols

# ==================== GENERAR INFORME ====================

def generar_informe(resultados_sensoriales, accuracy, importancia_aceptabilidad, params, timestamp):
    """Genera un informe completo en formato texto"""
    informe = []
    informe.append("="*70)
    informe.append("📊 INFORME DE ENTRENAMIENTO DE MODELOS")
    informe.append("="*70)
    informe.append(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    informe.append("="*70)
    informe.append("")
    
    # Parámetros usados
    informe.append("🔧 PARÁMETROS DE ENTRENAMIENTO:")
    informe.append("-"*40)
    for param, value in params.items():
        informe.append(f"   {param}: {value}")
    informe.append("")
    
    # Modelos Sensoriales
    informe.append("📈 MODELOS SENSORIALES (Random Forest Regressor):")
    informe.append("-"*40)
    for target, metrics in resultados_sensoriales.items():
        informe.append(f"\n  {target.upper()}:")
        informe.append(f"    R²:  {metrics['r2']:.4f}")
        informe.append(f"    RMSE: {metrics['rmse']:.4f}")
        informe.append(f"    MAE:  {metrics['mae']:.4f}")
        informe.append(f"    CV R²: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    
    informe.append("")
    informe.append("📈 MODELO DE ACEPTABILIDAD (Random Forest Classifier):")
    informe.append("-"*40)
    informe.append(f"  Accuracy: {accuracy:.4f}")
    informe.append("")
    informe.append("  Importancia de variables:")
    for _, row in importancia_aceptabilidad.iterrows():
        informe.append(f"    {row['variable']}: {row['importancia']:.4f}")
    
    informe.append("")
    informe.append("="*70)
    informe.append("✅ INFORME GENERADO AUTOMÁTICAMENTE")
    informe.append("="*70)
    
    return "\n".join(informe)

# ==================== INTERFAZ DE STREAMLIT ====================

def app_entrenar_modelos():
    """Función principal para el módulo de entrenamiento"""
    
    st.title("🧠 Módulo 2: Entrenamiento de Modelos")
    st.markdown("---")
    
    # Verificar que hay datos cargados
    if not st.session_state.get('datos_cargados', False):
        st.warning("⚠️ Primero debes cargar los datos en el Módulo 1")
        if st.button("📂 Ir a Carga de Datos"):
            st.session_state['modulo_actual'] = 'carga'
            st.rerun()
        return
    
    # Obtener datos del session_state
    dataframes = st.session_state['dataframes']
    
    # ==================== CONFIGURACIÓN DE PARÁMETROS ====================
    
    st.subheader("⚙️ Configuración de Algoritmos")
    st.markdown("Ajusta los parámetros de los modelos antes de entrenar")
    
    st.info("💡 **Random Forest** es un algoritmo de Ensemble Learning que combina múltiples árboles de decisión para mejorar la precisión y evitar el sobreajuste.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌲 Random Forest - Regresión (Sensorial)")
        
        n_estimators = st.slider(
            "Número de árboles (n_estimators)",
            min_value=10, max_value=500, value=100, step=10,
            help="Cantidad de árboles en el bosque. Más árboles = mejor precisión, pero más lento."
        )
        
        max_depth = st.slider(
            "Profundidad máxima (max_depth)",
            min_value=1, max_value=20, value=10, step=1,
            help="Profundidad máxima de cada árbol. Valores más altos pueden causar sobreajuste."
        )
        
        min_samples_split = st.slider(
            "Mínimo muestras para dividir (min_samples_split)",
            min_value=2, max_value=20, value=5, step=1,
            help="Número mínimo de muestras para dividir un nodo. Valores más altos reducen sobreajuste."
        )
    
    with col2:
        st.markdown("#### 🌲 Random Forest - Clasificación (Aceptabilidad)")
        st.markdown("*(Usa los mismos parámetros que la regresión)*")
        
        st.caption("📊 **Explicación de parámetros:**")
        st.caption("• **n_estimators**: Más árboles = mejor precisión, pero más lento")
        st.caption("• **max_depth**: Más profundidad = más detalle, pero puede sobreajustar")
        st.caption("• **min_samples_split**: Más alto = menos sobreajuste")
        
        min_samples_leaf = st.slider(
            "Mínimo muestras por hoja (min_samples_leaf)",
            min_value=1, max_value=10, value=2, step=1,
            help="Número mínimo de muestras en una hoja. Valores más altos reducen sobreajuste."
        )
    
    # ==================== BOTÓN DE ENTRENAMIENTO ====================
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        entrenar = st.button(
            "🚀 ENTRENAR MODELOS",
            type="primary",
            use_container_width=True
        )
    
    if entrenar:
        # Preparar parámetros
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
        
        with st.spinner("🔄 Entrenando modelos..."):
            try:
                # 1. Procesar datos
                df_procesado = procesar_datos(
                    dataframes['formulaciones'],
                    dataframes['sensorial'],
                    dataframes['propiedades']
                )
                
                # 2. Preparar aceptabilidad
                df_aceptabilidad = preparar_aceptabilidad(dataframes['aceptabilidad'])
                
                # 3. Entrenar modelos sensoriales
                targets = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
                          'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
                
                modelos, escaladores, resultados_sensoriales, feature_cols = entrenar_modelos_sensoriales(
                    df_procesado, targets, params
                )
                
                # 4. Entrenar modelo de aceptabilidad
                modelo_aceptabilidad, accuracy, importancia_aceptabilidad, report, feature_cols_acept = entrenar_modelo_aceptabilidad(
                    df_aceptabilidad, params
                )
                
                # 5. Guardar en session_state
                st.session_state['modelos_entrenados'] = True
                st.session_state['modelos'] = modelos
                st.session_state['escaladores'] = escaladores
                st.session_state['modelo_aceptabilidad'] = modelo_aceptabilidad
                st.session_state['resultados_sensoriales'] = resultados_sensoriales
                st.session_state['accuracy_aceptabilidad'] = accuracy
                st.session_state['importancia_aceptabilidad'] = importancia_aceptabilidad
                st.session_state['feature_cols'] = feature_cols
                st.session_state['params'] = params
                
                # 6. Guardar en disco con versionado
                paths = get_paths()
                
                # Guardar dataset procesado
                guardar_con_versionado(
                    df_procesado,
                    paths['dataset_procesado'],
                    "Dataset Procesado"
                )
                
                # Guardar modelos
                guardar_con_versionado(
                    modelos,
                    paths['modelos_sensoriales'],
                    "Modelos Sensoriales"
                )
                
                guardar_con_versionado(
                    modelo_aceptabilidad,
                    paths['modelo_aceptabilidad'],
                    "Modelo Aceptabilidad"
                )
                
                guardar_con_versionado(
                    escaladores,
                    paths['escaladores'],
                    "Escaladores"
                )
                
                guardar_con_versionado(
                    feature_cols,
                    paths['features'],
                    "Features"
                )
                
                # 7. Generar y guardar informe
                informe = generar_informe(
                    resultados_sensoriales,
                    accuracy,
                    importancia_aceptabilidad,
                    params,
                    datetime.now()
                )
                
                # Guardar informe con versionado
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                informe_path = paths['reports_dir'] / f"informe_entrenamiento_{timestamp}.txt"
                with open(informe_path, 'w', encoding='utf-8') as f:
                    f.write(informe)
                
                # Guardar informe actual (sin fecha)
                informe_actual = paths['reports_dir'] / "informe_entrenamiento_actual.txt"
                with open(informe_actual, 'w', encoding='utf-8') as f:
                    f.write(informe)
                
                # 8. Mostrar resultados
                st.success("✅ ¡Modelos entrenados y guardados exitosamente!")
                
                # Mostrar informe
                with st.expander("📄 Ver Informe de Entrenamiento", expanded=True):
                    st.text(informe)
                
                # Mostrar métricas en tablas
                st.subheader("📊 Resultados de los Modelos")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Modelos Sensoriales")
                    df_resultados = pd.DataFrame(resultados_sensoriales).T
                    df_resultados = df_resultados.round(4)
                    df_resultados.index.name = 'Atributo'
                    st.dataframe(df_resultados, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎯 Modelo de Aceptabilidad")
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    st.markdown("**Importancia de variables:**")
                    for _, row in importancia_aceptabilidad.iterrows():
                        st.progress(row['importancia'], text=f"{row['variable']}: {row['importancia']:.4f}")
                
                # 9. Botón para siguiente módulo
                st.markdown("---")
                if st.button("➡️ Ir al Módulo 3: Resultados y Recomendaciones", use_container_width=True):
                    st.session_state['modulo_actual'] = 'resultados'
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:
        # Mostrar información antes de entrenar
        st.info("👆 Ajusta los parámetros y presiona 'ENTRENAR MODELOS'")
        
        st.markdown("""
        ### 📋 ¿Qué va a pasar cuando entrenes?
        
        1. **Procesamiento de datos**: Se combinarán y transformarán los 4 archivos
        2. **Entrenamiento de 10 modelos sensoriales**: Uno para cada atributo (brillo, dulzura, etc.)
        3. **Entrenamiento de 1 modelo de aceptabilidad**: Para predecir la aceptación del consumidor
        4. **Evaluación automática**: Se calcularán métricas de rendimiento
        5. **Guardado**: Todo se guardará con versionado automático
        
        ### 📁 Archivos que se generarán:
        
        | Archivo | Ubicación | Descripción |
        |---------|-----------|-------------|
        | dataset_procesado.xlsx | data/processed/ | Datos combinados y procesados |
        | modelos_sensoriales.pkl | models/ | 10 modelos Random Forest |
        | modelo_aceptabilidad.pkl | models/ | Modelo de aceptabilidad |
        | escaladores.pkl | models/ | Escaladores para cada modelo |
        | features.pkl | models/ | Nombres de variables |
        | informe_entrenamiento_*.txt | data/processed/informes/ | Informe completo con fecha |
        """)
    
    # ==================== VER ARCHIVOS GUARDADOS ====================
    
    with st.expander("📂 Ver archivos guardados"):
        paths = get_paths()
        st.write("**Archivos actuales:**")
        
        for name, path in paths.items():
            if path.exists():
                size = path.stat().st_size / 1024  # KB
                st.write(f"  ✅ {name}: {path.name} ({size:.1f} KB)")
            else:
                st.write(f"  ⬜ {name}: No existe aún")
        
        # Ver backups
        backup_dir = paths['data_dir'] / 'backups'
        if backup_dir.exists():
            st.write("**Últimos backups:**")
            backups = sorted(backup_dir.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            for backup in backups:
                st.write(f"  📦 {backup.name}")

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    print("="*60)
    print("🧠 MÓDULO 2: ENTRENAMIENTO DE MODELOS")
    print("="*60)
    print("\nEste módulo se ejecuta dentro de Streamlit.")
    print("Ejecuta: streamlit run dashboard/pipeline_app.py")