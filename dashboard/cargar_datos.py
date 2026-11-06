"""Módulo 1: Carga, validación estructural y almacenamiento versionado de datasets."""

from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd
import streamlit as st

try:
    from dashboard.utils import crear_backup_si_existe, get_project_paths as get_default_paths
except ImportError:
    from utils import crear_backup_si_existe, get_project_paths as get_default_paths

# ==================== ESTRUCTURAS REQUERIDAS ====================

def get_estructuras_requeridas():
    """Define las columnas requeridas para cada archivo"""
    return {
        'formulaciones': {
            'columnas': [
                'codigo_gomita', 'fibra_tipo', 'azucar_g', 'jarabe_glucosa_g',
                'agua_g', 'gelatina_g', 'goma_xantica_g', 'acido_citrico_g',
                'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g'
            ],
            'descripcion': 'Formulaciones (solo ingredientes)',
            'nombre_archivo': 'formulaciones.xlsx',
            'ejemplo': {
                'codigo_gomita': 'Referencia',
                'fibra_tipo': 'Sin_Fibra',
                'azucar_g': 11.4,
                'jarabe_glucosa_g': 7.8,
                'agua_g': 8.922,
                'gelatina_g': 1.14,
                'goma_xantica_g': 0,
                'acido_citrico_g': 0.522,
                'stevia_g': 0,
                'esencia_g': 0.09,
                'sorbitol_g': 0,
                'fibra_g': 0
            }
        },
        'sensorial': {
            'columnas': [
                'codigo_gomita', 'fibra_tipo', 'brillo', 'claridad', 'dulzura',
                'acidez', 'adhesividad', 'elasticidad', 'arenosidad',
                'dureza', 'gomosidad', 'masticabilidad'
            ],
            'descripcion': 'Perfil Sensorial (atributos evaluados)',
            'nombre_archivo': 'perfil_sensorial.xlsx',
            'ejemplo': {
                'codigo_gomita': 'Referencia',
                'fibra_tipo': 'Sin_Fibra',
                'brillo': 1.7715,
                'claridad': 1.89349,
                'dulzura': 1.82623,
                'acidez': 3.25643,
                'adhesividad': 3.27324,
                'elasticidad': 3.5862,
                'arenosidad': 5.64153,
                'dureza': 1.817,
                'gomosidad': 1.98437,
                'masticabilidad': 1.87247
            }
        },
        'propiedades': {
            'columnas': [
                'fibra_tipo', 'WHC', 'SC', 'OHC', 'FRAP', 'DPPH', 'Polifenoles_totales'
            ],
            'descripcion': 'Propiedades de Fibra',
            'nombre_archivo': 'propiedades_fibra.xlsx',
            'ejemplo': {
                'fibra_tipo': 'Sin_Fibra',
                'WHC': 0,
                'SC': 0,
                'OHC': 0,
                'FRAP': 0,
                'DPPH': 0,
                'Polifenoles_totales': 0
            }
        },
        'aceptabilidad': {
            'columnas': [
                'edad', 'sexo', 'fuma', 'apariencia', 'color', 'sabor', 'sensacion',
                'aceptacion_global'
            ],
            'descripcion': 'Aceptabilidad del Consumidor',
            'nombre_archivo': 'aceptabilidad.xlsx',
            'ejemplo': {
                'edad': '18 a 24 años',
                'sexo': 'Masculino',
                'fuma': 'No',
                'apariencia': 'Me gusta',
                'color': 'Me gusta',
                'sabor': 'Me gusta mucho',
                'sensacion': 'Me gusta',
                'aceptacion_global': 'Alta'
            }
        }
    }

# ==================== FUNCIONES DE VERSIONADO ====================

def guardar_archivo_con_versionado(df, filepath, nombre_archivo):
    """
    Guarda un archivo con versionado automático
    """
    try:
        # Crear backup si el archivo ya existe
        backup_path = crear_backup_si_existe(filepath)
        
        # Guardar el nuevo archivo
        if filepath.suffix == '.csv':
            df.to_csv(filepath, index=False, decimal=',')
        else:
            df.to_excel(filepath, index=False)
        
        mensaje = f"✅ {nombre_archivo} guardado en: {filepath}"
        if backup_path:
            mensaje += f"\n   📦 Backup creado: {backup_path.name}"
        
        return True, mensaje, backup_path
        
    except Exception as e:
        return False, f"❌ Error al guardar {nombre_archivo}: {str(e)}", None

def guardar_todos_los_archivos(dataframes, paths):
    """
    Guarda todos los archivos con versionado automático
    """
    resultados = {
        'guardados': [],
        'errores': [],
        'backups': []
    }
    
    for nombre, df in dataframes.items():
        if nombre in paths:
            filepath = paths[nombre]
            exito, mensaje, backup = guardar_archivo_con_versionado(
                df, filepath, 
                get_estructuras_requeridas()[nombre]['nombre_archivo']
            )
            
            if exito:
                resultados['guardados'].append(mensaje)
                if backup:
                    resultados['backups'].append(backup)
            else:
                resultados['errores'].append(mensaje)
    
    return resultados

# ==================== NORMALIZACIÓN Y VALIDACIÓN ====================

COLUMN_ALIASES = {
    "id_gomita": "codigo_gomita",
    "id_formulacion": "codigo_gomita",
    "tipo_fibra": "fibra_tipo",
    "azucar": "azucar_g",
    "jarabe_glucosa": "jarabe_glucosa_g",
    "agua": "agua_g",
    "gelatina": "gelatina_g",
    "goma_xantica": "goma_xantica_g",
    "acido_citrico": "acido_citrico_g",
    "stevia": "stevia_g",
    "esencia": "esencia_g",
    "sorbitol": "sorbitol_g",
    "fibra": "fibra_g",
    "viable": "buena",
    "polifenoles": "Polifenoles_totales",
    "polifenoles_totales": "Polifenoles_totales",
    "habito_fumar": "fuma",
    "masticar": "sensacion",
    "promedio_aceptacion": "aceptacion_promedio",
    "categoria": "aceptacion_global",
}


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas (soporta tanto formato dashboard como formato paper)."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    rename_map = {
        col: COLUMN_ALIASES[col]
        for col in out.columns
        if col in COLUMN_ALIASES and COLUMN_ALIASES[col] not in out.columns
    }
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def validar_columnas(df, columnas_requeridas, nombre_archivo):
    """Valida que el dataframe tenga las columnas requeridas"""
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        return False, (
            f"❌ Faltan columnas en {nombre_archivo}: {', '.join(columnas_faltantes)} "
            f"(columnas encontradas: {', '.join(map(str, df.columns[:8]))})"
        )
    return True, f"✅ {nombre_archivo} válido: {len(df)} filas, {len(df.columns)} columnas"

def validar_tipos_datos(df, nombre_archivo):
    """Valida que los tipos de datos sean correctos"""
    errores = []
    
    columnas_numericas = {
        'formulaciones': [
            'azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g', 'goma_xantica_g',
            'acido_citrico_g', 'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g'
        ],
        'sensorial': [
            'brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 'elasticidad',
            'arenosidad', 'dureza', 'gomosidad', 'masticabilidad'
        ],
        'propiedades': [
            'WHC', 'SC', 'OHC', 'FRAP', 'DPPH', 'Polifenoles_totales'
        ]
    }
    
    if nombre_archivo in columnas_numericas:
        for col in columnas_numericas[nombre_archivo]:
            if col in df.columns:
                try:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    errores.append(f"  - {col}: debe ser numérico")
    
    return errores

def validar_codigos_coincidentes(df_formulaciones, df_sensorial):
    """Valida que los códigos de gomita coincidan (considerando formulaciones viables si existe 'buena')."""
    if 'buena' in df_formulaciones.columns:
        viables = df_formulaciones[pd.to_numeric(df_formulaciones['buena'], errors='coerce') == 1]
        codigos_esperados = set(viables['codigo_gomita'].astype(str).unique())
        todos_codigos_form = set(df_formulaciones['codigo_gomita'].astype(str).unique())
    else:
        codigos_esperados = set(df_formulaciones['codigo_gomita'].astype(str).unique())
        todos_codigos_form = codigos_esperados

    codigos_sensorial = set(df_sensorial['codigo_gomita'].astype(str).unique())
    
    faltantes_en_sensorial = codigos_esperados - codigos_sensorial
    solo_sensorial = codigos_sensorial - todos_codigos_form
    
    errores = []
    
    if faltantes_en_sensorial:
        errores.append(f"  - Códigos viables sin perfil sensorial: {', '.join(list(faltantes_en_sensorial)[:5])}")
    if solo_sensorial:
        errores.append(f"  - Códigos solo en sensorial: {', '.join(list(solo_sensorial)[:5])}")
    
    return errores, len(todos_codigos_form & codigos_sensorial)

def validar_tipos_fibra(df_formulaciones, df_propiedades):
    """Valida que los tipos de fibra coincidan"""
    tipos_formulaciones = set(df_formulaciones['fibra_tipo'].unique())
    tipos_propiedades = set(df_propiedades['fibra_tipo'].unique())
    
    faltantes = tipos_formulaciones - tipos_propiedades
    
    errores = []
    if faltantes:
        errores.append(f"  - Tipos de fibra sin propiedades: {', '.join(faltantes)}")
    
    return errores

# ==================== FUNCIÓN DE CARGA ====================

def cargar_archivo(archivo, nombre, estructuras):
    """Carga un archivo y valida su estructura"""
    try:
        nombre_archivo = archivo.name.lower()
        if nombre_archivo.endswith('.csv'):
            archivo.seek(0)
            df = pd.read_csv(archivo, sep=None, engine='python')
        else:
            df = pd.read_excel(archivo)

        df = normalizar_dataframe(df)

        columnas_requeridas = list(estructuras[nombre]['columnas'])
        # Si sensorial viene sin fibra_tipo (formato paper), se completará al cruzar con formulaciones
        if nombre == 'sensorial' and 'fibra_tipo' not in df.columns:
            columnas_requeridas = [c for c in columnas_requeridas if c != 'fibra_tipo']

        valido, mensaje = validar_columnas(df, columnas_requeridas, estructuras[nombre]['descripcion'])
        
        if not valido:
            return None, mensaje
        
        errores_tipos = validar_tipos_datos(df, nombre)
        if errores_tipos:
            return None, "❌ Errores de tipo de dato:\n" + "\n".join(errores_tipos)
        
        return df, mensaje
        
    except Exception as e:
        return None, f"❌ Error al cargar {estructuras[nombre]['descripcion']}: {str(e)}"

# ==================== FUNCIÓN PRINCIPAL ====================

def cargar_y_validar_datos(archivos):
    """
    Carga, valida y guarda todos los archivos con versionado
    """
    estructuras = get_estructuras_requeridas()
    paths = get_default_paths()
    
    dataframes = {}
    errores = []
    estadisticas = {}
    
    # 1. Cargar cada archivo
    for nombre, archivo in archivos.items():
        if archivo is None:
            errores.append(f"❌ No se cargó el archivo: {estructuras[nombre]['descripcion']}")
            continue
        
        df, mensaje = cargar_archivo(archivo, nombre, estructuras)
        
        if df is not None:
            dataframes[nombre] = df
            estadisticas[nombre] = {
                'filas': len(df),
                'columnas': len(df.columns),
                'mensaje': mensaje
            }
        else:
            errores.append(mensaje)
    
    # 2. Validaciones cruzadas
    if 'formulaciones' in dataframes and 'sensorial' in dataframes:
        if 'fibra_tipo' not in dataframes['sensorial'].columns and 'fibra_tipo' in dataframes['formulaciones'].columns:
            mapa_fibra = (
                dataframes['formulaciones']
                .drop_duplicates(subset=['codigo_gomita'])
                .set_index('codigo_gomita')['fibra_tipo']
            )
            dataframes['sensorial']['fibra_tipo'] = dataframes['sensorial']['codigo_gomita'].map(mapa_fibra)

        errores_codigos, coincidencias = validar_codigos_coincidentes(
            dataframes['formulaciones'], dataframes['sensorial']
        )
        if errores_codigos:
            errores.extend(errores_codigos)
        estadisticas['coincidencias'] = coincidencias
    
    if 'formulaciones' in dataframes and 'propiedades' in dataframes:
        errores_fibra = validar_tipos_fibra(
            dataframes['formulaciones'], dataframes['propiedades']
        )
        if errores_fibra:
            errores.extend(errores_fibra)
    
    # 3. Guardar archivos con versionado (si no hay errores)
    if not errores:
        # Crear paths para cada archivo
        archivos_paths = {
            'formulaciones': paths['formulaciones'],
            'sensorial': paths['sensorial'],
            'propiedades': paths['propiedades'],
            'aceptabilidad': paths['aceptabilidad']
        }
        
        resultados_guardado = guardar_todos_los_archivos(dataframes, archivos_paths)
        
        if resultados_guardado['errores']:
            errores.extend(resultados_guardado['errores'])
        else:
            estadisticas['archivos_guardados'] = len(resultados_guardado['guardados'])
            estadisticas['backups_creados'] = len(resultados_guardado['backups'])
            estadisticas['backups_lista'] = [str(b) for b in resultados_guardado['backups']]
    
    return dataframes, errores, estadisticas

# ==================== FUNCIÓN DE VISUALIZACIÓN EN STREAMLIT ====================

def mostrar_resumen_carga(dataframes, errores, estadisticas):
    """Muestra un resumen de la carga en Streamlit"""
    
    if errores:
        st.error("❌ Se encontraron errores en la carga:")
        for error in errores:
            st.write(f"- {error}")
        return False
    
    st.success("✅ Todos los archivos cargados y guardados correctamente")
    
    # Resumen de archivos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Formulaciones", estadisticas.get('formulaciones', {}).get('filas', 0))
    with col2:
        st.metric("📊 Perfil Sensorial", estadisticas.get('sensorial', {}).get('filas', 0))
    with col3:
        st.metric("🧬 Propiedades Fibra", estadisticas.get('propiedades', {}).get('filas', 0))
    with col4:
        st.metric("👥 Aceptabilidad", estadisticas.get('aceptabilidad', {}).get('filas', 0))
    
    # Coincidencias
    if 'coincidencias' in estadisticas:
        st.info(f"🔗 Códigos coincidentes entre formulaciones y sensorial: {estadisticas['coincidencias']}")
    
    # Información de guardado
    if 'archivos_guardados' in estadisticas:
        st.success(f"💾 {estadisticas['archivos_guardados']} archivos guardados en data/processed/")
        
        if 'backups_creados' in estadisticas and estadisticas['backups_creados'] > 0:
            st.info(f"📦 {estadisticas['backups_creados']} backups creados en data/processed/backups/")
            
            # Mostrar lista de backups
            with st.expander("📋 Ver lista de backups creados"):
                for backup in estadisticas.get('backups_lista', []):
                    st.write(f"  • {Path(backup).name}")
    
    # Mostrar preview de cada archivo
    with st.expander("📖 Ver preview de los archivos cargados"):
        for nombre, df in dataframes.items():
            st.subheader(f"{nombre.upper()}")
            st.write(f"{estadisticas.get(nombre, {}).get('mensaje', '')}")
            st.dataframe(df.head(3), use_container_width=True)
    
    # Mostrar ubicación de archivos
    with st.expander("📂 Ubicación de archivos"):
        paths = get_default_paths()
        st.write("**Archivos guardados en:**")
        for nombre, path in paths.items():
            if nombre != 'backup_dir':
                st.write(f"  • {nombre}: `{path}`")
        st.write("**Backups guardados en:**")
        st.write(f"  • `{paths['backup_dir']}`")
    
    return True

def mostrar_plantillas():
    """Muestra las plantillas de los archivos requeridos"""
    estructuras = get_estructuras_requeridas()
    
    with st.expander("📋 Ver plantillas de archivos requeridos"):
        for nombre, info in estructuras.items():
            st.subheader(f"📄 {info['descripcion']}")
            st.code(f"Columnas: {', '.join(info['columnas'])}")
            st.caption(f"Archivo: {info['nombre_archivo']}")
            
            # Mostrar ejemplo como tabla
            df_ejemplo = pd.DataFrame([info['ejemplo']])
            st.dataframe(df_ejemplo, use_container_width=True)
            st.markdown("---")

# ==================== FUNCIÓN PRINCIPAL PARA STREAMLIT ====================

def app_cargar_datos():
    """
    Función principal para Streamlit - Módulo de Carga de Datos
    """
    st.title("📂 Módulo 1: Carga de Datos")
    st.markdown("---")
    
    # Mostrar plantillas
    mostrar_plantillas()
    
    st.markdown("---")
    st.subheader("📤 Cargar Archivos")
    
    st.info("💡 **Los archivos de Formulaciones y Perfil Sensorial están SEPARADOS**")
    st.info("📦 **Los archivos existentes se respaldan automáticamente antes de guardar**")
    
    # Sidebar con instrucciones
    with st.sidebar:
        st.header("📋 Instrucciones")
        st.markdown("""
        1. Carga los 4 archivos en el orden correcto
        2. Cada archivo debe tener las columnas especificadas
        3. Formatos aceptados: .xlsx, .xls, .csv
        4. Los códigos de gomita deben coincidir entre Formulaciones y Sensorial
        """)
        
        st.markdown("---")
        st.caption("💡 **Tips:**")
        st.caption("- Usa punto (.) para decimales")
        st.caption("- No uses comillas en los textos")
        st.caption("- Asegúrate de que los códigos sean únicos")
        
        st.markdown("---")
        st.caption("📋 **Estructura separada:**")
        st.caption("• formulaciones.xlsx → Solo ingredientes")
        st.caption("• perfil_sensorial.xlsx → Solo atributos")
        
        st.markdown("---")
        st.caption("📦 **Versionado automático:**")
        st.caption("• Los archivos se guardan en `data/processed/`")
        st.caption("• Los respaldos van a `data/processed/backups/`")
        st.caption("• Los backups tienen fecha y hora")
    
    # Carga de archivos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Ingredientes")
        file_formulaciones = st.file_uploader(
            "formulaciones.xlsx",
            type=['xlsx', 'xls', 'csv'],
            key="formulaciones_upload"
        )
        
        st.markdown("#### 📊 Perfil Sensorial")
        file_sensorial = st.file_uploader(
            "perfil_sensorial.xlsx",
            type=['xlsx', 'xls', 'csv'],
            key="sensorial_upload"
        )
    
    with col2:
        st.markdown("#### 🧬 Propiedades de Fibra")
        file_propiedades = st.file_uploader(
            "propiedades_fibra.xlsx",
            type=['xlsx', 'xls', 'csv'],
            key="propiedades_upload"
        )
        
        st.markdown("#### 👥 Aceptabilidad")
        file_aceptabilidad = st.file_uploader(
            "aceptabilidad.xlsx",
            type=['xlsx', 'xls', 'csv'],
            key="aceptabilidad_upload"
        )
    
    st.markdown("---")
    
    # Botón de carga
    if st.button("🚀 Cargar, Validar y Guardar Datos", type="primary", use_container_width=True):
        
        archivos = {
            'formulaciones': file_formulaciones,
            'sensorial': file_sensorial,
            'propiedades': file_propiedades,
            'aceptabilidad': file_aceptabilidad
        }
        
        # Verificar que todos los archivos estén cargados
        faltantes = [nombre for nombre, archivo in archivos.items() if archivo is None]
        
        if faltantes:
            st.error(f"❌ Faltan archivos: {', '.join(faltantes)}")
            return
        
        with st.spinner("🔄 Cargando, validando y guardando datos..."):
            dataframes, errores, estadisticas = cargar_y_validar_datos(archivos)
        
        # Mostrar resultados
        if mostrar_resumen_carga(dataframes, errores, estadisticas):
            # Guardar en session_state para el siguiente módulo
            st.session_state['dataframes'] = dataframes
            st.session_state['estadisticas'] = estadisticas
            st.session_state['datos_cargados'] = True
            
            st.success("✅ Datos cargados y guardados correctamente. ¡Puedes pasar al siguiente módulo!")
            
            # Botón para siguiente paso
            if st.button("➡️ Ir al Módulo 2: Entrenamiento", use_container_width=True):
                st.session_state['modulo_actual'] = 'entrenamiento'
                st.rerun()
        else:
            st.error("❌ Corrige los errores y vuelve a intentar")

# ==================== EJECUCIÓN DIRECTA (PARA PRUEBAS) ====================

if __name__ == "__main__":
    print("="*60)
    print("📂 MÓDULO 1: CARGA DE DATOS CON VERSIONADO")
    print("="*60)
    print("\nEste módulo se ejecuta dentro de Streamlit.")
    print("Ejecuta: streamlit run dashboard/pipeline_app.py")
    print("\nCaracterísticas:")
    print("✅ Carga y valida 4 archivos")
    print("✅ Guarda en data/processed/")
    print("✅ Crea backups automáticos con fecha y hora")
    print("✅ Mantiene historial de versiones")
    
    estructuras = get_estructuras_requeridas()
    print("\n📋 Estructuras requeridas:")
    for nombre, info in estructuras.items():
        print(f"\n  {info['descripcion']}:")
        print(f"    Archivo: {info['nombre_archivo']}")
        print(f"    Columnas: {', '.join(info['columnas'])}")