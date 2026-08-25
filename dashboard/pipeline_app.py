"""
PIPELINE DE DATOS - APLICACIÓN PRINCIPAL
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Aplicación principal que orquesta los 6 módulos del pipeline
"""

import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="🍬 Pipeline de Datos - Gomitas IA",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== IMPORTAR MÓDULOS ====================

# Intentar importar desde dashboard/
try:
    from cargar_datos import app_cargar_datos
    from entrenar_modelos import app_entrenar_modelos
    from recomendador_recetas import app_recomendador
    from perfil_sensorial import app_perfil_sensorial
    from prediccion_aceptabilidad import app_prediccion_aceptabilidad
    from comparacion_formulaciones import app_comparacion
except ImportError:
    # Si no está en el mismo directorio, buscar en src
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from cargar_datos import app_cargar_datos
    from entrenar_modelos import app_entrenar_modelos
    from recomendador_recetas import app_recomendador
    from perfil_sensorial import app_perfil_sensorial
    from prediccion_aceptabilidad import app_prediccion_aceptabilidad
    from comparacion_formulaciones import app_comparacion

# ==================== NAVEGACIÓN ====================

def main():
    """Función principal de la aplicación"""
    
    # Título de la aplicación
    st.sidebar.title("🍬 Pipeline de Datos")
    st.sidebar.markdown("---")
    
    # Estado de navegación
    if 'modulo_actual' not in st.session_state:
        st.session_state['modulo_actual'] = 'carga'
    
    # Opciones de navegación
    opciones = {
        'carga': '📂 Carga de Datos',
        'entrenamiento': '🧠 Entrenamiento',
        'recomendador': '🧬 Recomendador',
        'perfil': '📊 Perfil Sensorial',
        'aceptabilidad': '👥 Aceptabilidad',
        'comparacion': '📋 Comparación'
    }
    
    # Mostrar progreso
    st.sidebar.markdown("### 📍 Progreso")
    pasos = ['📂 Carga de Datos', '🧠 Entrenamiento', '🧬 Recomendador', '📊 Perfil Sensorial', '👥 Aceptabilidad', '📋 Comparación']
    paso_actual = list(opciones.keys()).index(st.session_state['modulo_actual'])
    
    for i, paso in enumerate(pasos):
        if i < paso_actual:
            st.sidebar.markdown(f"✅ {paso}")
        elif i == paso_actual:
            st.sidebar.markdown(f"🔄 **{paso}**")
        else:
            st.sidebar.markdown(f"⬜ {paso}")
    
    st.sidebar.markdown("---")
    
    # Estado de datos cargados
    if st.session_state.get('datos_cargados', False):
        st.sidebar.success("✅ Datos cargados")
    else:
        st.sidebar.warning("⚠️ Datos no cargados")
    
    # Estado de modelos entrenados
    if st.session_state.get('modelos_entrenados', False):
        st.sidebar.success("✅ Modelos entrenados")
    else:
        st.sidebar.warning("⚠️ Modelos no entrenados")
    
    st.sidebar.markdown("---")
    
    # Navegación - Botones (Módulo 1)
    if st.sidebar.button("📂 Carga de Datos", use_container_width=True):
        st.session_state['modulo_actual'] = 'carga'
        st.rerun()
    
    # Navegación - Botones (Módulo 2)
    if st.sidebar.button(
        "🧠 Entrenamiento", 
        use_container_width=True, 
        disabled=not st.session_state.get('datos_cargados', False)
    ):
        st.session_state['modulo_actual'] = 'entrenamiento'
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Módulos de Resultados")
    st.sidebar.caption("*(Requieren modelos entrenados)*")
    
    # Navegación - Botones (Módulo 3)
    if st.sidebar.button(
        "🧬 Recomendador", 
        use_container_width=True, 
        disabled=not st.session_state.get('modelos_entrenados', False)
    ):
        st.session_state['modulo_actual'] = 'recomendador'
        st.rerun()
    
    # Navegación - Botones (Módulo 4)
    if st.sidebar.button(
        "📊 Perfil Sensorial", 
        use_container_width=True, 
        disabled=not st.session_state.get('modelos_entrenados', False)
    ):
        st.session_state['modulo_actual'] = 'perfil'
        st.rerun()
    
    # Navegación - Botones (Módulo 5)
    if st.sidebar.button(
        "👥 Aceptabilidad", 
        use_container_width=True, 
        disabled=not st.session_state.get('modelos_entrenados', False)
    ):
        st.session_state['modulo_actual'] = 'aceptabilidad'
        st.rerun()
    
    # Navegación - Botones (Módulo 6)
    if st.sidebar.button(
        "📋 Comparación", 
        use_container_width=True, 
        disabled=not st.session_state.get('modelos_entrenados', False)
    ):
        st.session_state['modulo_actual'] = 'comparacion'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Información del proyecto
    st.sidebar.caption("🍬 **Proyecto Gomitas IA**")
    st.sidebar.caption("Pipeline de datos para diseño de gomitas")
    st.sidebar.caption(f"v2.0 - {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}")
    
    # ==================== CONTENIDO ====================
    
    # Mostrar el módulo correspondiente
    if st.session_state['modulo_actual'] == 'carga':
        app_cargar_datos()
    
    elif st.session_state['modulo_actual'] == 'entrenamiento':
        app_entrenar_modelos()
    
    elif st.session_state['modulo_actual'] == 'recomendador':
        app_recomendador()
    
    elif st.session_state['modulo_actual'] == 'perfil':
        app_perfil_sensorial()
    
    elif st.session_state['modulo_actual'] == 'aceptabilidad':
        app_prediccion_aceptabilidad()
    
    elif st.session_state['modulo_actual'] == 'comparacion':
        app_comparacion()

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    main()