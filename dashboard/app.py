"""
DASHBOARD INTERACTIVO - PREDICCIÓN DE PERFIL SENSORIAL
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Aplicación Streamlit para predecir el perfil sensorial de gomitas
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

# Configuración de la página (debe ser lo primero)
st.set_page_config(
    page_title="🍬 Diseñador de Gomitas - Perfil Sensorial",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

MODELS_FILE = os.path.join(MODELS_DIR, 'modelos_sensoriales_completos.pkl')
FEATURES_FILE = os.path.join(MODELS_DIR, 'features_sensoriales.pkl')

# ==================== FUNCIONES ====================

@st.cache_resource
def load_models():
    """
    Carga los modelos entrenados (con caché para mejor rendimiento)
    """
    try:
        with open(MODELS_FILE, 'rb') as f:
            data = pickle.load(f)
        
        with open(FEATURES_FILE, 'rb') as f:
            features = pickle.load(f)
        
        return data['models'], data['scalers'], data['targets'], features
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {e}")
        return None, None, None, None

def predict_sensorial(formulacion, models, scalers, targets, features):
    """
    Predice todos los atributos sensoriales para una formulación
    """
    # Crear DataFrame con la formulación
    X_new = pd.DataFrame([formulacion])[features]
    
    # Predecir cada atributo
    predicciones = {}
    for target in targets:
        X_scaled = scalers[target].transform(X_new)
        predicciones[target] = models[target].predict(X_scaled)[0]
    
    return predicciones

def create_radar_chart(predicciones, targets):
    """
    Crea un gráfico de radar (telaraña) con el perfil sensorial
    """
    # Configurar
    categories = [t.upper() for t in targets]
    values = [predicciones[t] for t in targets]
    
    # Cerrar el círculo
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Dibujar
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # Configurar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 7)
    ax.set_yticks(range(1, 8))
    ax.set_yticklabels(range(1, 8), fontsize=8)
    ax.grid(True)
    ax.set_title('Perfil Sensorial - Radar', fontsize=14, pad=20)
    
    return fig

def create_bar_chart(predicciones, targets):
    """
    Crea un gráfico de barras con el perfil sensorial
    """
    categories = [t.upper() for t in targets]
    values = [predicciones[t] for t in targets]
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
    
    # Añadir valores
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, 7.5)
    ax.set_ylabel('Intensidad (escala 1-7)', fontsize=12)
    ax.set_title('Perfil Sensorial - Barras', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig

def display_ingredient_impact(predicciones, valores_base, ingrediente, valor_cambiado):
    """
    Muestra el impacto de cambiar un ingrediente
    """
    # Crear comparación
    cambios = {}
    for target, pred in predicciones.items():
        if target in valores_base:
            cambio = pred - valores_base[target]
            cambios[target] = cambio
    
    # Crear gráfico
    categories = [t.upper() for t in cambios.keys()]
    values = list(cambios.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.barh(categories, values, color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Cambio en la intensidad', fontsize=12)
    ax.set_title(f'Impacto de cambiar {ingrediente} a {valor_cambiado}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01 if val > 0 else bar.get_width() - 0.2,
                bar.get_y() + bar.get_height()/2,
                f'{val:+.2f}', va='center', fontsize=10, fontweight='bold')
    
    return fig

# ==================== INTERFAZ PRINCIPAL ====================

def main():
    """
    Función principal del dashboard
    """
    # Título
    st.title("🍬 Diseñador de Gomitas - Perfil Sensorial")
    st.markdown("---")
    
    # Cargar modelos
    with st.spinner("Cargando modelos de IA..."):
        models, scalers, targets, features = load_models()
    
    if models is None:
        st.error("❌ No se pudieron cargar los modelos. Asegúrate de haber entrenado los modelos primero.")
        st.info("Ejecuta: python src/modelo_sensorial_completo.py")
        return
    
    st.success(f"✅ Modelos cargados: {len(targets)} atributos sensoriales")
    
    # ==================== SIDEBAR ====================
    st.sidebar.header("🍭 Ajustar Receta")
    st.sidebar.markdown("---")
    
    # Valores predeterminados (tomados de una formulación típica)
    default_values = {
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
    
    # Sliders para ingredientes principales
    st.sidebar.subheader("🍬 Ingredientes Principales")
    
    azucar = st.sidebar.slider("Azúcar (g)", 0.0, 11.4, default_values['azucar_g'], 0.1)
    jarabe = st.sidebar.slider("Jarabe de Glucosa (g)", 0.0, 7.8, default_values['jarabe_glucosa_g'], 0.1)
    agua = st.sidebar.slider("Agua (g)", 0.0, 8.9, default_values['agua_g'], 0.1)
    gelatina = st.sidebar.slider("Gelatina (g)", 0.0, 1.2, default_values['gelatina_g'], 0.05)
    goma_xantica = st.sidebar.slider("Goma Xántica (g)", 0.0, 1.2, default_values['goma_xantica_g'], 0.05)
    acido_citrico = st.sidebar.slider("Ácido Cítrico (g)", 0.0, 0.6, default_values['acido_citrico_g'], 0.05)
    stevia = st.sidebar.slider("Stevia (g)", 0.0, 1.4, default_values['stevia_g'], 0.05)
    esencia = st.sidebar.slider("Esencia (g)", 0.0, 0.6, default_values['esencia_g'], 0.05)
    sorbitol = st.sidebar.slider("Sorbitol (g)", 0.0, 5.5, default_values['sorbitol_g'], 0.1)
    fibra = st.sidebar.slider("Fibra (g)", 0.0, 2.6, default_values['fibra_g'], 0.05)
    
    st.sidebar.markdown("---")
    
    # Tipo de fibra
    st.sidebar.subheader("🌿 Tipo de Fibra")
    fibra_tipo = st.sidebar.selectbox(
        "Seleccionar tipo:",
        ["Batata_Cáscara_3%", "Batata_Cáscara", "Batata_Cáscara_5%", "Sin_Fibra"]
    )
    
    # Botón de predicción
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predecir Perfil Sensorial", use_container_width=True)
    
    # Mostrar sugerencias
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Consejo: Ajusta los sliders y presiona 'Predecir' para ver cómo cambia el perfil sensorial.")
    
    # ==================== CUERPO PRINCIPAL ====================
    
    # Si no se ha presionado el botón, usar valores predeterminados
    if not predict_button:
        st.info("👆 Ajusta los ingredientes en el panel izquierdo y presiona 'Predecir Perfil Sensorial'")
        
        # Mostrar formulación actual
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Formulación Actual")
            st.json(default_values)
        
        # Mostrar ejemplo de perfil
        with col2:
            st.subheader("📊 Perfil Sensorial (ejemplo)")
            st.image("https://via.placeholder.com/400x300?text=Ajusta+y+predice", use_container_width=True)
        
        return
    
    # ==================== PREDICCIÓN ====================
    
    # Construir formulación
    formulacion = {
        'azucar_g': azucar,
        'jarabe_glucosa_g': jarabe,
        'agua_g': agua,
        'gelatina_g': gelatina,
        'goma_xantica_g': goma_xantica,
        'acido_citrico_g': acido_citrico,
        'stevia_g': stevia,
        'esencia_g': esencia,
        'sorbitol_g': sorbitol,
        'fibra_g': fibra,
        'WHC': default_values['WHC'],
        'SC': default_values['SC'],
        'OHC': default_values['OHC'],
        'FRAP': default_values['FRAP'],
        'DPPH': default_values['DPPH'],
        'Polifenoles_totales': default_values['Polifenoles_totales'],
        'ratio_gelatina_goma': default_values['ratio_gelatina_goma'],
        'ratio_azucar_edulcorante': default_values['ratio_azucar_edulcorante'],
        'peso_total_g': default_values['peso_total_g'],
        'porcentaje_fibra': default_values['porcentaje_fibra'],
        'proporcion_gelatina': default_values['proporcion_gelatina'],
        'edulcorantes_totales_g': default_values['edulcorantes_totales_g'],
        'dulzor_total_g': default_values['dulzor_total_g'],
        'proporcion_agua': default_values['proporcion_agua'],
        'ratio_fibra_agua': default_values['ratio_fibra_agua'],
        'solidos_totales_g': default_values['solidos_totales_g'],
        'humedad_estimada': default_values['humedad_estimada'],
        'fibra_Batata_Cáscara': 1 if fibra_tipo == "Batata_Cáscara" else 0,
        'fibra_Batata_Cáscara_3%': 1 if fibra_tipo == "Batata_Cáscara_3%" else 0,
        'fibra_Batata_Cáscara_5%': 1 if fibra_tipo == "Batata_Cáscara_5%" else 0,
        'fibra_Sin_Fibra': 1 if fibra_tipo == "Sin_Fibra" else 0
    }
    
    # Predecir
    with st.spinner("🔮 Prediciendo perfil sensorial..."):
        predicciones = predict_sensorial(formulacion, models, scalers, targets, features)
    
    # ==================== MOSTRAR RESULTADOS ====================
    
    st.success("✅ ¡Predicción completada!")
    st.markdown("---")
    
    # Columnas principales
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Formulación")
        st.json({
            'Azúcar': f"{azucar}g",
            'Jarabe Glucosa': f"{jarabe}g",
            'Agua': f"{agua}g",
            'Gelatina': f"{gelatina}g",
            'Goma Xántica': f"{goma_xantica}g",
            'Ácido Cítrico': f"{acido_citrico}g",
            'Stevia': f"{stevia}g",
            'Esencia': f"{esencia}g",
            'Sorbitol': f"{sorbitol}g",
            'Fibra': f"{fibra}g",
            'Tipo Fibra': fibra_tipo
        })
        
        # Tabla de predicciones
        st.subheader("📊 Predicciones")
        df_pred = pd.DataFrame({
            'Atributo': [t.upper() for t in targets],
            'Valor': [predicciones[t] for t in targets]
        })
        st.dataframe(df_pred, hide_index=True, use_container_width=True)
    
    with col2:
        # Radar Chart
        st.subheader("📈 Perfil Sensorial - Radar")
        fig_radar = create_radar_chart(predicciones, targets)
        st.pyplot(fig_radar)
        plt.close(fig_radar)
        
        # Bar Chart
        st.subheader("📊 Perfil Sensorial - Barras")
        fig_bar = create_bar_chart(predicciones, targets)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    
    # ==================== INTERPRETACIÓN ====================
    
    st.markdown("---")
    st.subheader("💡 Interpretación del Perfil Sensorial")
    
    # Generar interpretación automática
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dulzura = predicciones['dulzura']
        if dulzura > 4:
            st.success(f"🍬 **Dulzura Alta** ({dulzura:.2f})")
            st.caption("La gomita será muy dulce. Ideal para consumidores que prefieren sabores intensos.")
        elif dulzura > 2.5:
            st.info(f"🍬 **Dulzura Media** ({dulzura:.2f})")
            st.caption("Balance adecuado de dulzura. Aceptable para la mayoría de los consumidores.")
        else:
            st.warning(f"🍬 **Dulzura Baja** ({dulzura:.2f})")
            st.caption("Poco dulce. Considera aumentar azúcar o stevia.")
    
    with col2:
        adhesividad = predicciones['adhesividad']
        if adhesividad > 4:
            st.warning(f"🔗 **Adhesividad Alta** ({adhesividad:.2f})")
            st.caption("Se pega a los dientes. Puede ser incómodo para algunos consumidores.")
        elif adhesividad > 2.5:
            st.info(f"🔗 **Adhesividad Media** ({adhesividad:.2f})")
            st.caption("Adhesividad moderada. Textura aceptable.")
        else:
            st.success(f"🔗 **Adhesividad Baja** ({adhesividad:.2f})")
            st.caption("No se pega a los dientes. Excelente textura.")
    
    with col3:
        dureza = predicciones['dureza']
        if dureza > 3:
            st.warning(f"💪 **Dureza Alta** ({dureza:.2f})")
            st.caption("Gomita firme. Puede ser difícil de masticar para algunos.")
        elif dureza > 2:
            st.info(f"💪 **Dureza Media** ({dureza:.2f})")
            st.caption("Dureza balanceada. Textura agradable.")
        else:
            st.success(f"💪 **Dureza Baja** ({dureza:.2f})")
            st.caption("Gomita suave. Fácil de masticar.")
    
    # ==================== COMPARACIÓN CON OTRAS FORMULACIONES ====================
    
    st.markdown("---")
    st.subheader("🔄 Comparar con otra formulación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Formulación actual")
        st.json({
            'Azúcar': f"{azucar}g",
            'Fibra': f"{fibra}g",
            'Tipo': fibra_tipo
        })
    
    with col2:
        # Opción para guardar la formulación actual como referencia
        if st.button("💾 Guardar como referencia", use_container_width=True):
            st.session_state['referencia'] = predicciones.copy()
            st.success("✅ Referencia guardada")
    
    # Mostrar comparación si existe referencia
    if 'referencia' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Comparación con Referencia")
        
        # Crear DataFrame de comparación
        df_comp = pd.DataFrame({
            'Atributo': [t.upper() for t in targets],
            'Actual': [predicciones[t] for t in targets],
            'Referencia': [st.session_state['referencia'][t] for t in targets],
            'Diferencia': [predicciones[t] - st.session_state['referencia'][t] for t in targets]
        })
        
        # Colorear diferencias
        def color_diff(val):
            if val > 0.1:
                return 'color: green'
            elif val < -0.1:
                return 'color: red'
            return ''
        
        st.dataframe(
            df_comp.style.applymap(color_diff, subset=['Diferencia']),
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("🗑️ Limpiar referencia", use_container_width=True):
            del st.session_state['referencia']
            st.rerun()
    
    # ==================== RECOMENDACIONES ====================
    
    st.markdown("---")
    st.subheader("🎯 Recomendaciones para mejorar")
    
    # Generar recomendaciones basadas en las predicciones
    recomendaciones = []
    
    # Dulzura
    if predicciones['dulzura'] < 2.5:
        recomendaciones.append("📈 **Aumenta la dulzura**: Incrementa azúcar o stevia. La gelatina también ayuda a liberar sabores.")
    
    # Adhesividad
    if predicciones['adhesividad'] > 4.5:
        recomendaciones.append("📉 **Reduce la adhesividad**: Disminuye fibra o aumenta gelatina. La goma xántica también puede ayudar.")
    
    # Dureza
    if predicciones['dureza'] > 3.5:
        recomendaciones.append("📉 **Suaviza la textura**: Reduce azúcar o fibra. Aumenta agua o gelatina.")
    elif predicciones['dureza'] < 1.5:
        recomendaciones.append("📈 **Aumenta firmeza**: Incrementa azúcar, fibra o gelatina.")
    
    # Arenosidad
    if predicciones['arenosidad'] > 5:
        recomendaciones.append("📉 **Reduce arenosidad**: Disminuye fibra o stevia. Usa fibra más fina.")
    
    # Elasticidad
    if predicciones['elasticidad'] < 2.5:
        recomendaciones.append("📈 **Aumenta elasticidad**: Incrementa azúcar o usa fibra Batata_Cáscara_5%.")
    
    # Mostrar recomendaciones
    if recomendaciones:
        for rec in recomendaciones:
            st.info(rec)
    else:
        st.success("✅ ¡Perfil sensorial balanceado! No se requieren ajustes mayores.")

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    main()