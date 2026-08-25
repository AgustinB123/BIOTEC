"""
MÓDULO 4: PERFIL SENSORIAL
Autor: Proyecto Gomitas IA
Fecha: 2026-08-25
Descripción: Visualiza el perfil sensorial de cualquier receta
ESCALA: 1-9 (corregido)
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================

def get_paths():
    """Obtiene las rutas de los archivos"""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models'
    return {
        'modelos_sensoriales': models_dir / 'modelos_sensoriales.pkl',
        'escaladores': models_dir / 'escaladores.pkl',
        'features': models_dir / 'features.pkl'
    }

def cargar_modelos():
    """Carga los modelos entrenados"""
    paths = get_paths()
    
    try:
        with open(paths['modelos_sensoriales'], 'rb') as f:
            modelos = pickle.load(f)
        with open(paths['escaladores'], 'rb') as f:
            escaladores = pickle.load(f)
        with open(paths['features'], 'rb') as f:
            features = pickle.load(f)
        return modelos, escaladores, features, None
    except Exception as e:
        return None, None, None, str(e)

# ==================== DEFINIR LÍMITES ====================

def get_ingredient_limits():
    """Define los límites de los ingredientes"""
    return {
        'azucar_g': (0.2, 11.4),
        'jarabe_glucosa_g': (0.3, 7.8),
        'agua_g': (0.5, 8.922),
        'gelatina_g': (0.0, 1.2),
        'goma_xantica_g': (0.0, 1.2),
        'acido_citrico_g': (0.05, 0.6),
        'stevia_g': (0.0, 1.4),
        'esencia_g': (0.05, 0.6),
        'sorbitol_g': (0.0, 5.5),
        'fibra_g': (0.0, 2.6)
    }

def get_ingredient_descriptions():
    """Descripciones de cada ingrediente"""
    return {
        'azucar_g': 'Azúcar: Aporta dulzura y cuerpo a la gomita',
        'jarabe_glucosa_g': 'Jarabe de Glucosa: Evita la cristalización y aporta humedad',
        'agua_g': 'Agua: Base de la formulación, disuelve los ingredientes',
        'gelatina_g': 'Gelatina: Da la textura gomosa y elasticidad',
        'goma_xantica_g': 'Goma Xántica: Estabilizador y espesante natural',
        'acido_citrico_g': 'Ácido Cítrico: Aporta acidez y realza sabores',
        'stevia_g': 'Stevia: Edulcorante natural, más dulce que el azúcar',
        'esencia_g': 'Esencia: Aporta sabor y aroma a la gomita',
        'sorbitol_g': 'Sorbitol: Edulcorante que aporta humedad y textura',
        'fibra_g': 'Fibra: Aporta beneficios saludables y modifica la textura'
    }

# ==================== PREDECIR PERFIL ====================

def predecir_perfil(receta, modelos, escaladores, targets, features):
    """Predice el perfil sensorial de una receta"""
    receta_df = pd.DataFrame([receta])
    
    for col in features:
        if col not in receta_df.columns:
            receta_df[col] = 0
    
    receta_df = receta_df[features]
    
    predicciones = {}
    for target in targets:
        try:
            X_scaled = escaladores[target].transform(receta_df)
            predicciones[target] = modelos[target].predict(X_scaled)[0]
        except:
            predicciones[target] = 0
    
    return predicciones

# ==================== INTERFAZ DE STREAMLIT ====================

def app_perfil_sensorial():
    """Aplicación principal del Perfil Sensorial"""
    
    st.title("📊 Perfil Sensorial")
    st.markdown("---")
    
    # Verificar que hay modelos entrenados
    if not st.session_state.get('modelos_entrenados', False):
        st.warning("⚠️ Primero debes entrenar los modelos en el Módulo 2")
        if st.button("📂 Ir a Entrenamiento"):
            st.session_state['modulo_actual'] = 'entrenamiento'
            st.rerun()
        return
    
    # Cargar modelos
    modelos, escaladores, features, error = cargar_modelos()
    
    if error:
        st.error(f"❌ Error al cargar modelos: {error}")
        return
    
    targets = ['brillo', 'claridad', 'dulzura', 'acidez', 'adhesividad', 
               'elasticidad', 'arenosidad', 'dureza', 'gomosidad', 'masticabilidad']
    
    limits = get_ingredient_limits()
    descriptions = get_ingredient_descriptions()
    
    # ==================== SIDEBAR: INSTRUCCIONES ====================
    
    with st.sidebar:
        st.header("📋 ¿Cómo funciona?")
        st.markdown("""
        1. **Ajusta los ingredientes** usando los sliders
        2. **Visualiza el perfil** en el gráfico de radar
        3. **Lee la interpretación** automática
        4. **Compara** con el perfil ideal
        
        💡 **Tip:** Puedes guardar recetas para compararlas después.
        """)
        
        st.markdown("---")
        st.caption("📊 **Escala del perfil (1-9):**")
        st.caption("🟢 Alta: 7.0 - 9.0")
        st.caption("🟡 Media: 4.0 - 7.0")
        st.caption("🔴 Baja: 1.0 - 4.0")
    
    # ==================== CUERPO PRINCIPAL ====================
    
    # Explicación del Perfil Sensorial
    with st.expander("📖 ¿Qué es el Perfil Sensorial?", expanded=False):
        st.markdown("""
        ### El Perfil Sensorial describe cómo percibe una persona la gomita
        
        **Los 10 atributos que medimos (Escala 1-9):**
        
        | Atributo | Descripción | Bajo (1-3) | Medio (4-6) | Alto (7-9) |
        |----------|-------------|------------|-------------|------------|
        | **Brillo** | Percepción visual de brillo | Mate | Brillante | Muy brillante |
        | **Claridad** | Transparencia visual | Opaco | Transparente | Cristalino |
        | **Dulzura** | Intensidad del sabor dulce | Nada dulce | Dulce | Muy dulce |
        | **Acidez** | Intensidad del sabor ácido | Nada ácido | Ácido | Muy ácido |
        | **Adhesividad** | Grado en que se pega | No se pega | Se pega un poco | Muy pegajoso |
        | **Elasticidad** | Capacidad de recuperación | No elástico | Elástico | Muy elástico |
        | **Arenosidad** | Sensación de partículas | Smooth | Algo arenoso | Muy arenoso |
        | **Dureza** | Fuerza para morder | Muy suave | Firme | Muy duro |
        | **Gomosidad** | Sensación gomosa general | Sin gomosidad | Gomoso | Muy gomoso |
        | **Masticabilidad** | Energía para masticar | Fácil | Moderada | Difícil |
        """)
    
    # ==================== ENTRADA DE RECETA ====================
    
    st.subheader("🧪 Ajustar Receta")
    st.markdown("Mueve los sliders para crear tu receta ideal")
    
    # Dividir en dos columnas para los sliders
    col1, col2 = st.columns(2)
    
    # Valores por defecto (receta balanceada)
    default_receta = {
        'azucar_g': 5.0,
        'jarabe_glucosa_g': 4.0,
        'agua_g': 6.0,
        'gelatina_g': 0.8,
        'goma_xantica_g': 0.3,
        'acido_citrico_g': 0.3,
        'stevia_g': 0.5,
        'esencia_g': 0.3,
        'sorbitol_g': 2.0,
        'fibra_g': 1.0
    }
    
    # Sliders para ingredientes
    receta = {}
    
    with col1:
        st.markdown("#### 🍬 Base")
        
        receta['azucar_g'] = st.slider(
            "Azúcar (g)",
            min_value=0.0, max_value=12.0, value=default_receta['azucar_g'], step=0.1,
            help=descriptions['azucar_g']
        )
        
        receta['jarabe_glucosa_g'] = st.slider(
            "Jarabe de Glucosa (g)",
            min_value=0.0, max_value=8.0, value=default_receta['jarabe_glucosa_g'], step=0.1,
            help=descriptions['jarabe_glucosa_g']
        )
        
        receta['agua_g'] = st.slider(
            "Agua (g)",
            min_value=0.0, max_value=9.0, value=default_receta['agua_g'], step=0.1,
            help=descriptions['agua_g']
        )
        
        receta['gelatina_g'] = st.slider(
            "Gelatina (g)",
            min_value=0.0, max_value=1.5, value=default_receta['gelatina_g'], step=0.05,
            help=descriptions['gelatina_g']
        )
        
        receta['goma_xantica_g'] = st.slider(
            "Goma Xántica (g)",
            min_value=0.0, max_value=1.5, value=default_receta['goma_xantica_g'], step=0.05,
            help=descriptions['goma_xantica_g']
        )
    
    with col2:
        st.markdown("#### 🍭 Sabor y Aditivos")
        
        receta['acido_citrico_g'] = st.slider(
            "Ácido Cítrico (g)",
            min_value=0.0, max_value=0.7, value=default_receta['acido_citrico_g'], step=0.05,
            help=descriptions['acido_citrico_g']
        )
        
        receta['stevia_g'] = st.slider(
            "Stevia (g)",
            min_value=0.0, max_value=1.5, value=default_receta['stevia_g'], step=0.05,
            help=descriptions['stevia_g']
        )
        
        receta['esencia_g'] = st.slider(
            "Esencia (g)",
            min_value=0.0, max_value=0.7, value=default_receta['esencia_g'], step=0.05,
            help=descriptions['esencia_g']
        )
        
        receta['sorbitol_g'] = st.slider(
            "Sorbitol (g)",
            min_value=0.0, max_value=6.0, value=default_receta['sorbitol_g'], step=0.1,
            help=descriptions['sorbitol_g']
        )
        
        receta['fibra_g'] = st.slider(
            "Fibra (g)",
            min_value=0.0, max_value=3.0, value=default_receta['fibra_g'], step=0.05,
            help=descriptions['fibra_g']
        )
    
    # ==================== BOTÓN DE PREDICCIÓN ====================
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predecir = st.button(
            "🔮 PREDECIR PERFIL SENSORIAL",
            type="primary",
            use_container_width=True
        )
    
    # ==================== PREDECIR Y MOSTRAR ====================
    
    if predecir:
        with st.spinner("🔮 Prediciendo perfil sensorial..."):
            perfil = predecir_perfil(receta, modelos, escaladores, targets, features)
        
        # ==================== MOSTRAR RESULTADOS ====================
        
        st.success("✅ Perfil sensorial predicho")
        st.markdown("---")
        
        # Crear gráficos en columnas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de Radar (interactivo con Plotly) - Escala 1-9
            st.subheader("📈 Perfil Sensorial - Radar Chart")
            
            categories = [a.upper() for a in perfil.keys()]
            values = [perfil[a] for a in perfil.keys()]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Perfil de la receta',
                line=dict(color='#2E86AB', width=2),
                fillcolor='rgba(46, 134, 171, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 9],  # ← CAMBIADO: 0-9
                        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9]
                    )
                ),
                showlegend=True,
                height=500,
                title="Perfil Sensorial (Escala 1-9)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabla de valores - CORREGIDO
            st.subheader("📊 Valores")
            
            # Crear listas con la misma longitud
            atributos = [a.upper() for a in perfil.keys()]
            valores = [f"{perfil[a]:.2f}" for a in perfil.keys()]
            niveles = []
            
            for a in perfil.keys():
                valor = perfil[a]
                if valor >= 7:
                    niveles.append("🟢 Alta")
                elif valor >= 4:
                    niveles.append("🟡 Media")
                else:
                    niveles.append("🔴 Baja")
            
            df_perfil = pd.DataFrame({
                'Atributo': atributos,
                'Valor': valores,
                'Nivel': niveles
            })
            
            st.dataframe(df_perfil, use_container_width=True, hide_index=True)
            
            # Resumen rápido
            st.markdown("---")
            st.markdown("#### 📋 Resumen")
            
            alta = sum(1 for v in perfil.values() if v >= 7)
            media = sum(1 for v in perfil.values() if 4 <= v < 7)
            baja = sum(1 for v in perfil.values() if v < 4)
            
            st.metric("🟢 Atributos Altos (7-9)", alta)
            st.metric("🟡 Atributos Medios (4-6)", media)
            st.metric("🔴 Atributos Bajos (1-3)", baja)
        
        # ==================== GRÁFICO DE BARRAS ====================
        
        st.subheader("📊 Perfil Sensorial - Barras")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = [a.upper() for a in perfil.keys()]
        values = [perfil[a] for a in perfil.keys()]
        
        # Colores según nivel (escala 1-9)
        colors = []
        for v in values:
            if v >= 7:
                colors.append('#2ECC71')  # Verde (Alta)
            elif v >= 4:
                colors.append('#F1C40F')  # Amarillo (Media)
            else:
                colors.append('#E74C3C')  # Rojo (Baja)
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
        
        # Añadir valores
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylim(0, 9.5)  # ← CAMBIADO: 0-9.5
        ax.set_ylabel('Intensidad (1-9)', fontsize=12)
        ax.set_title('Perfil Sensorial - Barras', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Alta (≥7)')
        ax.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='Media (≥4)')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # ==================== INTERPRETACIÓN AUTOMÁTICA ====================
        
        st.subheader("💡 Interpretación Automática")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🍬 Dulzura")
            dulzura = perfil['dulzura']
            if dulzura >= 7:
                st.success(f"🟢 **Alta** ({dulzura:.2f})")
                st.caption("La gomita será muy dulce. Ideal para consumidores que prefieren sabores intensos.")
            elif dulzura >= 4:
                st.info(f"🟡 **Media** ({dulzura:.2f})")
                st.caption("Balance adecuado de dulzura. Aceptable para la mayoría de los consumidores.")
            else:
                st.warning(f"🔴 **Baja** ({dulzura:.2f})")
                st.caption("Poco dulce. Considera aumentar azúcar o stevia.")
        
        with col2:
            st.markdown("#### 🔗 Adhesividad")
            adhesividad = perfil['adhesividad']
            if adhesividad >= 7:
                st.warning(f"🔴 **Alta** ({adhesividad:.2f})")
                st.caption("Se pega a los dientes. Puede ser incómodo para algunos consumidores.")
            elif adhesividad >= 4:
                st.info(f"🟡 **Media** ({adhesividad:.2f})")
                st.caption("Adhesividad moderada. Textura aceptable.")
            else:
                st.success(f"🟢 **Baja** ({adhesividad:.2f})")
                st.caption("No se pega a los dientes. Excelente textura.")
        
        with col3:
            st.markdown("#### 💪 Dureza")
            dureza = perfil['dureza']
            if dureza >= 7:
                st.warning(f"🔴 **Alta** ({dureza:.2f})")
                st.caption("Gomita firme. Puede ser difícil de masticar para algunos.")
            elif dureza >= 4:
                st.info(f"🟡 **Media** ({dureza:.2f})")
                st.caption("Dureza balanceada. Textura agradable.")
            else:
                st.success(f"🟢 **Baja** ({dureza:.2f})")
                st.caption("Gomita suave. Fácil de masticar.")
        
        # ==================== RECOMENDACIONES ====================
        
        st.subheader("🎯 Recomendaciones para Mejorar")
        
        recomendaciones = []
        
        # Dulzura
        if perfil['dulzura'] < 4:
            recomendaciones.append("📈 **Aumenta la dulzura**: Incrementa azúcar o stevia. La gelatina también ayuda a liberar sabores.")
        
        # Adhesividad
        if perfil['adhesividad'] > 7:
            recomendaciones.append("📉 **Reduce la adhesividad**: Disminuye fibra o aumenta gelatina. La goma xántica también puede ayudar.")
        
        # Dureza
        if perfil['dureza'] > 7:
            recomendaciones.append("📉 **Suaviza la textura**: Reduce azúcar o fibra. Aumenta agua o gelatina.")
        elif perfil['dureza'] < 2:
            recomendaciones.append("📈 **Aumenta firmeza**: Incrementa azúcar, fibra o gelatina.")
        
        # Elasticidad
        if perfil['elasticidad'] < 4:
            recomendaciones.append("📈 **Aumenta elasticidad**: Incrementa azúcar o usa fibra Batata_Cáscara_5%.")
        
        # Arenosidad
        if perfil['arenosidad'] > 7:
            recomendaciones.append("📉 **Reduce arenosidad**: Disminuye fibra o stevia. Usa fibra más fina.")
        
        if recomendaciones:
            for rec in recomendaciones:
                st.info(rec)
        else:
            st.success("✅ ¡Perfil sensorial balanceado! No se requieren ajustes mayores.")
        
        # ==================== COMPARAR CON RECETA IDEAL ====================
        
        st.subheader("🎯 Comparar con Receta Ideal")
        
        # Perfil ideal (valores objetivos en escala 1-9)
        perfil_ideal = {
            'brillo': 7.0,
            'claridad': 6.0,
            'dulzura': 7.0,
            'acidez': 3.0,
            'adhesividad': 3.0,
            'elasticidad': 7.0,
            'arenosidad': 3.0,
            'dureza': 4.0,
            'gomosidad': 6.0,
            'masticabilidad': 5.0
        }
        
        # Calcular diferencias
        diferencias = {}
        for key in perfil.keys():
            diferencias[key] = perfil[key] - perfil_ideal[key]
        
        # Graficar comparación
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = [a.upper() for a in perfil.keys()]
        valores_actuales = [perfil[a] for a in perfil.keys()]
        valores_ideales = [perfil_ideal[a] for a in perfil.keys()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, valores_actuales, width, label='Tu receta', color='#2E86AB')
        bars2 = ax.bar(x + width/2, valores_ideales, width, label='Ideal', color='#A2D5C6')
        
        ax.set_xlabel('Atributo', fontsize=12)
        ax.set_ylabel('Intensidad (1-9)', fontsize=12)
        ax.set_title('Comparación: Tu Receta vs Perfil Ideal', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 9.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # ==================== GUARDAR RECETA ====================
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Guardar esta receta", use_container_width=True):
                if 'recetas_guardadas' not in st.session_state:
                    st.session_state['recetas_guardadas'] = []
                
                st.session_state['recetas_guardadas'].append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'receta': receta.copy(),
                    'perfil': perfil.copy(),
                    'tipo': 'manual'
                })
                
                st.success("✅ Receta guardada!")
        
        with col2:
            # Mostrar cantidad de recetas guardadas
            if 'recetas_guardadas' in st.session_state:
                st.info(f"📋 Recetas guardadas: {len(st.session_state['recetas_guardadas'])}")
    
    else:
        # Mostrar ejemplo cuando no se ha predicho
        st.info("👆 Ajusta los ingredientes y presiona 'PREDECIR PERFIL SENSORIAL'")
        
        # Mostrar receta actual
        st.markdown("#### 📋 Receta actual")
        df_receta_actual = pd.DataFrame({
            'Ingrediente': [descriptions.get(ing, ing) for ing in default_receta.keys()],
            'Cantidad (g)': [f"{default_receta[ing]:.2f}" for ing in default_receta.keys()]
        })
        st.dataframe(df_receta_actual, use_container_width=True, hide_index=True)

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    print("="*60)
    print("📊 MÓDULO 4: PERFIL SENSORIAL (ESCALA 1-9)")
    print("="*60)
    print("\nEste módulo se ejecuta dentro de Streamlit.")
    print("Ejecuta: streamlit run dashboard/pipeline_app.py")