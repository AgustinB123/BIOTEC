"""
👥 MÓDULO 5: PREDICCIÓN DE ACEPTABILIDAD
Autor: Proyecto Gomitas IA
Fecha: 2026-08-25
Descripción: Predice la aceptación del consumidor basado en el perfil sensorial
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
        'modelo_aceptabilidad': models_dir / 'modelo_aceptabilidad.pkl',
        'modelos_sensoriales': models_dir / 'modelos_sensoriales.pkl',
        'escaladores': models_dir / 'escaladores.pkl',
        'features': models_dir / 'features.pkl'
    }

def cargar_modelos():
    """Carga los modelos entrenados"""
    paths = get_paths()
    
    try:
        with open(paths['modelo_aceptabilidad'], 'rb') as f:
            modelo_aceptabilidad = pickle.load(f)
        with open(paths['modelos_sensoriales'], 'rb') as f:
            modelos_sensoriales = pickle.load(f)
        with open(paths['escaladores'], 'rb') as f:
            escaladores = pickle.load(f)
        with open(paths['features'], 'rb') as f:
            features = pickle.load(f)
        return modelo_aceptabilidad, modelos_sensoriales, escaladores, features, None
    except Exception as e:
        return None, None, None, None, str(e)

# ==================== PREDECIR PERFIL ====================

def predecir_perfil(receta, modelos_sensoriales, escaladores, targets, features):
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
            predicciones[target] = modelos_sensoriales[target].predict(X_scaled)[0]
        except:
            predicciones[target] = 0
    
    return predicciones

# ==================== PREDECIR ACEPTABILIDAD ====================

def predecir_aceptabilidad(perfil, modelo_aceptabilidad):
    """
    Predice la aceptación basado en el perfil sensorial
    
    El modelo de aceptabilidad usa las siguientes características:
    - edad_num: edad codificada (1-8)
    - sexo_num: sexo codificado (0=Femenino, 1=Masculino, 2=Otro)
    - fuma_num: fuma codificado (0=No, 1=A veces, 2=Si)
    - apariencia_num: apariencia (1-5)
    - color_num: color (1-5)
    - sabor_num: sabor (1-5)
    - sensacion_num: sensación (1-5)
    
    Para una predicción general, usamos valores promedio de consumidores:
    - Edad: 25-34 años (4)
    - Sexo: Femenino (0)
    - Fuma: No (0)
    """
    # Perfil sensorial → atributos de aceptabilidad (escala 1-9 → 1-5)
    # Mapeo de escala 1-9 a 1-5
    def mapear_escala(valor, origen_min=1, origen_max=9, destino_min=1, destino_max=5):
        """Mapea un valor de escala 1-9 a escala 1-5"""
        return destino_min + (valor - origen_min) * (destino_max - destino_min) / (origen_max - origen_min)
    
    # Calcular atributos de aceptabilidad (escala 1-5)
    apariencia = mapear_escala((perfil.get('brillo', 3) + perfil.get('claridad', 3)) / 2)
    color = mapear_escala(perfil.get('brillo', 3))
    sabor = mapear_escala(perfil.get('dulzura', 3))
    sensacion = mapear_escala((perfil.get('elasticidad', 3) + perfil.get('gomosidad', 3) + (7 - perfil.get('adhesividad', 3))) / 3)
    
    # Características demográficas promedio (consumidor típico)
    X = np.array([[
        4,  # edad_num: 25-34 años
        0,  # sexo_num: Femenino
        0,  # fuma_num: No fuma
        apariencia,
        color,
        sabor,
        sensacion
    ]])
    
    # Predecir
    try:
        prediccion = modelo_aceptabilidad.predict(X)[0]
        probabilidades = modelo_aceptabilidad.predict_proba(X)[0]
        
        # Mapeo de clases
        clases = {0: 'Baja', 1: 'Media', 2: 'Alta', 3: 'Muy Alta'}
        clase_predicha = clases.get(prediccion, 'Media')
        
        # Probabilidad de la clase predicha
        prob_clase = probabilidades[prediccion] if prediccion < len(probabilidades) else 0.5
        
        return clase_predicha, prob_clase, probabilidades
    except:
        return 'Media', 0.5, [0, 0.5, 0.5, 0]

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
        'azucar_g': '🍬 Azúcar: Aporta dulzura y cuerpo. Es el principal edulcorante.',
        'jarabe_glucosa_g': '🍯 Jarabe de Glucosa: Evita la cristalización, aporta humedad y textura suave.',
        'agua_g': '💧 Agua: Base de la formulación. Disuelve los ingredientes y da humedad.',
        'gelatina_g': '🧪 Gelatina: Da la textura gomosa y elasticidad. Es la clave de la gomosidad.',
        'goma_xantica_g': '🌿 Goma Xántica: Estabilizador natural que mejora la textura y retiene agua.',
        'acido_citrico_g': '🍋 Ácido Cítrico: Aporta acidez y realza los sabores. Da el toque ácido.',
        'stevia_g': '🌱 Stevia: Edulcorante natural. Es 200-300 veces más dulce que el azúcar.',
        'esencia_g': '🌸 Esencia: Aporta sabor y aroma. Define el perfil de sabor de la gomita.',
        'sorbitol_g': '🍬 Sorbitol: Edulcorante que aporta humedad y previene el endurecimiento.',
        'fibra_g': '🌾 Fibra: Aporta beneficios saludables y modifica la textura. Aumenta la adhesividad.'
    }

# ==================== INTERFAZ DE STREAMLIT ====================

def app_prediccion_aceptabilidad():
    """Aplicación principal de Predicción de Aceptabilidad"""
    
    st.title("👥 Predicción de Aceptabilidad")
    st.markdown("---")
    
    # Verificar que hay modelos entrenados
    if not st.session_state.get('modelos_entrenados', False):
        st.warning("⚠️ Primero debes entrenar los modelos en el Módulo 2")
        if st.button("📂 Ir a Entrenamiento"):
            st.session_state['modulo_actual'] = 'entrenamiento'
            st.rerun()
        return
    
    # Cargar modelos
    modelo_aceptabilidad, modelos_sensoriales, escaladores, features, error = cargar_modelos()
    
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
        1. **Ajusta los ingredientes** o usa los valores predeterminados
        2. **El sistema predice** el perfil sensorial (escala 1-9)
        3. **El modelo de aceptabilidad** estima la aceptación del consumidor
        4. **Recibe recomendaciones** para mejorar la aceptación
        
        💡 **El modelo de aceptabilidad** fue entrenado con datos de consumidores reales y sintéticos.
        """)
        
        st.markdown("---")
        st.caption("📊 **Niveles de aceptación:**")
        st.caption("🟢 Muy Alta: 4.5 - 5.0")
        st.caption("🟢 Alta: 4.0 - 4.5")
        st.caption("🟡 Media: 3.0 - 4.0")
        st.caption("🔴 Baja: 1.0 - 3.0")
        
        st.markdown("---")
        st.caption("📊 **Escala de atributos: 1-9**")
    
    # ==================== CUERPO PRINCIPAL ====================
    
    # Explicación de Aceptabilidad
    with st.expander("📖 ¿Qué es la Aceptabilidad?", expanded=False):
        st.markdown("""
        ### La Aceptabilidad mide qué tan probable es que un consumidor compre y repita el producto
        
        **Factores que influyen en la aceptación:**
        
        | Factor | Influencia | ¿Cómo mejorarlo? |
        |--------|------------|------------------|
        | **Color** | 24% | Usar colores atractivos y naturales |
        | **Sensación en boca** | 24% | Textura suave y agradable |
        | **Sabor** | 22% | Balance entre dulce y ácido |
        | **Apariencia** | 16% | Presentación atractiva |
        | **Edad** | 10% | Adaptar a cada segmento |
        | **Sexo** | 3% | Diferencias menores |
        | **Fuma** | 2% | Impacto mínimo |
        
        **Estos porcentajes vienen de nuestro modelo entrenado con 383 respuestas de consumidores.** 🎯
        
        **Escala de atributos sensoriales: 1-9**
        """)
    
    # ==================== ENTRADA DE RECETA ====================
    
    st.subheader("🧪 Receta de la Gomita")
    st.markdown("Ajusta los ingredientes para ver cómo afecta la aceptación")
    
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
            help=descriptions['azucar_g'],
            key="acep_azucar"
        )
        
        receta['jarabe_glucosa_g'] = st.slider(
            "Jarabe de Glucosa (g)",
            min_value=0.0, max_value=8.0, value=default_receta['jarabe_glucosa_g'], step=0.1,
            help=descriptions['jarabe_glucosa_g'],
            key="acep_jarabe"
        )
        
        receta['agua_g'] = st.slider(
            "Agua (g)",
            min_value=0.0, max_value=9.0, value=default_receta['agua_g'], step=0.1,
            help=descriptions['agua_g'],
            key="acep_agua"
        )
        
        receta['gelatina_g'] = st.slider(
            "Gelatina (g)",
            min_value=0.0, max_value=1.5, value=default_receta['gelatina_g'], step=0.05,
            help=descriptions['gelatina_g'],
            key="acep_gelatina"
        )
        
        receta['goma_xantica_g'] = st.slider(
            "Goma Xántica (g)",
            min_value=0.0, max_value=1.5, value=default_receta['goma_xantica_g'], step=0.05,
            help=descriptions['goma_xantica_g'],
            key="acep_goma"
        )
    
    with col2:
        st.markdown("#### 🍭 Sabor y Aditivos")
        
        receta['acido_citrico_g'] = st.slider(
            "Ácido Cítrico (g)",
            min_value=0.0, max_value=0.7, value=default_receta['acido_citrico_g'], step=0.05,
            help=descriptions['acido_citrico_g'],
            key="acep_acido"
        )
        
        receta['stevia_g'] = st.slider(
            "Stevia (g)",
            min_value=0.0, max_value=1.5, value=default_receta['stevia_g'], step=0.05,
            help=descriptions['stevia_g'],
            key="acep_stevia"
        )
        
        receta['esencia_g'] = st.slider(
            "Esencia (g)",
            min_value=0.0, max_value=0.7, value=default_receta['esencia_g'], step=0.05,
            help=descriptions['esencia_g'],
            key="acep_esencia"
        )
        
        receta['sorbitol_g'] = st.slider(
            "Sorbitol (g)",
            min_value=0.0, max_value=6.0, value=default_receta['sorbitol_g'], step=0.1,
            help=descriptions['sorbitol_g'],
            key="acep_sorbitol"
        )
        
        receta['fibra_g'] = st.slider(
            "Fibra (g)",
            min_value=0.0, max_value=3.0, value=default_receta['fibra_g'], step=0.05,
            help=descriptions['fibra_g'],
            key="acep_fibra"
        )
    
    # ==================== DEMOGRAFÍA ====================
    
    st.subheader("👤 Perfil del Consumidor")
    st.markdown("Selecciona el perfil del consumidor para una predicción más precisa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.selectbox(
            "Rango de Edad",
            options=['6 a 12 años', 'De 13 a 17 años', '18 a 24 años', 
                    '25 a 34 años', '35 a 44 años', '45 a 54 años', 
                    '55 a 64 años', 'Más de 65 años'],
            index=3,
            key="acep_edad"
        )
    
    with col2:
        sexo = st.selectbox(
            "Sexo",
            options=['Femenino', 'Masculino', 'Otro'],
            index=0,
            key="acep_sexo"
        )
    
    with col3:
        fuma = st.selectbox(
            "¿Fuma?",
            options=['No', 'A veces', 'Si'],
            index=0,
            key="acep_fuma"
        )
    
    # ==================== BOTÓN DE PREDICCIÓN ====================
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predecir = st.button(
            "🔮 PREDECIR ACEPTABILIDAD",
            type="primary",
            use_container_width=True
        )
    
    # ==================== PREDECIR Y MOSTRAR ====================
    
    if predecir:
        with st.spinner("🔮 Prediciendo perfil sensorial y aceptabilidad..."):
            # 1. Predecir perfil sensorial
            perfil = predecir_perfil(receta, modelos_sensoriales, escaladores, targets, features)
            
            # 2. Predecir aceptabilidad
            # Mapear edad a número
            edad_map = {
                '6 a 12 años': 1, 'De 13 a 17 años': 2, '18 a 24 años': 3,
                '25 a 34 años': 4, '35 a 44 años': 5, '45 a 54 años': 6,
                '55 a 64 años': 7, 'Más de 65 años': 8
            }
            sexo_map = {'Femenino': 0, 'Masculino': 1, 'Otro': 2}
            fuma_map = {'No': 0, 'A veces': 1, 'Si': 2}
            
            # Usar el perfil sensorial para predecir aceptabilidad
            # Mapeo de escala 1-9 a 1-5
            def mapear_escala(valor, origen_min=1, origen_max=9, destino_min=1, destino_max=5):
                return destino_min + (valor - origen_min) * (destino_max - destino_min) / (origen_max - origen_min)
            
            apariencia = mapear_escala((perfil.get('brillo', 3) + perfil.get('claridad', 3)) / 2)
            color = mapear_escala(perfil.get('brillo', 3))
            sabor = mapear_escala(perfil.get('dulzura', 3))
            sensacion = mapear_escala((perfil.get('elasticidad', 3) + perfil.get('gomosidad', 3) + (7 - perfil.get('adhesividad', 3))) / 3)
            
            # Crear vector de características
            X_acept = np.array([[
                edad_map[edad],
                sexo_map[sexo],
                fuma_map[fuma],
                apariencia,
                color,
                sabor,
                sensacion
            ]])
            
            # Predecir
            try:
                prediccion = modelo_aceptabilidad.predict(X_acept)[0]
                probabilidades = modelo_aceptabilidad.predict_proba(X_acept)[0]
                
                clases = {0: 'Baja', 1: 'Media', 2: 'Alta', 3: 'Muy Alta'}
                clase_predicha = clases.get(prediccion, 'Media')
                probabilidad = probabilidades[prediccion] if prediccion < len(probabilidades) else 0.5
                
                # Score de aceptación (0-100)
                score_aceptacion = probabilidad * 100
                
                # Determinar emoji y color
                if clase_predicha == 'Muy Alta':
                    emoji = "🌟"
                    color_acept = "#2ECC71"
                    descripcion = "Excelente! La gente va a amar esta gomita."
                elif clase_predicha == 'Alta':
                    emoji = "😊"
                    color_acept = "#27AE60"
                    descripcion = "Muy buena aceptación. La mayoría de la gente va a disfrutarla."
                elif clase_predicha == 'Media':
                    emoji = "😐"
                    color_acept = "#F1C40F"
                    descripcion = "Aceptación promedio. Puede mejorar ajustando la receta."
                else:
                    emoji = "😟"
                    color_acept = "#E74C3C"
                    descripcion = "Baja aceptación. Se recomienda revisar la receta."
                
            except:
                clase_predicha = 'Media'
                probabilidad = 0.5
                score_aceptacion = 50
                emoji = "😐"
                color_acept = "#F1C40F"
                descripcion = "No se pudo calcular con precisión. Usando valor estimado."
        
        # ==================== MOSTRAR RESULTADOS ====================
        
        st.success("✅ Predicción completada")
        st.markdown("---")
        
        # ==================== TARJETA DE ACEPTABILIDAD ====================
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color_acept}20; border: 2px solid {color_acept};">
                <h1 style="font-size: 48px; margin: 0;">{emoji}</h1>
                <h2 style="margin: 0; color: {color_acept};">{clase_predicha.upper()}</h2>
                <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{score_aceptacion:.1f}%</p>
                <p style="margin: 5px 0;">{descripcion}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ==================== DETALLES DE ACEPTABILIDAD ====================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribución de Probabilidades")
            
            # Crear gráfico de barras con las probabilidades
            clases_nombres = ['Baja', 'Media', 'Alta', 'Muy Alta']
            colores = ['#E74C3C', '#F1C40F', '#27AE60', '#2ECC71']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(clases_nombres, probabilidades, color=colores, edgecolor='black', linewidth=1)
            
            # Añadir valores
            for bar, val in zip(bars, probabilidades):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Probabilidad', fontsize=12)
            ax.set_title('Probabilidad de cada nivel de aceptación', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("📊 Factores de Aceptación")
            
            # Mostrar factores que influyen en la aceptación
            factores = {
                'Color': 0.24,
                'Sensación en boca': 0.24,
                'Sabor': 0.22,
                'Apariencia': 0.16,
                'Edad': 0.10,
                'Sexo': 0.03,
                'Fuma': 0.01
            }
            
            # Crear gráfico de barras horizontales
            fig, ax = plt.subplots(figsize=(8, 5))
            
            factores_ordenados = sorted(factores.items(), key=lambda x: x[1], reverse=True)
            nombres = [f[0] for f in factores_ordenados]
            valores = [f[1] for f in factores_ordenados]
            colores = ['#2E86AB', '#A2D5C6', '#F5B041', '#F7DC6F', '#85C1E9', '#AED6F1', '#D5D8DC']
            
            bars = ax.barh(nombres, valores, color=colores, edgecolor='black', linewidth=0.5)
            
            for bar, val in zip(bars, valores):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.0%}', va='center', fontsize=9)
            
            ax.set_xlabel('Influencia en la aceptación', fontsize=12)
            ax.set_title('Factores que influyen en la aceptación', fontsize=14)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # ==================== PERFIL SENSORIAL ====================
        
        st.subheader("📊 Perfil Sensorial de la Receta (Escala 1-9)")
        
        # Radar chart con Plotly - Escala 1-9
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
                    range=[0, 9],
                    tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9]
                )
            ),
            showlegend=True,
            height=450,
            title="Perfil Sensorial (Escala 1-9)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==================== RECOMENDACIONES ====================
        
        st.subheader("🎯 Recomendaciones para Mejorar la Aceptación")
        
        # Generar recomendaciones basadas en el perfil (escala 1-9)
        recomendaciones = []
        
        # Color / Brillo
        if perfil.get('brillo', 0) < 4:
            recomendaciones.append("📈 **Aumenta el brillo**: Incrementa la fibra o usa colorantes que den brillo. El brillo atrae visualmente.")
        
        # Sabor / Dulzura
        if perfil.get('dulzura', 0) < 4:
            recomendaciones.append("📈 **Aumenta la dulzura**: Incrementa azúcar o stevia. La gelatina también ayuda a liberar sabores.")
        elif perfil.get('dulzura', 0) > 7:
            recomendaciones.append("📉 **Reduce la dulzura**: Demasiado dulce puede empalagar. Reduce azúcar o stevia.")
        
        # Sensación en boca (elasticidad + gomosidad + adhesividad)
        sensacion = (perfil.get('elasticidad', 3) + perfil.get('gomosidad', 3)) / 2
        if sensacion < 4:
            recomendaciones.append("📈 **Mejora la sensación en boca**: Aumenta gelatina o azúcar para una textura más gomosa.")
        if perfil.get('adhesividad', 0) > 6:
            recomendaciones.append("📉 **Reduce la adhesividad**: Disminuye fibra o aumenta gelatina. La goma xántica también puede ayudar.")
        
        # Apariencia (brillo + claridad)
        apariencia = (perfil.get('brillo', 3) + perfil.get('claridad', 3)) / 2
        if apariencia < 4:
            recomendaciones.append("📈 **Mejora la apariencia**: Aumenta el brillo y la claridad para que se vea más atractiva.")
        
        # Acidez
        if perfil.get('acidez', 0) > 6:
            recomendaciones.append("📉 **Reduce la acidez**: Disminuye ácido cítrico. Demasiado ácido puede no gustar.")
        
        if not recomendaciones:
            recomendaciones.append("✅ ¡El perfil de tu receta está bien balanceado! La aceptación debería ser buena.")
        
        for rec in recomendaciones:
            if rec.startswith("✅"):
                st.success(rec)
            else:
                st.info(rec)
        
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
                    'aceptabilidad': {
                        'clase': clase_predicha,
                        'score': score_aceptacion,
                        'probabilidad': probabilidad
                    },
                    'tipo': 'aceptabilidad'
                })
                
                st.success("✅ Receta guardada!")
        
        with col2:
            if 'recetas_guardadas' in st.session_state:
                st.info(f"📋 Recetas guardadas: {len(st.session_state['recetas_guardadas'])}")
    
    else:
        # Mostrar ejemplo cuando no se ha predicho
        st.info("👆 Ajusta los ingredientes y presiona 'PREDECIR ACEPTABILIDAD'")
        
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
    print("👥 MÓDULO 5: PREDICCIÓN DE ACEPTABILIDAD")
    print("="*60)
    print("\nEste módulo se ejecuta dentro de Streamlit.")
    print("Ejecuta: streamlit run dashboard/pipeline_app.py")
    print("\n📊 ESCALA DE ATRIBUTOS: 1-9")