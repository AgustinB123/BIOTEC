"""
🧬 RECOMENDADOR DE RECETAS - CONTROL TOTAL
Autor: Proyecto Gomitas IA
Fecha: 2026-08-25
Descripción: Algoritmo Genético con control completo sobre maximizar/minimizar/restricciones
ESCALA: 1-9
"""

import os
import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
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
    """Descripciones educativas de cada ingrediente"""
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

# ==================== DESCRIPCIÓN DE ATRIBUTOS ====================

def get_atributo_descriptions():
    """Descripciones educativas de cada atributo sensorial"""
    return {
        'brillo': {
            'nombre': '✨ Brillo',
            'descripcion': 'Percepción visual de brillo en la superficie de la gomita',
            'bajo': 'Mate, sin brillo',
            'medio': 'Brillo moderado',
            'alto': 'Muy brillante, atractivo visualmente'
        },
        'claridad': {
            'nombre': '💎 Claridad',
            'descripcion': 'Transparencia o claridad visual de la gomita',
            'bajo': 'Opaco, turbio',
            'medio': 'Algo transparente',
            'alto': 'Cristalino, muy transparente'
        },
        'dulzura': {
            'nombre': '🍬 Dulzura',
            'descripcion': 'Intensidad del sabor dulce percibido',
            'bajo': 'Poco dulce, insípido',
            'medio': 'Dulzura balanceada',
            'alto': 'Muy dulce, intenso'
        },
        'acidez': {
            'nombre': '🍋 Acidez',
            'descripcion': 'Intensidad del sabor ácido percibido',
            'bajo': 'Nada ácido, suave',
            'medio': 'Acidez balanceada',
            'alto': 'Muy ácido, intenso'
        },
        'adhesividad': {
            'nombre': '🔗 Adhesividad',
            'descripcion': 'Grado en que la gomita se pega a los dientes',
            'bajo': 'No se pega, textura limpia',
            'medio': 'Se pega un poco',
            'alto': 'Muy pegajoso, incómodo'
        },
        'elasticidad': {
            'nombre': '🌀 Elasticidad',
            'descripcion': 'Capacidad de recuperación después de morder',
            'bajo': 'No elástico, se rompe fácilmente',
            'medio': 'Elástico moderado',
            'alto': 'Muy elástico, se estira y vuelve'
        },
        'arenosidad': {
            'nombre': '🏖️ Arenosidad',
            'descripcion': 'Sensación de partículas o granos al masticar',
            'bajo': 'Textura suave, sin granos',
            'medio': 'Algo arenoso',
            'alto': 'Muy arenoso, granuloso'
        },
        'dureza': {
            'nombre': '💪 Dureza',
            'descripcion': 'Fuerza necesaria para morder la gomita',
            'bajo': 'Muy suave, se muerde fácilmente',
            'medio': 'Firmeza moderada',
            'alto': 'Muy duro, difícil de morder'
        },
        'gomosidad': {
            'nombre': '🧸 Gomosidad',
            'descripcion': 'Sensación gomosa general al masticar',
            'bajo': 'Sin gomosidad, textura blanda',
            'medio': 'Gomosidad moderada',
            'alto': 'Muy gomoso, textura característica'
        },
        'masticabilidad': {
            'nombre': '🦷 Masticabilidad',
            'descripcion': 'Energía necesaria para masticar completamente',
            'bajo': 'Fácil de masticar, se deshace rápido',
            'medio': 'Masticación moderada',
            'alto': 'Difícil de masticar, requiere esfuerzo'
        }
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

# ==================== ALGORITMO GENÉTICO ====================

class RecomendadorGeneticoCompleto:
    """Algoritmo Genético con control total sobre maximizar/minimizar"""
    
    def __init__(self, modelos, escaladores, targets, features, limits):
        self.models = modelos
        self.scalers = escaladores
        self.targets = targets
        self.features = features
        self.limits = limits
        self.ingredient_names = list(limits.keys())
        self.historial = []
        self.mejores_puntuaciones = []
        self.diversidad = []
    
    def crear_individuo(self):
        """Crea una receta aleatoria"""
        receta = {}
        for ingrediente, (min_val, max_val) in self.limits.items():
            receta[ingrediente] = round(random.uniform(min_val, max_val), 3)
        return receta
    
    def evaluar_individuo(self, receta, configuracion):
        """
        Evalúa una receta según la configuración del usuario
        """
        perfil = predecir_perfil(receta, self.models, self.scalers, self.targets, self.features)
        
        puntuacion = 0
        
        # Evaluar cada atributo según su configuración
        for attr, config in configuracion.items():
            if attr not in perfil or attr == 'restricciones':
                continue
            
            valor = perfil[attr]
            accion = config.get('accion', 'ignorar')
            peso = config.get('peso', 0.0)
            
            if accion == 'ignorar' or peso == 0:
                continue
            
            # Normalizar valor a 0-1 (escala 1-9)
            valor_normalizado = (valor - 1) / 8.0
            
            if accion == 'maximizar':
                contribucion = valor_normalizado * peso
            elif accion == 'minimizar':
                contribucion = (1 - valor_normalizado) * peso
            else:
                continue
            
            puntuacion += contribucion
        
        # Penalizaciones por restricciones
        penalizacion = 0
        if 'restricciones' in configuracion:
            for attr, (min_val, max_val) in configuracion['restricciones'].items():
                if attr in perfil:
                    if perfil[attr] < min_val:
                        penalizacion += (min_val - perfil[attr]) * 2
                    if perfil[attr] > max_val:
                        penalizacion += (perfil[attr] - max_val) * 2
        
        puntuacion -= penalizacion
        
        return puntuacion, perfil
    
    def seleccionar_padres(self, poblacion, fitness):
        """Selección por torneo"""
        padres = []
        for _ in range(2):
            indices = random.sample(range(len(poblacion)), min(4, len(poblacion)))
            mejor_idx = max(indices, key=lambda i: fitness[i] + random.uniform(0, 0.1))
            padres.append(poblacion[mejor_idx])
        return padres
    
    def cruzar_adaptativo(self, padre1, padre2, generacion, generaciones_totales):
        """Cruza con mutación adaptativa"""
        hijo = {}
        
        tasa_mutacion = 0.4 * (1 - generacion / generaciones_totales) + 0.05
        
        for ingrediente in self.ingredient_names:
            if random.random() < 0.5:
                hijo[ingrediente] = padre1[ingrediente]
            else:
                hijo[ingrediente] = padre2[ingrediente]
            
            if random.random() < tasa_mutacion:
                min_val, max_val = self.limits[ingrediente]
                amplitud = (max_val - min_val) * 0.4 * (1 - generacion / generaciones_totales + 0.05)
                mutacion = random.uniform(-amplitud, amplitud)
                hijo[ingrediente] = max(min_val, min(max_val, hijo[ingrediente] + mutacion))
            
            hijo[ingrediente] = round(hijo[ingrediente], 3)
        
        return hijo
    
    def recomendar(self, configuracion, poblacion_size=150, generaciones=80, elitismo=0.15, verbose=True):
        """Ejecuta el algoritmo genético"""
        
        if verbose:
            st.markdown("### 🧬 Ejecutando Algoritmo Genético")
            st.markdown(f"**Población:** {poblacion_size} recetas")
            st.markdown(f"**Generaciones:** {generaciones}")
            st.markdown(f"**Elitismo:** {elitismo*100:.0f}%")
        
        poblacion = [self.crear_individuo() for _ in range(poblacion_size)]
        
        self.historial = []
        self.mejores_puntuaciones = []
        self.diversidad = []
        
        mejor_receta = None
        mejor_puntuacion = -float('inf')
        mejor_perfil = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for gen in range(generaciones):
            fitness = []
            perfiles = []
            
            for individuo in poblacion:
                puntuacion, perfil = self.evaluar_individuo(individuo, configuracion)
                fitness.append(puntuacion)
                perfiles.append(perfil)
            
            mejor_idx = np.argmax(fitness)
            gen_mejor_puntuacion = fitness[mejor_idx]
            self.mejores_puntuaciones.append(gen_mejor_puntuacion)
            
            fitness_std = np.std(fitness)
            self.diversidad.append(fitness_std)
            
            if gen_mejor_puntuacion > mejor_puntuacion:
                mejor_puntuacion = gen_mejor_puntuacion
                mejor_receta = poblacion[mejor_idx].copy()
                mejor_perfil = perfiles[mejor_idx].copy()
            
            if verbose:
                status_text.text(f"🧬 Gen {gen}: Mejor = {gen_mejor_puntuacion:.3f} | Diversidad = {fitness_std:.3f}")
            progress_bar.progress((gen + 1) / generaciones)
            
            nueva_poblacion = []
            
            num_elite = int(poblacion_size * elitismo)
            elite_indices = np.argsort(fitness)[-num_elite:]
            for idx in elite_indices:
                nueva_poblacion.append(poblacion[idx].copy())
            
            while len(nueva_poblacion) < poblacion_size:
                padres = self.seleccionar_padres(poblacion, fitness)
                hijo = self.cruzar_adaptativo(padres[0], padres[1], gen, generaciones)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        progress_bar.empty()
        status_text.empty()
        
        if verbose:
            st.success(f"✅ Mejor puntuación: {mejor_puntuacion:.3f}")
        
        return mejor_receta, mejor_perfil, mejor_puntuacion
    
    def graficar_evolucion(self):
        """Genera gráficos de la evolución"""
        if not self.mejores_puntuaciones:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(self.mejores_puntuaciones, 'b-', linewidth=2)
        ax1.set_xlabel('Generación', fontsize=12)
        ax1.set_ylabel('Mejor Puntuación', fontsize=12)
        ax1.set_title('📈 Evolución de la Puntuación', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.mejores_puntuaciones[-1], color='g', linestyle='--', alpha=0.5, label='Final')
        
        ax2.plot(self.diversidad, 'g-', linewidth=2)
        ax2.set_xlabel('Generación', fontsize=12)
        ax2.set_ylabel('Diversidad', fontsize=12)
        ax2.set_title('🧬 Diversidad de la Población', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ==================== INTERFAZ DE STREAMLIT ====================

def app_recomendador():
    """Aplicación principal del Recomendador de Recetas"""
    
    st.title("🧬 Recomendador de Recetas")
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
    attr_descriptions = get_atributo_descriptions()
    
    # ==================== SIDEBAR: CONFIGURACIÓN ====================
    
    with st.sidebar:
        st.header("⚙️ Configuración del Algoritmo")
        
        poblacion_size = st.slider(
            "📊 Tamaño de Población",
            min_value=50, max_value=300, value=150, step=10,
            help="Número de recetas que se prueban en cada generación"
        )
        
        generaciones = st.slider(
            "🔄 Número de Generaciones",
            min_value=20, max_value=150, value=80, step=10,
            help="Ciclos de evolución. Más = mejor convergencia"
        )
        
        elitismo = st.slider(
            "🏆 Elitismo (%)",
            min_value=5, max_value=30, value=15, step=5,
            help="Porcentaje de mejores recetas que pasan intactas"
        )
        
        st.markdown("---")
        st.caption("📊 **Escala de atributos: 1-9**")
        st.caption("🟢 Alta: 7-9 | 🟡 Media: 4-6 | 🔴 Baja: 1-3")
    
    # ==================== CONFIGURACIÓN DE ATRIBUTOS ====================
    
    st.subheader("🎯 Configurar Objetivo de la Receta")
    st.markdown("""
    Para **cada atributo**, elige:
    - **Acción**: Maximizar (que sea ALTO), Minimizar (que sea BAJO), o Ignorar
    - **Peso**: Importancia (0 = no importa, 1 = muy importante)
    """)
    
    # Diccionario para almacenar la configuración
    configuracion = {}
    
    # Mostrar cada atributo con sus controles
    for target in targets:
        info = attr_descriptions.get(target, {})
        nombre = info.get('nombre', target.upper())
        descripcion = info.get('descripcion', '')
        
        with st.expander(f"{nombre} - {descripcion}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Acción: Maximizar, Minimizar o Ignorar
                accion = st.selectbox(
                    "Acción",
                    options=['Maximizar (↑)', 'Minimizar (↓)', 'Ignorar (—)'],
                    index=0,
                    key=f"accion_{target}",
                    help=f"¿Qué quieres hacer con {nombre}?"
                )
                
                # Mapear acción a valor interno
                if accion == 'Maximizar (↑)':
                    accion_interna = 'maximizar'
                elif accion == 'Minimizar (↓)':
                    accion_interna = 'minimizar'
                else:
                    accion_interna = 'ignorar'
            
            with col2:
                # Peso (importancia)
                peso = st.slider(
                    "Peso",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                    key=f"peso_{target}",
                    help="0 = no importa, 1 = muy importante",
                    disabled=(accion_interna == 'ignorar')
                )
                if accion_interna == 'ignorar':
                    peso = 0.0
            
            with col3:
                # Mostrar referencia de escala
                st.caption(f"📊 **{nombre}**")
                st.caption(f"🔴 {info.get('bajo', 'Bajo')}")
                st.caption(f"🟡 {info.get('medio', 'Medio')}")
                st.caption(f"🟢 {info.get('alto', 'Alto')}")
            
            # Guardar configuración
            configuracion[target] = {
                'accion': accion_interna,
                'peso': peso
            }
    
    # ==================== RESTRICCIONES ====================
    
    st.subheader("🔒 Restricciones (Opcional)")
    st.markdown("""
    Establece rangos mínimos y máximos para los atributos.
    Las recetas que no cumplan estas restricciones serán penalizadas.
    """)
    
    # Crear pestañas para organizar las restricciones
    tabs = st.tabs(["🍬 Sabor", "📊 Textura", "👀 Apariencia"])
    
    restricciones = {}
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🍬 Dulzura**")
            dulzura_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="dulz_min")
            dulzura_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="dulz_max")
            if dulzura_min > 1.0 or dulzura_max < 9.0:
                restricciones['dulzura'] = (dulzura_min, dulzura_max)
            
            st.markdown("**🍋 Acidez**")
            acidez_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="acid_min")
            acidez_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="acid_max")
            if acidez_min > 1.0 or acidez_max < 9.0:
                restricciones['acidez'] = (acidez_min, acidez_max)
        
        with col2:
            st.markdown("**🔗 Adhesividad**")
            adhesividad_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="adh_min")
            adhesividad_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="adh_max")
            if adhesividad_min > 1.0 or adhesividad_max < 9.0:
                restricciones['adhesividad'] = (adhesividad_min, adhesividad_max)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌀 Elasticidad**")
            elasticidad_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="elast_min")
            elasticidad_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="elast_max")
            if elasticidad_min > 1.0 or elasticidad_max < 9.0:
                restricciones['elasticidad'] = (elasticidad_min, elasticidad_max)
            
            st.markdown("**🧸 Gomosidad**")
            gomosidad_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="gom_min")
            gomosidad_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="gom_max")
            if gomosidad_min > 1.0 or gomosidad_max < 9.0:
                restricciones['gomosidad'] = (gomosidad_min, gomosidad_max)
        
        with col2:
            st.markdown("**💪 Dureza**")
            dureza_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="dur_min")
            dureza_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="dur_max")
            if dureza_min > 1.0 or dureza_max < 9.0:
                restricciones['dureza'] = (dureza_min, dureza_max)
            
            st.markdown("**🏖️ Arenosidad**")
            arenosidad_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="aren_min")
            arenosidad_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="aren_max")
            if arenosidad_min > 1.0 or arenosidad_max < 9.0:
                restricciones['arenosidad'] = (arenosidad_min, arenosidad_max)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✨ Brillo**")
            brillo_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="brill_min")
            brillo_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="brill_max")
            if brillo_min > 1.0 or brillo_max < 9.0:
                restricciones['brillo'] = (brillo_min, brillo_max)
        
        with col2:
            st.markdown("**💎 Claridad**")
            claridad_min = st.number_input("Mínima", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key="clar_min")
            claridad_max = st.number_input("Máxima", min_value=1.0, max_value=9.0, value=9.0, step=0.1, key="clar_max")
            if claridad_min > 1.0 or claridad_max < 9.0:
                restricciones['claridad'] = (claridad_min, claridad_max)
    
    # Agregar restricciones a la configuración
    if restricciones:
        configuracion['restricciones'] = restricciones
    
    # ==================== RESUMEN DE CONFIGURACIÓN ====================
    
    with st.expander("📋 Resumen de Configuración", expanded=False):
        st.markdown("### 🎯 Resumen de tu configuración")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Maximizar (que sean ALTOS):**")
            maximizar = [f"{attr_descriptions.get(a, {}).get('nombre', a)}" 
                        for a, c in configuracion.items() 
                        if a != 'restricciones' and c.get('accion') == 'maximizar' and c.get('peso', 0) > 0]
            if maximizar:
                for item in maximizar:
                    st.write(f"  • {item}")
            else:
                st.write("  • Ninguno")
        
        with col2:
            st.markdown("**📉 Minimizar (que sean BAJOS):**")
            minimizar = [f"{attr_descriptions.get(a, {}).get('nombre', a)}" 
                        for a, c in configuracion.items() 
                        if a != 'restricciones' and c.get('accion') == 'minimizar' and c.get('peso', 0) > 0]
            if minimizar:
                for item in minimizar:
                    st.write(f"  • {item}")
            else:
                st.write("  • Ninguno")
        
        if restricciones:
            st.markdown("**🔒 Restricciones:**")
            for attr, (min_val, max_val) in restricciones.items():
                nombre = attr_descriptions.get(attr, {}).get('nombre', attr)
                st.write(f"  • {nombre}: {min_val:.1f} - {max_val:.1f}")
    
    # ==================== BOTÓN DE OPTIMIZACIÓN ====================
    
    st.markdown("---")
    
    # Verificar que hay al menos un objetivo definido
    tiene_objetivo = any(
        c.get('accion') != 'ignorar' and c.get('peso', 0) > 0 
        for a, c in configuracion.items() 
        if a != 'restricciones'
    )
    
    if not tiene_objetivo:
        st.warning("⚠️ Debes seleccionar al menos un atributo para maximizar o minimizar (con peso > 0)")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        optimizar = st.button(
            "🚀 ENCONTRAR MEJOR RECETA",
            type="primary",
            use_container_width=True,
            disabled=not tiene_objetivo
        )
    
    # ==================== EJECUTAR OPTIMIZACIÓN ====================
    
    if optimizar:
        with st.spinner("🧬 Evolucionando recetas..."):
            # Crear recomendador
            recomendador = RecomendadorGeneticoCompleto(
                modelos, escaladores, targets, features, limits
            )
            
            # Ejecutar
            mejor_receta, mejor_perfil, mejor_puntuacion = recomendador.recomendar(
                configuracion,
                poblacion_size=poblacion_size,
                generaciones=generaciones,
                elitismo=elitismo/100,
                verbose=True
            )
        
        # ==================== MOSTRAR RESULTADOS ====================
        
        st.success("✅ ¡Receta encontrada!")
        st.markdown("---")
        
        # Gráfico de evolución
        st.subheader("📈 Evolución del Algoritmo")
        fig = recomendador.graficar_evolucion()
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        
        # Receta encontrada
        st.subheader("📋 Mejor Receta Encontrada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧪 Ingredientes")
            
            df_receta = pd.DataFrame({
                'Ingrediente': [descriptions.get(ing, ing) for ing in mejor_receta.keys()],
                'Cantidad (g)': [f"{mejor_receta[ing]:.2f}" for ing in mejor_receta.keys()],
                'Rango': [f"{limits[ing][0]:.1f} - {limits[ing][1]:.1f}" for ing in mejor_receta.keys()]
            })
            st.dataframe(df_receta, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📊 Perfil Sensorial Predicho (1-9)")
            
            # CORREGIDO: Crear listas de la misma longitud
            atributos = [attr_descriptions.get(a, {}).get('nombre', a.upper()) for a in mejor_perfil.keys()]
            valores = [f"{mejor_perfil[a]:.2f}" for a in mejor_perfil.keys()]
            niveles = []
            
            for a in mejor_perfil.keys():
                valor = mejor_perfil[a]
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
        
        # Verificar restricciones
        if restricciones:
            st.subheader("✅ Verificación de Restricciones")
            
            cols = st.columns(min(len(restricciones), 4))
            for i, (attr, (min_val, max_val)) in enumerate(restricciones.items()):
                with cols[i % 4]:
                    valor = mejor_perfil.get(attr, 0)
                    cumple = min_val <= valor <= max_val
                    nombre = attr_descriptions.get(attr, {}).get('nombre', attr.upper())
                    if cumple:
                        st.success(f"✅ {nombre}: {valor:.2f}")
                        st.caption(f"Rango: {min_val:.1f} - {max_val:.1f}")
                    else:
                        st.error(f"❌ {nombre}: {valor:.2f}")
                        st.caption(f"Rango: {min_val:.1f} - {max_val:.1f}")
        
        # ==================== INTERPRETACIÓN ====================
        
        st.subheader("💡 Interpretación de Resultados")
        
        # Identificar atributos destacados
        destacados = sorted(
            [(a, mejor_perfil.get(a, 0)) for a in mejor_perfil.keys()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        st.markdown(f"""
        **🎯 Puntuación final:** {mejor_puntuacion:.3f}
        
        **🏆 Atributos más destacados (escala 1-9):**
        """)
        for attr, val in destacados:
            nombre = attr_descriptions.get(attr, {}).get('nombre', attr.upper())
            st.write(f"  • {nombre}: {val:.2f}/9.0")
        
        # ==================== GUARDAR RECETA ====================
        
        st.markdown("---")
        
        if st.button("💾 Guardar esta receta", use_container_width=True):
            if 'recetas_guardadas' not in st.session_state:
                st.session_state['recetas_guardadas'] = []
            
            st.session_state['recetas_guardadas'].append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'receta': mejor_receta,
                'perfil': mejor_perfil,
                'puntuacion': mejor_puntuacion,
                'configuracion': configuracion,
                'tipo': 'recomendador_completo'
            })
            
            st.success("✅ Receta guardada! Puedes compararla en el módulo de Comparación")

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    print("="*60)
    print("🧬 RECOMENDADOR DE RECETAS - CONTROL TOTAL")
    print("="*60)
    print("\nEste módulo se ejecuta dentro de Streamlit.")
    print("Ejecuta: streamlit run dashboard/pipeline_app.py")
    print("\n📊 ESCALA DE ATRIBUTOS: 1-9")