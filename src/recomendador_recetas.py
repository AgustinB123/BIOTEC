"""
RECOMENDADOR DE RECETAS - VERSIÓN MEJORADA
Autor: Proyecto Gomitas IA
Fecha: 2026-08-24
Descripción: Algoritmo Genético con mutación adaptativa y objetivos combinados
"""

import os
import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'data', 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS_FILE = os.path.join(MODELS_DIR, 'modelos_sensoriales_completos.pkl')
FEATURES_FILE = os.path.join(MODELS_DIR, 'features_sensoriales.pkl')

# ==================== 1. CARGAR MODELOS ====================

def load_models():
    """Carga los modelos entrenados"""
    print("📂 Cargando modelos...")
    
    with open(MODELS_FILE, 'rb') as f:
        data = pickle.load(f)
    
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)
    
    print(f"   ✅ {len(data['models'])} modelos sensoriales cargados")
    print(f"   ✅ {len(features)} features cargados")
    
    return data['models'], data['scalers'], data['targets'], features

# ==================== 2. DEFINIR LÍMITES ====================

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

# ==================== 3. PREDECIR PERFIL ====================

def predecir_perfil(receta, models, scalers, targets, features):
    """Predice el perfil sensorial de una receta"""
    receta_df = pd.DataFrame([receta])
    
    for col in features:
        if col not in receta_df.columns:
            receta_df[col] = 0
    
    receta_df = receta_df[features]
    
    predicciones = {}
    for target in targets:
        X_scaled = scalers[target].transform(receta_df)
        predicciones[target] = models[target].predict(X_scaled)[0]
    
    return predicciones

# ==================== 4. ALGORITMO GENÉTICO MEJORADO ====================

class RecomendadorGeneticoMejorado:
    """
    Algoritmo Genético con mutación adaptativa y objetivos combinados
    """
    
    def __init__(self, models, scalers, targets, features, limits):
        self.models = models
        self.scalers = scalers
        self.targets = targets
        self.features = features
        self.limits = limits
        self.ingredient_names = list(limits.keys())
        self.historial = []
        self.mejores_puntuaciones = []
        self.diversidad = []
        
    def crear_individuo(self):
        """Crea un individuo aleatorio"""
        receta = {}
        for ingrediente, (min_val, max_val) in self.limits.items():
            receta[ingrediente] = round(random.uniform(min_val, max_val), 3)
        return receta
    
    def evaluar_individuo(self, receta, objetivo):
        """Evalúa una receta según el objetivo"""
        perfil = predecir_perfil(receta, self.models, self.scalers, self.targets, self.features)
        
        puntuacion = 0
        
        # Atributos a maximizar
        for attr, peso in objetivo.get('maximizar', {}).items():
            if attr in perfil:
                valor_normalizado = perfil[attr] / 7.0
                puntuacion += valor_normalizado * peso
        
        # Atributos a minimizar
        for attr, peso in objetivo.get('minimizar', {}).items():
            if attr in perfil:
                valor_normalizado = 1 - (perfil[attr] / 7.0)
                puntuacion += valor_normalizado * peso
        
        # Penalizaciones
        penalizacion = 0
        if 'restricciones' in objetivo:
            for attr, (min_val, max_val) in objetivo['restricciones'].items():
                if attr in perfil:
                    if perfil[attr] < min_val:
                        penalizacion += (min_val - perfil[attr]) * 3
                    if perfil[attr] > max_val:
                        penalizacion += (perfil[attr] - max_val) * 3
        
        puntuacion -= penalizacion
        
        return puntuacion, perfil
    
    def seleccionar_padres(self, poblacion, fitness, num_padres=2):
        """Selección por torneo con diversidad"""
        padres = []
        for _ in range(num_padres):
            indices = random.sample(range(len(poblacion)), min(4, len(poblacion)))
            mejor_idx = max(indices, key=lambda i: fitness[i] + random.uniform(0, 0.1))
            padres.append(poblacion[mejor_idx])
        return padres
    
    def cruzar_adaptativo(self, padre1, padre2, generacion, generaciones_totales):
        """Cruza con mutación adaptativa"""
        hijo = {}
        
        # La mutación disminuye con el tiempo (exploración → explotación)
        tasa_mutacion = 0.4 * (1 - generacion / generaciones_totales) + 0.05
        
        for ingrediente in self.ingredient_names:
            # Mezcla uniforme
            if random.random() < 0.5:
                hijo[ingrediente] = padre1[ingrediente]
            else:
                hijo[ingrediente] = padre2[ingrediente]
            
            # Mutación adaptativa
            if random.random() < tasa_mutacion:
                min_val, max_val = self.limits[ingrediente]
                # Amplitud de mutación: mayor al principio
                amplitud = (max_val - min_val) * 0.4 * (1 - generacion / generaciones_totales + 0.05)
                mutacion = random.uniform(-amplitud, amplitud)
                hijo[ingrediente] = max(min_val, min(max_val, hijo[ingrediente] + mutacion))
            
            hijo[ingrediente] = round(hijo[ingrediente], 3)
        
        return hijo
    
    def recomendar(self, objetivo, poblacion_size=150, generaciones=80, 
                   elitismo=0.15, verbose=True):
        """
        Encuentra la mejor receta con mutación adaptativa
        """
        print(f"\n🧬 Iniciando Algoritmo Genético (Mejorado)...")
        print(f"   Población: {poblacion_size}")
        print(f"   Generaciones: {generaciones}")
        print(f"   Elitismo: {elitismo*100:.0f}%")
        print(f"   Mutación: Adaptativa (exploración → explotación)")
        
        poblacion = [self.crear_individuo() for _ in range(poblacion_size)]
        
        self.historial = []
        self.mejores_puntuaciones = []
        self.diversidad = []
        
        mejor_receta = None
        mejor_puntuacion = -float('inf')
        mejor_perfil = None
        
        # Ventana para detectar estancamiento
        ventana_estancamiento = 10
        ultimas_puntuaciones = []
        
        for gen in range(generaciones):
            # Evaluar población
            fitness = []
            perfiles = []
            
            for individuo in poblacion:
                puntuacion, perfil = self.evaluar_individuo(individuo, objetivo)
                fitness.append(puntuacion)
                perfiles.append(perfil)
            
            # Guardar mejor
            mejor_idx = np.argmax(fitness)
            gen_mejor_puntuacion = fitness[mejor_idx]
            self.mejores_puntuaciones.append(gen_mejor_puntuacion)
            
            # Calcular diversidad
            fitness_std = np.std(fitness)
            self.diversidad.append(fitness_std)
            
            if gen_mejor_puntuacion > mejor_puntuacion:
                mejor_puntuacion = gen_mejor_puntuacion
                mejor_receta = poblacion[mejor_idx].copy()
                mejor_perfil = perfiles[mejor_idx].copy()
            
            # Detectar estancamiento
            ultimas_puntuaciones.append(gen_mejor_puntuacion)
            if len(ultimas_puntuaciones) > ventana_estancamiento:
                ultimas_puntuaciones.pop(0)
            
            # Si está estancado, aumentar mutación
            if len(ultimas_puntuaciones) == ventana_estancamiento:
                if max(ultimas_puntuaciones) - min(ultimas_puntuaciones) < 0.01:
                    if verbose and gen % 5 == 0:
                        print(f"   ⚠️ Estancamiento detectado en gen {gen}")
            
            if verbose and (gen % 10 == 0 or gen == generaciones - 1):
                print(f"   Generación {gen}: Mejor = {gen_mejor_puntuacion:.3f}, "
                      f"Diversidad = {fitness_std:.3f}")
            
            # Crear nueva población
            nueva_poblacion = []
            
            # Elitismo
            num_elite = int(poblacion_size * elitismo)
            elite_indices = np.argsort(fitness)[-num_elite:]
            for idx in elite_indices:
                nueva_poblacion.append(poblacion[idx].copy())
            
            # Generar el resto con mutación adaptativa
            while len(nueva_poblacion) < poblacion_size:
                padres = self.seleccionar_padres(poblacion, fitness)
                hijo = self.cruzar_adaptativo(padres[0], padres[1], gen, generaciones)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        print(f"\n   ✅ Mejor puntuación: {mejor_puntuacion:.3f}")
        
        return mejor_receta, mejor_perfil, mejor_puntuacion
    
    def graficar_evolucion(self, save_path=None):
        """Grafica la evolución"""
        if not self.mejores_puntuaciones:
            print("⚠️ No hay historial para graficar")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Evolución de la puntuación
        ax1.plot(self.mejores_puntuaciones, 'b-', linewidth=2)
        ax1.set_xlabel('Generación', fontsize=12)
        ax1.set_ylabel('Puntuación', fontsize=12)
        ax1.set_title('Evolución de la Puntuación', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Diversidad de la población
        ax2.plot(self.diversidad, 'g-', linewidth=2)
        ax2.set_xlabel('Generación', fontsize=12)
        ax2.set_ylabel('Diversidad (std)', fontsize=12)
        ax2.set_title('Diversidad de la Población', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Gráfico guardado: {save_path}")
        
        plt.close()
        return fig

# ==================== 6. EJEMPLOS CON OBJETIVOS COMBINADOS ====================

def ejemplo_balanceado():
    """Ejemplo: Balancear dulzura, elasticidad y bajo adhesividad"""
    print("\n" + "="*60)
    print("🍬 EJEMPLO: Receta Balanceada (Dulzura + Elasticidad - Adhesividad)")
    print("="*60)
    
    objetivo = {
        'maximizar': {
            'dulzura': 0.4,
            'elasticidad': 0.3,
            'brillo': 0.3
        },
        'minimizar': {
            'adhesividad': 0.5,
            'acidez': 0.3,
            'arenosidad': 0.2
        },
        'restricciones': {
            'dulzura': (2.5, 7.0),
            'adhesividad': (1.0, 4.5),
            'acidez': (1.0, 3.5)
        }
    }
    
    # Cargar modelos
    models, scalers, targets, features = load_models()
    limits = get_ingredient_limits()
    
    # Crear recomendador
    recomendador = RecomendadorGeneticoMejorado(models, scalers, targets, features, limits)
    
    # Ejecutar
    receta, perfil, puntuacion = recomendador.recomendar(
        objetivo,
        poblacion_size=150,
        generaciones=80,
        elitismo=0.15,
        verbose=True
    )
    
    # Guardar gráfico
    save_path = os.path.join(PLOTS_DIR, 'recomendador_balanceado_evolucion.png')
    recomendador.graficar_evolucion(save_path)
    
    # Mostrar resultados
    print("\n📊 MEJOR RECETA ENCONTRADA:")
    for ing, valor in receta.items():
        print(f"   {ing}: {valor:.2f}g")
    
    print("\n📊 PERFIL PREDICHO:")
    for attr, valor in perfil.items():
        print(f"   {attr}: {valor:.2f}")
    
    # Verificar restricciones
    print("\n✅ VERIFICACIÓN DE RESTRICCIONES:")
    for attr, (min_val, max_val) in objetivo.get('restricciones', {}).items():
        if attr in perfil:
            valor = perfil[attr]
            cumple = min_val <= valor <= max_val
            estado = "✅" if cumple else "❌"
            print(f"   {estado} {attr}: {valor:.2f} (rango: {min_val}-{max_val})")
    
    return receta, perfil

def ejemplo_saludable():
    """Ejemplo: Receta saludable (bajo azúcar, alta fibra, baja adhesividad)"""
    print("\n" + "="*60)
    print("🍬 EJEMPLO: Receta Saludable (Bajo Azúcar + Alta Fibra)")
    print("="*60)
    
    objetivo = {
        'maximizar': {
            'fibra_g': 0.3,
            'elasticidad': 0.3,
            'brillo': 0.2
        },
        'minimizar': {
            'azucar_g': 0.4,
            'adhesividad': 0.3,
            'acidez': 0.2
        },
        'restricciones': {
            'azucar_g': (0.2, 4.0),     # Poco azúcar
            'fibra_g': (1.5, 2.6),       # Alta fibra
            'adhesividad': (1.0, 4.0),   # Baja adhesividad
            'dulzura': (2.0, 5.0)        # Dulzura aceptable
        }
    }
    
    models, scalers, targets, features = load_models()
    limits = get_ingredient_limits()
    
    recomendador = RecomendadorGeneticoMejorado(models, scalers, targets, features, limits)
    
    receta, perfil, puntuacion = recomendador.recomendar(
        objetivo,
        poblacion_size=150,
        generaciones=80,
        elitismo=0.15,
        verbose=True
    )
    
    save_path = os.path.join(PLOTS_DIR, 'recomendador_saludable_evolucion.png')
    recomendador.graficar_evolucion(save_path)
    
    print("\n📊 MEJOR RECETA ENCONTRADA:")
    for ing, valor in receta.items():
        print(f"   {ing}: {valor:.2f}g")
    
    print("\n📊 PERFIL PREDICHO:")
    for attr, valor in perfil.items():
        print(f"   {attr}: {valor:.2f}")
    
    print("\n✅ VERIFICACIÓN DE RESTRICCIONES:")
    for attr, (min_val, max_val) in objetivo.get('restricciones', {}).items():
        if attr in perfil:
            valor = perfil[attr]
            cumple = min_val <= valor <= max_val
            estado = "✅" if cumple else "❌"
            print(f"   {estado} {attr}: {valor:.2f} (rango: {min_val}-{max_val})")
    
    return receta, perfil

# ==================== 7. FUNCIÓN PRINCIPAL ====================

def main():
    print("\n" + "="*60)
    print("🍬 RECOMENDADOR DE RECETAS - VERSIÓN MEJORADA")
    print("="*60)
    
    print("\n📋 OPCIONES:")
    print("   1. Receta Balanceada (Dulzura + Elasticidad - Adhesividad)")
    print("   2. Receta Saludable (Bajo Azúcar + Alta Fibra)")
    print("   3. Ambos ejemplos")
    
    try:
        opcion = int(input("\nSelecciona una opción (1-3): "))
    except:
        opcion = 3
    
    if opcion == 1:
        ejemplo_balanceado()
    elif opcion == 2:
        ejemplo_saludable()
    else:
        ejemplo_balanceado()
        ejemplo_saludable()

if __name__ == "__main__":
    main()