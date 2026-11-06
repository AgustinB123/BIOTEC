"""Módulo 6: Comparación multidimensional de formulaciones y perfiles sensoriales."""

from datetime import datetime
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

try:
    from dashboard.utils import (
        cargar_modelos_sensoriales as cargar_modelos,
        get_ingredient_descriptions as _get_desc,
        get_ingredient_limits,
        get_project_paths as get_paths,
        predecir_perfil,
    )
except ImportError:
    from utils import (
        cargar_modelos_sensoriales as cargar_modelos,
        get_ingredient_descriptions as _get_desc,
        get_ingredient_limits,
        get_project_paths as get_paths,
        predecir_perfil,
    )

warnings.filterwarnings("ignore")


def cargar_dataset():
    """Carga el dataset procesado para comparar con formulaciones experimentales."""
    paths = get_paths()
    try:
        df = pd.read_excel(paths["dataset_procesado"])
        return df, None
    except Exception as exc:
        return None, str(exc)


def get_ingredient_descriptions():
    """Retorna nombres breves de cada ingrediente."""
    return _get_desc(short=True)

# ==================== INTERFAZ DE STREAMLIT ====================

def app_comparacion():
    """Aplicación principal de Comparación de Formulaciones"""
    
    st.title("📋 Comparación de Formulaciones")
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
        1. **Selecciona recetas** para comparar
        2. **Visualiza** las diferencias en gráficos
        3. **Analiza** qué receta es mejor para cada atributo
        
        **Puedes comparar:**
        - 📝 Recetas guardadas
        - 🧪 Recetas del dataset
        - ✏️ Recetas personalizadas
        """)
        
        st.markdown("---")
        st.caption("💡 **Tip:** Selecciona 2-4 recetas para una comparación clara.")
        
        st.markdown("---")
        st.caption("📊 **Atributos comparados (Escala 1-9):**")
        st.caption("• Brillo • Claridad • Dulzura")
        st.caption("• Acidez • Adhesividad • Elasticidad")
        st.caption("• Arenosidad • Dureza • Gomosidad")
        st.caption("• Masticabilidad")
    
    # ==================== SELECCIÓN DE RECETAS ====================
    
    st.subheader("📝 Seleccionar Recetas para Comparar")
    
    # Opciones de selección
    tab1, tab2, tab3 = st.tabs(["📝 Recetas Guardadas", "🧪 Del Dataset", "✏️ Personalizada"])
    
    recetas_seleccionadas = []
    nombres_recetas = []
    
    with tab1:
        st.markdown("#### Recetas que has guardado")
        
        if 'recetas_guardadas' in st.session_state and st.session_state['recetas_guardadas']:
            df_guardadas = pd.DataFrame([
                {
                    'Nombre': f"Receta {i+1} ({r.get('timestamp', '')[:16]})",
                    'Tipo': r.get('tipo', 'manual'),
                    'Puntuación': r.get('puntuacion', 0) if r.get('puntuacion') else 'N/A'
                }
                for i, r in enumerate(st.session_state['recetas_guardadas'])
            ])
            st.dataframe(df_guardadas, use_container_width=True, hide_index=True)
            
            seleccion_guardadas = st.multiselect(
                "Seleccionar recetas guardadas",
                options=range(len(st.session_state['recetas_guardadas'])),
                format_func=lambda i: f"Receta {i+1} ({st.session_state['recetas_guardadas'][i].get('timestamp', '')[:16]})",
                key="guardadas"
            )
            
            for i in seleccion_guardadas:
                receta = st.session_state['recetas_guardadas'][i]
                if 'perfil' in receta:
                    recetas_seleccionadas.append(receta['perfil'])
                elif 'receta' in receta:
                    # Predecir perfil si no está guardado
                    perfil = predecir_perfil(receta['receta'], modelos, escaladores, targets, features)
                    recetas_seleccionadas.append(perfil)
                nombres_recetas.append(f"Guardada {i+1}")
        else:
            st.info("No hay recetas guardadas. Usa los otros módulos para guardar recetas.")
    
    with tab2:
        st.markdown("#### Recetas del Dataset")
        
        df_dataset, error_dataset = cargar_dataset()
        
        if error_dataset:
            st.warning(f"No se pudo cargar el dataset: {error_dataset}")
        else:
            # Mostrar recetas disponibles
            recetas_dataset = df_dataset[['codigo_gomita', 'fibra_tipo']].drop_duplicates()
            st.dataframe(recetas_dataset.head(10), use_container_width=True, hide_index=True)
            
            seleccion_dataset = st.multiselect(
                "Seleccionar recetas del dataset",
                options=recetas_dataset['codigo_gomita'].tolist(),
                key="dataset"
            )
            
            for codigo in seleccion_dataset:
                # Buscar el perfil en el dataset
                row = df_dataset[df_dataset['codigo_gomita'] == codigo].iloc[0]
                perfil = {target: row[target] for target in targets if target in row}
                if perfil:
                    recetas_seleccionadas.append(perfil)
                    nombres_recetas.append(codigo)
    
    with tab3:
        st.markdown("#### Crear Receta Personalizada")
        st.markdown("Puedes crear hasta 2 recetas personalizadas para comparar.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Receta A**")
            usar_a = st.checkbox("Usar Receta A", key="use_a")
            if usar_a:
                receta_a = {}
                for ing in ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g']:
                    receta_a[ing] = st.slider(
                        f"{descriptions[ing]} (g)",
                        min_value=0.0, max_value=12.0, value=5.0, step=0.1,
                        key=f"a_{ing}"
                    )
                for ing in ['goma_xantica_g', 'acido_citrico_g', 'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g']:
                    receta_a[ing] = st.slider(
                        f"{descriptions[ing]} (g)",
                        min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                        key=f"a_{ing}"
                    )
                
                if st.button("Predecir Receta A", key="pred_a"):
                    perfil_a = predecir_perfil(receta_a, modelos, escaladores, targets, features)
                    recetas_seleccionadas.append(perfil_a)
                    nombres_recetas.append("Personalizada A")
        
        with col2:
            st.markdown("**Receta B**")
            usar_b = st.checkbox("Usar Receta B", key="use_b")
            if usar_b:
                receta_b = {}
                for ing in ['azucar_g', 'jarabe_glucosa_g', 'agua_g', 'gelatina_g']:
                    receta_b[ing] = st.slider(
                        f"{descriptions[ing]} (g)",
                        min_value=0.0, max_value=12.0, value=5.0, step=0.1,
                        key=f"b_{ing}"
                    )
                for ing in ['goma_xantica_g', 'acido_citrico_g', 'stevia_g', 'esencia_g', 'sorbitol_g', 'fibra_g']:
                    receta_b[ing] = st.slider(
                        f"{descriptions[ing]} (g)",
                        min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                        key=f"b_{ing}"
                    )
                
                if st.button("Predecir Receta B", key="pred_b"):
                    perfil_b = predecir_perfil(receta_b, modelos, escaladores, targets, features)
                    recetas_seleccionadas.append(perfil_b)
                    nombres_recetas.append("Personalizada B")
    
    # ==================== MOSTRAR COMPARACIÓN ====================
    
    if recetas_seleccionadas:
        st.markdown("---")
        st.subheader(f"📊 Comparación de {len(recetas_seleccionadas)} Recetas")
        st.caption("📊 **Escala de atributos: 1-9**")
        
        # ==================== 1. GRÁFICO DE RADAR COMPARATIVO ====================
        
        st.markdown("### 📈 Comparación en Radar (Escala 1-9)")
        
        # Crear figura de Plotly
        fig = go.Figure()
        
        # Paleta de colores
        colors = ['#2E86AB', '#A2D5C6', '#F5B041', '#E74C3C', '#9B59B6', '#2ECC71', '#1ABC9C', '#3498DB']
        
        categories = [a.upper() for a in targets]
        
        for i, (perfil, nombre) in enumerate(zip(recetas_seleccionadas, nombres_recetas)):
            valores = [perfil.get(a, 0) for a in targets]
            valores += valores[:1]  # Cerrar el círculo
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Cerrar el círculo
            
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=categories + [categories[0]],
                name=nombre[:20],  # Limitar nombre
                line=dict(color=colors[i % len(colors)], width=2),
                fillcolor=colors[i % len(colors)] + '30'
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
            height=500,
            title="Comparación de Perfiles Sensoriales (1-9)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==================== 2. GRÁFICO DE BARRAS AGRUPADAS ====================
        
        st.markdown("### 📊 Comparación en Barras (Escala 1-9)")
        
        # Crear DataFrame para el gráfico
        df_comp = pd.DataFrame()
        for i, (perfil, nombre) in enumerate(zip(recetas_seleccionadas, nombres_recetas)):
            df_comp[nombre[:20]] = [perfil.get(a, 0) for a in targets]  # Limitar nombre
        df_comp['Atributo'] = [a.upper() for a in targets]
        df_comp = df_comp.set_index('Atributo')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_comp.plot(kind='bar', ax=ax, color=colors[:len(recetas_seleccionadas)])
        ax.set_xlabel('Atributo', fontsize=12)
        ax.set_ylabel('Intensidad (1-9)', fontsize=12)
        ax.set_title('Comparación de Atributos por Receta', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 9.5)
        
        # Añadir líneas de referencia
        ax.axhline(y=7, color='green', linestyle='--', alpha=0.3, label='Alta (≥7)')
        ax.axhline(y=4, color='orange', linestyle='--', alpha=0.3, label='Media (≥4)')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # ==================== 3. TABLA COMPARATIVA ====================
        
        st.markdown("### 📋 Tabla Comparativa")
        
        # Crear tabla con valores y diferencias
        df_tabla = pd.DataFrame()
        df_tabla['Atributo'] = [a.upper() for a in targets]
        
        for i, (perfil, nombre) in enumerate(zip(recetas_seleccionadas, nombres_recetas)):
            df_tabla[nombre[:20]] = [f"{perfil.get(a, 0):.2f}" for a in targets]
        
        # Calcular diferencias respecto a la primera receta (solo si hay más de 1)
        if len(recetas_seleccionadas) > 1:
            base = [recetas_seleccionadas[0].get(a, 0) for a in targets]
            for i, nombre in enumerate(nombres_recetas[1:], 2):
                col_name = f'Δ vs {nombres_recetas[0][:10]}'
                diferencias = [recetas_seleccionadas[i-1].get(a, 0) - base[j] for j, a in enumerate(targets)]
                df_tabla[col_name] = [f"{d:+.2f}" for d in diferencias]
        
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)
        
        # ==================== 4. ANÁLISIS DE GANADORES ====================
        
        st.markdown("### 🏆 Análisis de Ganadores por Atributo")
        
        # Determinar qué receta gana en cada atributo
        ganadores = {}
        for i, attr in enumerate(targets):
            valores = [perfil.get(attr, 0) for perfil in recetas_seleccionadas]
            max_idx = np.argmax(valores)
            min_idx = np.argmin(valores)
            ganadores[attr.upper()] = {
                'ganador': nombres_recetas[max_idx],
                'valor': valores[max_idx],
                'perdedor': nombres_recetas[min_idx],
                'diferencia': valores[max_idx] - valores[min_idx]
            }
        
        # Mostrar en columnas
        cols = st.columns(3)
        for i, (attr, info) in enumerate(ganadores.items()):
            with cols[i % 3]:
                # Color según valor
                if info['valor'] >= 7:
                    color = "🟢"
                elif info['valor'] >= 4:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.metric(
                    label=f"{color} {attr}",
                    value=info['ganador'][:15],
                    delta=f"{info['valor']:.2f} (vs {info['perdedor'][:10]}: {info['diferencia']:+.2f})"
                )
        
        # ==================== 5. RESUMEN DE PUNTUACIONES ====================
        
        st.markdown("### 📊 Resumen de Puntuaciones")
        
        # Calcular puntuación promedio de cada receta
        promedios = []
        for perfil in recetas_seleccionadas:
            avg = np.mean([perfil.get(a, 0) for a in targets])
            promedios.append(avg)
        
        # Crear gráfico de barras de promedios
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colores_barras = ['#2ECC71' if p >= 7 else '#F1C40F' if p >= 4 else '#E74C3C' for p in promedios]
        bars = ax.bar(nombres_recetas, promedios, color=colores_barras, edgecolor='black', linewidth=1)
        
        # Añadir valores
        for bar, val in zip(bars, promedios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylim(0, 9.5)
        ax.set_ylabel('Puntuación Promedio (1-9)', fontsize=12)
        ax.set_title('Puntuación Promedio de Cada Receta', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=7, color='green', linestyle='--', alpha=0.3, label='Excelente (≥7)')
        ax.axhline(y=4, color='orange', linestyle='--', alpha=0.3, label='Aceptable (≥4)')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # ==================== 6. RECOMENDACIÓN ====================
        
        st.markdown("### 💡 Recomendación")
        
        mejor_idx = np.argmax(promedios)
        
        # Identificar atributos destacados de la mejor receta
        mejor_perfil = recetas_seleccionadas[mejor_idx]
        destacados = sorted(
            [(attr, mejor_perfil.get(attr, 0)) for attr in targets],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        st.success(f"""
        **🏆 La receta con mejor perfil general es: {nombres_recetas[mejor_idx]}**
        
        - **Puntuación promedio:** {promedios[mejor_idx]:.2f}/9.0
        - **Atributos destacados:**
        """)
        
        for attr, val in destacados:
            if val >= 7:
                emoji = "🟢"
            elif val >= 4:
                emoji = "🟡"
            else:
                emoji = "🔴"
            st.write(f"  {emoji} {attr.upper()}: {val:.2f}/9.0")
        
        # ==================== 7. GUARDAR COMPARACIÓN ====================
        
        st.markdown("---")
        
        if st.button("📊 Exportar Comparación", use_container_width=True):
            # Crear DataFrame con todos los datos de comparación
            df_export = pd.DataFrame()
            for i, (perfil, nombre) in enumerate(zip(recetas_seleccionadas, nombres_recetas)):
                for attr in targets:
                    df_export.loc[attr.upper(), nombre[:20]] = perfil.get(attr, 0)
            
            # Añadir promedios
            for i, (perfil, nombre) in enumerate(zip(recetas_seleccionadas, nombres_recetas)):
                df_export.loc['Promedio', nombre[:20]] = promedios[i]
            
            # Mostrar tabla para exportar
            st.dataframe(df_export.round(2), use_container_width=True)
            
            # Botón para descargar CSV
            csv = df_export.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"comparacion_formulaciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success("✅ Comparación lista para descargar!")
    
    else:
        st.info("👆 Selecciona recetas para comparar usando las pestañas de arriba")
        
        # Mostrar ejemplo de cómo se verá
        st.markdown("""
        ### 📊 ¿Cómo se verá la comparación?
        
        Cuando selecciones recetas, verás:
        
        1. **Gráfico de Radar** - Compara los perfiles completos (Escala 1-9)
        2. **Gráfico de Barras** - Compara atributo por atributo
        3. **Tabla Comparativa** - Valores numéricos y diferencias
        4. **Análisis de Ganadores** - Qué receta gana en cada atributo
        5. **Resumen de Puntuaciones** - Promedios de cada receta
        6. **Recomendación** - Cuál es la mejor receta en general
        
        💡 **Selecciona 2-4 recetas** para una comparación clara y útil.
        """)
        
        # Mostrar ejemplo de perfil ideal
        with st.expander("📖 Ver Perfil Ideal (Escala 1-9)"):
            perfil_ideal = {
                'Brillo': 7.0,
                'Claridad': 6.0,
                'Dulzura': 7.0,
                'Acidez': 3.0,
                'Adhesividad': 3.0,
                'Elasticidad': 7.0,
                'Arenosidad': 3.0,
                'Dureza': 4.0,
                'Gomosidad': 6.0,
                'Masticabilidad': 5.0
            }
            df_ideal = pd.DataFrame({
                'Atributo': perfil_ideal.keys(),
                'Valor Ideal': perfil_ideal.values(),
                'Nivel': ['🟢 Alta' if v >= 7 else '🟡 Media' if v >= 4 else '🔴 Baja' for v in perfil_ideal.values()]
            })
            st.dataframe(df_ideal, use_container_width=True, hide_index=True)
            st.caption("💡 Este es el perfil objetivo para una gomita exitosa.")

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    print("="*60)
    print("📋 MÓDULO 6: COMPARACIÓN DE FORMULACIONES")
    print("="*60)
    print("\nEste módulo se ejecuta dentro de Streamlit.")
    print("Ejecuta: streamlit run dashboard/pipeline_app.py")
    print("\n📊 ESCALA DE ATRIBUTOS: 1-9")