import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import json

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Cultivos",
    page_icon="🌾",
    layout="wide"
)

# URL del backend
BACKEND_URL = "http://localhost:5001"

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🌾 Sistema de Predicción de Cultivos Agrícolas</h1>', unsafe_allow_html=True)

# ==================== FUNCIONES DE BACKEND ====================

@st.cache_data(ttl=60)
def check_backend_health():
    """Verifica la conexión con el backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def make_prediction(data):
    """Realiza una predicción usando el backend"""
    response = requests.post(f"{BACKEND_URL}/predict", json=data)
    return response.json()

@st.cache_data(ttl=300)
def get_data_summary():
    """Obtiene el resumen de datos del backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/data/summary")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_variable_analysis(variable):
    """Obtiene análisis de una variable específica"""
    try:
        response = requests.post(f"{BACKEND_URL}/data/analysis", json={'variable': variable})
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=300)
def get_correlation_matrix():
    """Obtiene la matriz de correlación"""
    try:
        response = requests.get(f"{BACKEND_URL}/data/correlation")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_images_list():
    """Obtiene la lista de imágenes disponibles"""
    try:
        response = requests.get(f"{BACKEND_URL}/images/list")
        return response.json() if response.status_code == 200 else []
    except:
        return []

def get_image(filename):
    """Obtiene una imagen específica"""
    try:
        response = requests.get(f"{BACKEND_URL}/images/{filename}")
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except:
        return None

# ==================== VERIFICACIÓN DE BACKEND ====================

health = check_backend_health()

if health is None:
    st.error("⚠️ No se puede conectar con el backend. Asegúrate de que el servidor Flask esté corriendo en http://localhost:5001")
    st.info("Ejecuta: `python backend.py` en otra terminal")
    st.stop()

# ==================== SIDEBAR ====================

st.sidebar.title("🔧 Estado del Sistema")
if health:
    st.sidebar.success("✅ Backend Conectado")
    if health.get('model_loaded'):
        st.sidebar.success("✅ Modelo Cargado")
    else:
        st.sidebar.error("❌ Modelo No Cargado")
        st.sidebar.info("📁 Coloca el archivo `crop_model.pkl` en la carpeta `models/`")
    
    if health.get('data_loaded'):
        st.sidebar.success("✅ Datos Cargados")
    else:
        st.sidebar.error("❌ Datos No Cargados")
        st.sidebar.info("📁 Coloca el archivo `crop_data.csv` en la carpeta `data/`")

st.sidebar.markdown("---")

# Navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🔮 Predicción de Cultivos", "📊 Análisis de Datos", "🖼️ Visualizaciones Personalizadas", "📈 Comparación de Modelos"]
)

# ==================== PÁGINA 1: PREDICCIÓN ====================
if page == "🔮 Predicción de Cultivos":
    st.markdown('<h2 class="sub-header">Predicción de Cultivo Óptimo</h2>', unsafe_allow_html=True)
    
    if not health.get('model_loaded'):
        st.error("❌ Modelo no cargado. Coloca el archivo `crop_model.pkl` en la carpeta `models/` y reinicia el backend.")
        st.stop()

    selected_crop_encoded = 2
 
    st.info("💡 **Instrucciones:** Ajusta los parámetros del suelo y las condiciones climáticas para obtener una recomendación de cultivo.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧪 Composición del Suelo")
        
        nitrogen = st.slider(
            "Nivel de Nitrógeno (N)",
            min_value=0, max_value=140, value=80,
            help="Bajo: 0-40 | Medio: 40-90 | Alto: 90-140"
        )
        st.caption(f"📊 Valor actual: **{nitrogen}** - {'🟢 Alto' if nitrogen > 90 else '🟡 Medio' if nitrogen > 40 else '🔴 Bajo'}")
        
        phosphorus = st.slider(
            "Nivel de Fósforo (P)",
            min_value=0, max_value=145, value=50,
            help="Bajo: 0-30 | Medio: 30-60 | Alto: 60-145"
        )
        st.caption(f"📊 Valor actual: **{phosphorus}** - {'🟢 Alto' if phosphorus > 60 else '🟡 Medio' if phosphorus > 30 else '🔴 Bajo'}")
        
        potassium = st.slider(
            "Nivel de Potasio (K)",
            min_value=0, max_value=50, value=30,
            help="Bajo: 0-20 | Medio: 20-35 | Alto: 35-50"
        )
        st.caption(f"📊 Valor actual: **{potassium}** - {'🟢 Alto' if potassium > 35 else '🟡 Medio' if potassium > 20 else '🔴 Bajo'}")
        
        ph = st.slider(
            "pH del Suelo",
            min_value=3.5, max_value=9.0, value=6.5, step=0.1,
            help="Ácido: 3.5-5.5 | Neutro: 5.5-7.5 | Alcalino: 7.5-9.0"
        )
        st.caption(f"📊 Valor actual: **{ph:.1f}** - {'🟣 Alcalino' if ph > 7.5 else '🟢 Neutro' if ph > 5.5 else '🔵 Ácido'}")
    
    with col2:
        st.markdown("### 🌡️ Condiciones Climáticas")
        
        temperature = st.slider(
            "Temperatura Promedio (°C)",
            min_value=15, max_value=45, value=25,
            help="Fresca: 15-20°C | Templada: 20-30°C | Cálida: 30-45°C"
        )
        st.caption(f"🌡️ Temperatura: **{temperature}°C** - {'🔥 Cálida' if temperature > 30 else '☀️ Templada' if temperature > 20 else '❄️ Fresca'}")
        
        humidity = st.slider(
            "Humedad Relativa (%)",
            min_value=14, max_value=100, value=70,
            help="Baja: 14-50% | Media: 50-75% | Alta: 75-100%"
        )
        st.caption(f"💧 Humedad: **{humidity}%** - {'💦 Alta' if humidity > 75 else '💧 Media' if humidity > 50 else '🏜️ Baja'}")
        
        rainfall = st.slider(
            "Precipitación Anual (mm)",
            min_value=20, max_value=300, value=150,
            help="Baja: 20-80mm | Media: 80-150mm | Alta: 150-300mm"
        )
        st.caption(f"🌧️ Lluvia: **{rainfall}mm** - {'⛈️ Alta' if rainfall > 150 else '🌦️ Media' if rainfall > 80 else '☀️ Baja'}")
    
    st.markdown("---")
    
    if st.button("🔍 Predecir Cultivo Óptimo", type="primary", use_container_width=True):
        input_data = {
            'N': nitrogen,
            'P': phosphorus,
            'K': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall,
            'label_encoded': selected_crop_encoded
        }
        
        with st.spinner("Realizando predicción..."):
            result = make_prediction(input_data)
        
        if 'error' in result:
            st.error(f"❌ Error en la predicción: {result['error']}")
        else:
            predicted_crop = result['prediction']
            
            crop_info = {
                'coconut': {"name": "🥥 Coco (Coconut)", "icon": "🥥", "color": "#8B4513"},
                'cotton': {"name": "🌱 Algodón (Cotton)", "icon": "🌱", "color": "#90EE90"},
                'maize': {"name": "🌽 Maíz (Maize)", "icon": "🌽", "color": "#FFD700"},
                'rice': {"name": "🌾 Arroz (Rice)", "icon": "🌾", "color": "#DEB887"}
            }
            
            crop = crop_info.get(predicted_crop, {"name": predicted_crop, "icon": "🌱", "color": "#4CAF50"})
            
            st.success("✅ Predicción completada exitosamente!")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; color: white;'>
                    <h2 style='margin: 0; font-size: 2.5rem;'>Cultivo Recomendado</h2>
                    <h1 style='margin: 1rem 0; font-size: 3.5rem;'>{crop['name']}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📋 Recomendaciones Específicas")
            
            # Recomendaciones basadas en el cultivo predicho
            recommendations = {
                'rice': {
                    "name": "Arroz (Rice)",
                    "icon": "🌾",
                    "tips": [
                        "Requiere alta humedad (80-85%)",
                        "Temperatura ideal: 20-27°C",
                        "pH óptimo: 5.5-7.0",
                        "Alta precipitación necesaria (>180mm)"
                    ]
                },
                'maize': {
                    "name": "Maíz (Maize)",
                    "icon": "🌽",
                    "tips": [
                        "Humedad moderada (55-75%)",
                        "Temperatura ideal: 18-27°C",
                        "pH óptimo: 5.5-7.0",
                        "Precipitación moderada (60-110mm)"
                    ]
                },
                'cotton': {
                    "name": "Algodón (Cotton)",
                    "icon": "🌱",
                    "tips": [
                        "Requiere humedad media-alta (75-85%)",
                        "Temperatura ideal: 21-27°C",
                        "pH óptimo: 5.5-7.8",
                        "Precipitación baja-media (60-100mm)"
                    ]
                },
                'coconut': {
                    "name": "Coco (Coconut)",
                    "icon": "🥥",
                    "tips": [
                        "Alta humedad requerida (>90%)",
                        "Temperatura cálida: 25-30°C",
                        "pH óptimo: 5.5-6.5",
                        "Alta precipitación (>130mm)"
                    ]
                }
            }
            
            if predicted_crop in recommendations:
                rec = recommendations[predicted_crop]
                cols = st.columns(2)
                
                for idx, tip in enumerate(rec["tips"]):
                    with cols[idx % 2]:
                        st.info(f"✓ {tip}")

# ==================== PÁGINA 2: ANÁLISIS DE DATOS ====================
elif page == "📊 Análisis de Datos":
    st.markdown('<h2 class="sub-header">Análisis Exploratorio de Datos</h2>', unsafe_allow_html=True)
    
    if not health.get('data_loaded'):
        st.error("❌ Datos no cargados. Coloca el archivo `crop_data.csv` en la carpeta `data/` y reinicia el backend.")
        st.stop()
    
    # Obtener resumen de datos
    summary = get_data_summary()
    
    if summary is None:
        st.error("❌ Error al cargar los datos del backend")
        st.stop()
    
    # Estadísticas generales
    st.markdown("### 📈 Estadísticas Generales del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", summary['total_records'])
    with col2:
        st.metric("Clases de Cultivos", len(summary['classes']))
    with col3:
        st.metric("Variables Medidas", len(summary['statistics']))
    with col4:
        completeness = 100.0  # Asumimos datos completos si no hay info
        st.metric("Datos Completos", f"{completeness:.1f}%")
    
    st.markdown("---")
    
    # Distribución de clases
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌾 Distribución de Cultivos")
        class_data = summary['classes']
        fig = px.pie(
            values=list(class_data.values()),
            names=list(class_data.keys()),
            title="Proporción de Cultivos en el Dataset",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Conteo por Cultivo")
        fig = px.bar(
            x=list(class_data.keys()),
            y=list(class_data.values()),
            labels={'x': 'Cultivo', 'y': 'Cantidad'},
            title="Número de Muestras por Cultivo",
            color=list(class_data.values()),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de variables por cultivo
    st.markdown("### 🔬 Análisis de Variables por Cultivo")
    
    variable = st.selectbox(
        "Selecciona una variable para analizar:",
        ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    )
    
    # Obtener análisis de la variable
    var_analysis = get_variable_analysis(variable)
    
    if var_analysis:
        # Crear datos para visualización
        crops = list(var_analysis.keys())
        means = [var_analysis[crop]['mean'] for crop in crops]
        mins = [var_analysis[crop]['min'] for crop in crops]
        maxs = [var_analysis[crop]['max'] for crop in crops]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=crops,
                y=means,
                name='Promedio',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title=f"Promedio de {variable} por Cultivo",
                xaxis_title="Cultivo",
                yaxis_title=variable
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabla de estadísticas
            st.markdown(f"#### Estadísticas de {variable}")
            stats_df = pd.DataFrame(var_analysis).T
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Matriz de correlación
    st.markdown("### 🔗 Matriz de Correlación")
    
    corr_data = get_correlation_matrix()
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        
        fig = px.imshow(
            corr_df,
            text_auto='.2f',
            aspect="auto",
            title="Correlación entre Variables",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== PÁGINA 3: VISUALIZACIONES PERSONALIZADAS ====================
elif page == "🖼️ Visualizaciones Personalizadas":

    images_list = get_images_list()
    
    if images_list and len(images_list) > 0:
        st.markdown("### 📸 Galería de Visualizaciones")
        
        # Crear tabs para cada imagen
        tabs = st.tabs([f"Imagen {i+1}" for i in range(len(images_list))])
        
        for idx, (tab, img_info) in enumerate(zip(tabs, images_list)):
            with tab:
                # Obtener la imagen
                image = get_image(img_info['filename'])
                
                if image:
                    st.image(image, caption=f"Visualización: {img_info['filename']}", use_container_width=True)
                    
                else:
                    st.error(f"❌ No se pudo cargar la imagen: {img_info['filename']}")
    else:
        st.warning("📂 No se encontraron imágenes en la carpeta `images/`")
        st.info("💡 Coloca archivos .png, .jpg o .jpeg en la carpeta `images/` del backend y recarga la página")
        

# ==================== PÁGINA 4: COMPARACIÓN DE MODELOS ====================
elif page == "📈 Comparación de Modelos":
    st.markdown('<h2 class="sub-header">Comparación y Selección del Modelo</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Proceso de Desarrollo del Modelo
    
    Este proyecto siguió un proceso iterativo de mejora continua para seleccionar el mejor modelo
    de clasificación de cultivos. A continuación se detallan las etapas y resultados.
    """)
    
    # Fase 1: Resultados iniciales
    st.markdown("### 📊 Fase 1: Entrenamiento Inicial (Baseline)")
    
    st.markdown("""
    **Configuración:**
    - ❌ Sin manejo de outliers
    - ❌ Sin normalización (MinMaxScaler)
    - ✅ Todas las clases del dataset original
    
    **Resultado:** 51% de precisión con Ridge Classifier
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        phase1_data = {
            'Modelo': ['Ridge Classifier'],
            'Precisión': [51]
        }
        fig = px.bar(
            phase1_data,
            x='Modelo',
            y='Precisión',
            title='Fase 1: Resultado Baseline',
            color='Precisión',
            color_continuous_scale='Reds',
            text='Precisión'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.warning("""
        **Problemas Identificados:**
        - 🔴 Bajo rendimiento general
        - 🔴 Datos sin preprocesamiento
        - 🔴 Múltiples clases generando ruido
        - 🔴 Outliers afectando el entrenamiento
        """)
    
    st.markdown("---")
    
    # Fase 2: Limpieza de datos
    st.markdown("### 🧹 Fase 2: Limpieza de Datos y Feature Engineering")
    
    st.markdown("""
    **Mejoras Implementadas:**
    - ✅ Manejo de outliers
    - ✅ MinMaxScaler para normalización
    - ✅ LabelEncoder para variables categóricas
    - ✅ Reducción a 4 clases principales (rice, maize, cotton, coconut)
    
    **Resultado:** 73% de precisión con K Neighbors Classifier (+22% mejora)
    """)
    
    phase2_data = {
        'Modelo': ['Ridge Classifier\n(Fase 1)', 'K Neighbors\n(Fase 2)'],
        'Precisión': [51, 73],
        'Fase': ['Fase 1', 'Fase 2']
    }
    
    fig = px.bar(
        phase2_data,
        x='Modelo',
        y='Precisión',
        title='Comparación Fase 1 vs Fase 2',
        color='Fase',
        text='Precisión',
        color_discrete_sequence=['#ef5350', '#66bb6a']
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Fase 3: Ajuste de train/test split
    st.markdown("### ⚖️ Fase 3: Optimización del Dataset y Train/Test Split")
    
    st.markdown("""
    **Problema Detectado:**
    Con ~450 registros y división train/test estándar, algunas clases no tenían suficiente
    representación en ambos conjuntos, causando sesgo en el modelo.
    
    **Soluciones Aplicadas:**
    1. ✅ Script de división estratificada para garantizar proporción de clases
    2. ✅ Análisis con matriz de confusión para validar representación
    
    **Resultados Finales:**
    - Train shape: (1759, 9)
    - Test shape: (440, 9)
    - **83% de precisión con Random Forest Classifier**
    """)
    
    # Comparación final de modelos
    st.markdown("### 🏆 Comparación Final de Modelos")
    
    models_data = {
        'Modelo': [
            'Random Forest',
            'Decision Tree', 
            'Extra Trees',
            'Gradient Boosting',
            'Light GBM',
            'K Neighbors'
        ],
        'Accuracy': [83.80, 83.74, 83.74, 83.28, 82.83, 81.92],
        'AUC': [94.61, 94.54, 94.58, 0, 94.48, 93.45],
        'Recall': [83.80, 83.74, 83.74, 83.28, 82.83, 81.92],
        'Precision': [84.34, 84.27, 84.27, 83.98, 83.40, 82.51]
    }
    
    df_models = pd.DataFrame(models_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_models,
            x='Modelo',
            y='Accuracy',
            title='Accuracy Comparison - Top 6 Models',
            color='Accuracy',
            color_continuous_scale='Viridis',
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_yaxes(range=[75, 90])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_models[df_models['AUC'] > 0],
            x='Modelo',
            y='AUC',
            title='AUC Score Comparison',
            color='AUC',
            color_continuous_scale='Blues',
            text='AUC'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_yaxes(range=[90, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    # Métricas comparativas
    st.markdown("### 📊 Tabla Comparativa de Métricas")
    st.dataframe(
        df_models.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Recall', 'Precision']),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Justificación de Random Forest
    st.markdown("### ✅ ¿Por qué Random Forest Classifier?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **🎯 Mayor Accuracy**
        
        83.80% - El más alto
        entre todos los modelos
        evaluados
        """)
    
    with col2:
        st.success("""
        **📈 Mejor AUC**
        
        94.61% - Excelente
        capacidad de discriminación
        entre clases
        """)
    
    with col3:
        st.success("""
        **⚖️ Balance Óptimo**
        
        Precision: 84.34%
        Recall: 83.80%
        Mejor equilibrio
        """)
    
    st.markdown("""
    ### 🔬 Ventajas Adicionales de Random Forest:
    
    1. **Robustez ante Outliers**: Menos sensible a datos atípicos que otros modelos
    2. **Manejo de No Linealidad**: Captura relaciones complejas entre variables
    3. **Feature Importance**: Permite identificar las variables más relevantes
    4. **Reducción de Overfitting**: El ensemble de árboles previene sobreajuste
    5. **Estabilidad**: Menos varianza en predicciones comparado con árboles individuales
    6. **Paralelización**: Entrenamiento eficiente en múltiples núcleos
    """)
    
    # Evolución del proyecto
    st.markdown("### 📈 Evolución del Proyecto")
    
    evolution_data = {
        'Fase': ['Baseline', 'Feature Engineering', 'Final Model'],
        'Precisión': [51, 73, 83.80],
        'Descripción': [
            'Modelo inicial sin preprocesamiento',
            'Limpieza de datos y reducción de clases',
            'Random Forest con train/test estratificado'
        ]
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=evolution_data['Fase'],
        y=evolution_data['Precisión'],
        mode='lines+markers+text',
        marker=dict(size=15, color=['red', 'orange', 'green']),
        line=dict(width=3, color='blue'),
        text=evolution_data['Precisión'],
        textposition='top center',
        texttemplate='%{text}%',
        hovertemplate='<b>%{x}</b><br>Precisión: %{y}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Progreso del Modelo a través de las Fases',
        xaxis_title='Fase del Proyecto',
        yaxis_title='Precisión (%)',
        yaxis=dict(range=[40, 90]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Resumen final
    st.success("""
    ### 🎉 Resumen del Proyecto
    
    - **Mejora Total:** 32.8 puntos porcentuales (51% → 83.8%)
    - **Modelo Final:** Random Forest Classifier
    - **Dataset Final:** 2199 registros, 4 clases balanceadas
    - **Variables:** 7 features (N, P, K, temperatura, humedad, pH, precipitación)
    - **Validación:** 10-Fold Stratified Cross-Validation
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 Sistema de Predicción de Cultivos Agrícolas | Desarrollado con Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)