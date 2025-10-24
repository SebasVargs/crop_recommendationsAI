from flask import Flask, request, jsonify, send_file
from pycaret.classification import load_model
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Rutas de archivos
CSV_PATH = os.path.join(DATA_DIR, 'crop_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model')
PREDICTIONS_LOG_PATH = os.path.join(DATA_DIR, 'predictions_log.json')

# Variables globales para caché
_model_cache = None
_data_cache = None

def get_model():
    """Carga y cachea el modelo entrenado con PyCaret"""
    global _model_cache
    if _model_cache is None:
        if os.path.exists(MODEL_PATH + '.pkl'):
            try:
                _model_cache = load_model(MODEL_PATH)
                print(f"✅ Modelo cargado correctamente desde: {MODEL_PATH}.pkl")
                print(f"   Tipo: {type(_model_cache)}")
            except Exception as e:
                print(f"❌ Error al cargar modelo: {e}")
                _model_cache = None
        else:
            print(f"⚠️ Archivo de modelo no encontrado: {MODEL_PATH}.pkl")
            _model_cache = None
    return _model_cache

def get_data():
    """Carga y cachea los datos"""
    global _data_cache
    if _data_cache is None:
        if os.path.exists(CSV_PATH):
            try:
                _data_cache = pd.read_csv(CSV_PATH)
                print(f"✅ Datos cargados: {len(_data_cache)} registros")
            except Exception as e:
                print(f"❌ Error al cargar datos: {e}")
                return None
        else:
            print(f"⚠️ Archivo CSV no encontrado: {CSV_PATH}")
            return None
    return _data_cache

def reload_data():
    """Recarga los datos del CSV"""
    global _data_cache
    _data_cache = None
    return get_data()

def reload_model():
    """Recarga el modelo"""
    global _model_cache
    _model_cache = None
    return get_model()

# ==================== ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor"""
    model_file = MODEL_PATH + ".pkl"
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': os.path.exists(model_file),
        'data_loaded': os.path.exists(CSV_PATH)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Realiza predicción con el modelo"""
    try:
        model = get_model()
        if model is None:
            return jsonify({'error': 'Modelo no cargado. Coloca crop_model.pkl en la carpeta models/'}), 400
        
        data = request.json
        print(f"📥 Datos recibidos para predicción: {data}")

        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label_encoded']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo faltante: {field}'}), 400
        
        # Campo opcional enviado desde el frontend (LabelEncoder del cultivo)
        selected_label = data.get('label_encoded', None)
        if selected_label is not None:
            print(f"🎯 LabelEncoder recibido {selected_label}")

        label = data.get('label', None)
        label_encoded = float(data['label_encoded'])
        
        input_data = pd.DataFrame([[
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall'],
            data['label_encoded']
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label_encoded'])


        print(f"📊 Input shape: {input_data.shape}")
        print(f"📊 Input data: {input_data}")
        
        prediction = model.predict(input_data)
        pred_value = prediction[0]

        print(f"🔮 Predicción raw: {prediction}")
        print(f"🔮 Tipo de predicción: {type(pred_value)}")
        
        crop_mapping_numeric = {
            2: 'coconut',
            4: 'cotton',
            5: 'maize',
            6: 'rice'
        }

        if isinstance(pred_value, (int, np.integer, float)):
            predicted_crop = crop_mapping_numeric.get(int(pred_value), f'unknown_{pred_value}')
        elif isinstance(pred_value, str):
            predicted_crop = pred_value
        else:
            predicted_crop = str(pred_value)

        print(f"✅ Cultivo predicho: {predicted_crop}")

        log_prediction(data, predicted_crop)
        
        result = {
            'prediction': predicted_crop,
            'prediction_raw': str(pred_value),
            'label_selected': selected_label,
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        }

        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)[0]
                result['probabilities'] = {
                    crop_mapping_numeric.get(i, f'class_{i}'): float(prob)
                    for i, prob in enumerate(probabilities)
                }
                print(f"📊 Probabilidades: {result['probabilities']}")
        except Exception as e:
            print(f"⚠️ No se pudieron obtener probabilidades: {e}")

        return jsonify(result), 200

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error en predicción: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/data/summary', methods=['GET'])
def data_summary():
    """Retorna resumen estadístico de los datos"""
    try:
        df = get_data()
        if df is None:
            return jsonify({'error': 'Datos no cargados. Coloca crop_data.csv en la carpeta data/'}), 400
        
        # Filtrar clases principales si existen
        if 'label' in df.columns:
            valid_labels = ['rice', 'maize', 'cotton', 'coconut']
            df_filtered = df[df['label'].isin(valid_labels)]
            
            if len(df_filtered) == 0:
                df_filtered = df  # Si no hay coincidencias, usar todo el dataset
        else:
            df_filtered = df
        
        summary = {
            'total_records': len(df_filtered),
            'classes': {},
            'statistics': {},
            'columns': list(df_filtered.columns)
        }
        
        # Distribución de clases
        if 'label' in df_filtered.columns:
            summary['classes'] = df_filtered['label'].value_counts().to_dict()
        
        # Estadísticas por variable numérica
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for col in numeric_cols:
            if col in df_filtered.columns:
                col_data = df_filtered[col].dropna()
                if len(col_data) > 0:
                    summary['statistics'][col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'median': float(col_data.median())
                    }
        
        return jsonify(summary), 200
    
    except Exception as e:
        print(f"❌ Error en data_summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/analysis', methods=['POST'])
def data_analysis():
    """Retorna análisis específico de una variable"""
    try:
        df = get_data()
        if df is None:
            return jsonify({'error': 'Datos no cargados'}), 400
        
        data = request.json
        variable = data.get('variable', 'N')
        
        if variable not in df.columns:
            return jsonify({'error': f'Variable {variable} no encontrada'}), 400
        
        # Filtrar clases principales si existe la columna label
        if 'label' in df.columns:
            valid_labels = ['rice', 'maize', 'cotton', 'coconut']
            df_filtered = df[df['label'].isin(valid_labels)]
            
            # Análisis por cultivo
            analysis = {}
            for crop in valid_labels:
                crop_data = df_filtered[df_filtered['label'] == crop][variable].dropna()
                if len(crop_data) > 0:
                    analysis[crop] = {
                        'mean': float(crop_data.mean()),
                        'min': float(crop_data.min()),
                        'max': float(crop_data.max()),
                        'std': float(crop_data.std()),
                        'median': float(crop_data.median()),
                        'count': int(len(crop_data))
                    }
        else:
            # Si no hay columna label, solo estadísticas generales
            col_data = df[variable].dropna()
            analysis = {
                'general': {
                    'mean': float(col_data.mean()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'std': float(col_data.std()),
                    'median': float(col_data.median()),
                    'count': int(len(col_data))
                }
            }
        
        return jsonify(analysis), 200
    
    except Exception as e:
        print(f"❌ Error en data_analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/correlation', methods=['GET'])
def correlation_matrix():
    """Retorna matriz de correlación"""
    try:
        df = get_data()
        if df is None:
            return jsonify({'error': 'Datos no cargados'}), 400
        
        # Filtrar clases principales si existe la columna label
        if 'label' in df.columns:
            valid_labels = ['rice', 'maize', 'cotton', 'coconut']
            df_filtered = df[df['label'].isin(valid_labels)]
        else:
            df_filtered = df
        
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Filtrar solo las columnas que existen
        available_cols = [col for col in numeric_cols if col in df_filtered.columns]
        
        if len(available_cols) == 0:
            return jsonify({'error': 'No se encontraron columnas numéricas'}), 400
        
        corr_matrix = df_filtered[available_cols].corr()
        
        return jsonify(corr_matrix.to_dict()), 200
    
    except Exception as e:
        print(f"❌ Error en correlation_matrix: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/export', methods=['GET'])
def export_data():
    """Exporta los datos filtrados"""
    try:
        df = get_data()
        if df is None:
            return jsonify({'error': 'Datos no cargados'}), 400
        
        if 'label' in df.columns:
            valid_labels = ['rice', 'maize', 'cotton', 'coconut']
            df_filtered = df[df['label'].isin(valid_labels)]
        else:
            df_filtered = df
        
        # Crear archivo temporal
        temp_path = os.path.join(DATA_DIR, 'export_temp.csv')
        df_filtered.to_csv(temp_path, index=False)
        
        return send_file(
            temp_path, 
            as_attachment=True, 
            download_name='crop_data_filtered.csv',
            mimetype='text/csv'
        )
    
    except Exception as e:
        print(f"❌ Error en export_data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predictions/log', methods=['GET'])
def get_predictions_log():
    """Obtiene el log de predicciones"""
    try:
        if os.path.exists(PREDICTIONS_LOG_PATH):
            with open(PREDICTIONS_LOG_PATH, 'r') as f:
                logs = json.load(f)
            return jsonify(logs), 200
        else:
            return jsonify([]), 200
    
    except Exception as e:
        print(f"❌ Error en predictions_log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/list', methods=['GET'])
def list_images():
    """Lista todas las imágenes subidas"""
    try:
        images = []
        if os.path.exists(IMAGES_DIR):
            for filename in os.listdir(IMAGES_DIR):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    filepath = os.path.join(IMAGES_DIR, filename)
                    images.append({
                        'filename': filename,
                        'path': filepath,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    })
        
        # Ordenar por fecha de modificación (más reciente primero)
        images.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify(images), 200
    
    except Exception as e:
        print(f"❌ Error en list_images: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    """Retorna una imagen específica"""
    try:
        filepath = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath)
        else:
            return jsonify({'error': 'Imagen no encontrada'}), 404
    
    except Exception as e:
        print(f"❌ Error en get_image: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== FUNCIONES AUXILIARES ====================

def log_prediction(input_data, prediction):
    """Guarda el log de predicciones"""
    try:
        # Cargar logs existentes
        if os.path.exists(PREDICTIONS_LOG_PATH):
            with open(PREDICTIONS_LOG_PATH, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Agregar nueva predicción
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': prediction
        }
        logs.append(log_entry)
        
        # Mantener solo las últimas 1000 predicciones
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Guardar logs
        with open(PREDICTIONS_LOG_PATH, 'w') as f:
            json.dump(logs, f, indent=2)
    
    except Exception as e:
        print(f"⚠️ Error guardando log: {e}")

# ==================== INICIALIZACIÓN ====================

if __name__ == '__main__':
    print("=" * 70)
    print("🌾 BACKEND DE PREDICCIÓN DE CULTIVOS AGRÍCOLAS")
    print("=" * 70)
    print(f"📁 Directorio base: {BASE_DIR}")
    print(f"📁 Directorio de datos: {DATA_DIR}")
    print(f"📁 Directorio de modelos: {MODEL_DIR}")
    print(f"📁 Directorio de imágenes: {IMAGES_DIR}")
    print("=" * 70)
    
    # Verificar archivos existentes
    print("\n🔍 VERIFICANDO ARCHIVOS...")
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modelo encontrado: {MODEL_PATH}")
        model = get_model()
        if model is not None:
            print(f"   Tipo: {type(model)}")
            if hasattr(model, 'n_features_in_'):
                print(f"   Features esperadas: {model.n_features_in_}")
            else:
                print("   ⚠️  Este modelo no tiene el atributo 'n_features_in_' (posiblemente no es un modelo de sklearn)")
    else:
        print(f"❌ Modelo NO encontrado: {MODEL_PATH}")
        print("   ⚠️  Coloca el archivo 'crop_model.pkl' en la carpeta 'models/'")
    
    print()
    
    if os.path.exists(CSV_PATH):
        print(f"✅ Datos encontrados: {CSV_PATH}")
        df = get_data()
        if df is not None:
            print(f"   Registros: {len(df)}")
            print(f"   Columnas: {list(df.columns)}")
            if 'label' in df.columns:
                print(f"   Clases: {df['label'].unique().tolist()}")
    else:
        print(f"❌ Datos NO encontrados: {CSV_PATH}")
        print("   ⚠️  Coloca el archivo 'crop_data.csv' en la carpeta 'data/'")
    
    print()
    
    # Verificar imágenes
    if os.path.exists(IMAGES_DIR):
        image_files = [f for f in os.listdir(IMAGES_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if image_files:
            print(f"✅ Imágenes encontradas: {len(image_files)}")
            for img in image_files[:5]:  # Mostrar solo las primeras 5
                print(f"   - {img}")
            if len(image_files) > 5:
                print(f"   ... y {len(image_files) - 5} más")
        else:
            print(f"ℹ️  No hay imágenes en: {IMAGES_DIR}")
            print("   💡 Coloca archivos .png, .jpg o .jpeg para visualizaciones")
    
    print("=" * 70)
    print("🚀 INICIANDO SERVIDOR EN http://localhost:5001")
    print("=" * 70)
    print("💡 Mantén esta terminal abierta mientras usas la aplicación")
    print("💡 Presiona Ctrl+C para detener el servidor")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)