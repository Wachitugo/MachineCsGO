from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Cargar el modelo y el scaler desde archivos pickle
model_path = 'checkpoints/logistic_model.pkl'
scaler_path = 'checkpoints/scaler.pkl'

with open(model_path, 'rb') as model_file:
    logistic_model = pickle.load(model_file)
    
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Obtener los datos del formulario
        dinero_equipo = float(request.form['dinero_equipo']) if request.form['dinero_equipo'].strip() else None
        granadas = float(request.form['granadas']) if request.form['granadas'].strip() else None
        kills = float(request.form['kills']) if request.form['kills'].strip() else None
        
        # Validar que todos los campos estén completos
        if dinero_equipo is None or granadas is None or kills is None:
            return render_template('error.html', message='Todos los campos son obligatorios')
        
        # Crear el array de entrada
        input_data = np.array([[dinero_equipo, granadas, kills]])
        
        # Escalar los datos de entrada
        input_data_scaled = scaler.transform(input_data)
        
        # Hacer la predicción
        prediction = logistic_model.predict(input_data_scaled)
        
        # Convertir la predicción a una respuesta legible
        result = 'Ganar' if prediction[0] == 1 else 'Perder'
        
        return render_template('result.html', result=result)
    
    except ValueError as e:
        return render_template('error.html', message='Error al procesar los datos: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
