from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Cargar el DataFrame desde el archivo pickle
pickle_path = 'checkpoints/sin_bajos_ag.pkl'
with open(pickle_path, 'rb') as file:
    sin_bajos_ag = pickle.load(file)

# Verificar la carga correcta
print(sin_bajos_ag.head())
# Preparar los datos
X = sin_bajos_ag[['DineroEquipo', 'Granadas', 'Kills']]
y = sin_bajos_ag['GanaRonda']

# Escalar los datos
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = logistic_model.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Guardar el modelo y el scaler en archivos pickle
with open('checkpoints/logistic_model.pkl', 'wb') as model_file:
    pickle.dump(logistic_model, model_file)

with open('checkpoints/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Modelo y scaler guardados en archivos pickle con éxito.")
