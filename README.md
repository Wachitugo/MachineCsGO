### Machine Learning Cs GO
## Instrucciones para Iniciar
### Clonar el Repositorio
Abre la terminal o el símbolo del sistema y navega al directorio donde deseas clonar el repositorio. Luego, ejecuta el siguiente comando para clonar el repositorio desde Git:
```python
git clone https://github.com/Wachitugo/MachineCsGO.git             
```
El proyecto se estructura de la siguiente manera:

```python
MachineCsGO
-- checkpoints
---- logistic_model.pkl
---- scaler.pkl
---- sin_bajos_ag.pkl
-- dataset
---- Anexo_ET_demo_round_traces.csv
-- model
---- data.py
---- model.py
-- templates
---- index.html
---- result.html
-- readme.md
-- requirements.txt
-- script.py
```
### Crear Entorno Virtual
1. Navega al directorio del proyecto en la terminal.
2. Crea un entorno virtual ejecutando:
3. Copiar código
```python
python -m venv env
```
Esto creará un nuevo directorio env en tu proyecto que contendrá el entorno virtual.

### Instalar Dependencias
Activa el entorno virtual:

En Windows:
```bash
.\env\Scripts\Activate.ps1
```
En macOS y Linux:

```bash
env/bin/activate
```
Instala las dependencias del proyecto utilizando el archivo requirements.txt:
```bash
pip install -r requirements.txt
```
### Ejecutar el proyecto
Ejecuta el script Python principal con:

```bash
python script.py
```
