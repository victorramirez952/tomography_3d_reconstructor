# Tomography 3D Reconstruction

Sistema para generar cortes elipsoidales y reconstruir modelos 3D a partir de datos de tomografía.

---

## Instalación

Instalar las librerías requeridas:

```bash
pip install -r requirements.txt
```

---

## Generador de Cortes Elipsoidales

**Archivo Principal:** `simple_generator.py`

Genera cortes de medio elipsoide a partir de una imagen de máscara base para reconstrucción 3D.

### Configuración

Editar las variables en la función `main()` de `simple_generator.py`:

| Variable | Descripción |
|----------|-------------|
| `mask_path` | Ruta a la imagen de máscara base (formato PNG) |
| `n_slices` | Número de cortes a generar (entero) |
| `output_directory` | Ruta donde se guardarán las imágenes generadas |
| `num_start` | Número inicial para la secuencia de imágenes (entero, puede ser negativo) |
| `increase` | Dirección de la secuencia de numeración (booleano) |

**Parámetro Increase:**
- `True`: Secuencia ascendente (genera lado izquierdo/inicial del modelo 3D)
- `False`: Secuencia descendente (genera lado derecho/final del modelo 3D)

### Ejecución

```bash
python simple_generator.py
```

---

## Reconstrucción 3D

**Archivo Principal:** `tomography_3d_reconstruction.py`

Reconstruye modelos 3D a partir de máscaras de cortes de tomografía.

### Configuración

Editar las variables en `config.py`:

#### Dimensiones Físicas

| Variable | Descripción |
|----------|-------------|
| `X_LENGTH_MM` | Ancho de la máscara en milímetros |
| `Y_LENGTH_MM` | Altura de la máscara en milímetros |
| `TOTAL_DEPTH_MM` | Profundidad del escaneo de tomografía/resonancia |

#### Estructura de Datos

| Variable | Descripción |
|----------|-------------|
| `DATA_PATH` | Ruta a carpeta que contiene tres subdirectorios |

**Subdirectorios Requeridos:**
- `Section_0`: Máscaras para cerrar lado izquierdo/inicial del modelo 3D
- `Section_1`: Máscaras obtenidas de la tomografía/resonancia
- `Section_2`: Máscaras para cerrar lado derecho/final del modelo 3D

#### Control de Salida

| Variable | Descripción |
|----------|-------------|
| `SHOW_3D_VISUALIZATION` | Mostrar visualización 3D (True/False) |
| `EXPORT_OBJ_MODEL` | Exportar modelo 3D como archivo OBJ (True/False) |
| `OBJ_FILENAME` | Ruta de salida para archivo OBJ |
| `INTERACTIVE_HTML` | Ruta de salida para visualización HTML interactiva |

**Rutas de Ejemplo:**
```python
OBJ_FILENAME = "model_3d.obj"
OBJ_FILENAME = "/home/user/Documents/model_3d.obj"

INTERACTIVE_HTML = "visualization.html"
INTERACTIVE_HTML = "/home/user/Documents/visualization.html"
```

### Ejecución

```bash
python tomography_3d_reconstruction.py
```

---

## Estructura del Proyecto

```
config.py                          Archivo de configuración
simple_generator.py                Generador de cortes
tomography_3d_reconstruction.py    Script principal de reconstrucción
ellipsoid_slice_generator.py       Lógica de generación de elipsoide
image_loader.py                    Carga de imágenes de máscaras
voxel_processor.py                 Procesamiento de datos de voxeles
surface_extractor.py               Extracción de superficie (marching cubes)
visualizer.py                      Visualización 3D
volume_calculator.py               Cálculos de volumen
obj_exporter.py                    Exportación de archivos OBJ
```
