# Face Age

Aplicacion web en Flask para reconocimiento facial en tiempo real desde el navegador. La interfaz captura video desde la camara del visitante, envia frames al backend y muestra en pantalla una estimacion de edad, genero y emocion para cada rostro detectado.

## Caracteristicas

- Captura de camara directamente desde el navegador.
- Analisis de rostros en tiempo real con OpenCV.
- Estimacion de edad, genero y emocion.
- Overlay visual sobre el video con cajas y etiquetas de resultados.
- Descarga de capturas del video con el overlay dibujado.
- Registro persistente de eventos y errores en `logs/face-age.log`.
- Control en pantalla para ajustar el refresco del analisis entre 1 y 10 segundos.
- Un solo boton para encender o apagar la camara segun el estado actual.
- El valor del refresco se guarda en `localStorage` y se restaura al recargar la pagina.

## Requisitos

- Python 3.10 o superior.
- Acceso a una camara web en el navegador.
- Dependencias instaladas desde `requirements.txt` o el entorno definido por `pyproject.toml`.

## Instalacion

### Con venv y pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Con uv

```bash
uv sync
source .venv/bin/activate
```

## Ejecucion

```bash
python app.py
```

Por defecto la aplicacion levanta en `http://127.0.0.1:5000`.

### Opciones disponibles

```bash
python app.py --host 0.0.0.0 --port 8000 --models-dir models
```

- `--host`: direccion donde escucha Flask.
- `--port`: puerto del servidor.
- `--models-dir`: directorio donde se descargan y guardan los modelos.

## Flujo de uso

1. Abrir la pagina en el navegador.
2. Dar permisos de camara.
3. Hacer clic en `Encender camara` y volver a pulsar el mismo boton para apagarla.
4. Ver los resultados superpuestos sobre el video.
5. Usar `Capturar` para descargar una imagen con el overlay.

## Modelos

En el primer arranque, la aplicacion descarga automaticamente los modelos necesarios dentro de `models/` si no estan presentes:

- `deploy_age2.prototxt`
- `age_net.caffemodel`
- `deploy_gender2.prototxt`
- `gender_net.caffemodel`
- `emotion-ferplus-8.onnx`

## Variables de entorno

Estas variables ayudan cuando el frontend y el backend no comparten el mismo origen en produccion:

- `FACE_AGE_ALLOWED_ORIGIN`: origen permitido por CORS para navegadores. Por defecto usa `*`.

Ejemplo:

```bash
export FACE_AGE_ANALYZE_URL="https://api.ejemplo.com/analyze"
export FACE_AGE_CLIENT_LOG_URL="https://api.ejemplo.com/client-log"
export FACE_AGE_ALLOWED_ORIGIN="https://frontend.ejemplo.com"
python app.py
```

## Logs

El backend escribe eventos y errores en `logs/face-age.log` usando rotacion de archivos. Ahí quedan registrados:

- Apertura de camara desde el navegador.
- Errores al abrir la camara.
- Errores al analizar frames.
- Eventos enviados desde el cliente con contexto basico.

## Estructura del proyecto

- `app.py`: servidor Flask, logging y endpoints HTTP.
- `main.py`: carga de modelos y analisis de imagenes.
- `templates/index.html`: plantilla HTML principal.
- `static/css/style.css`: estilos de la interfaz.
- `static/js/app.js`: logica del navegador.
- `models/`: modelos descargados por la aplicacion.
- `captures/`: capturas generadas por la app.
- `logs/`: archivo de log rotativo del backend.

## Notas de despliegue

- Si la app se publica detras de un proxy o bajo otro dominio, usa las variables de entorno para fijar las URLs del backend.
- La camara del navegador requiere contexto seguro en muchos entornos de produccion.
- Si el analisis devuelve errores, revisa primero `logs/face-age.log`.