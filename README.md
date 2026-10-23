# Spotify Tracks · Predicción de popularidad de canciones

**Asignatura:** MLY1101 – Machine Learning · **Evaluación Parcial N°2 (Caso C)**
**Integrantes:** _completar_

Proyecto de Machine Learning que busca anticipar qué canciones serán populares en Spotify a partir de sus características de audio, su duración, si son explícitas y su género. Sigue la metodología CRISP-DM.

---

## Estructura del proyecto

```
├── data/
│   └── Spotify_Tracks_Dataset.csv      # Dataset original (114.000  filas)
├── notebooks/
│   └── EDA.ipynb                       # Notebook ejecutable con todo el desarrollo
├── images/                             # Gráficos generados por el notebook
├── models/                             # Modelos entrenados (se generan al ejecutar)
├── requirements.txt                    # Librerías necesarias
└── README.md                           # Este informe
```

## Instalación de Librerías

```bash
pip install -r requirements.txt    # luego: Run All
```

El notebook usa semilla fija (`SEED = 42`) y rutas relativas, por lo que no requiere modificaciones.

---

## 1. Descripción del problema de negocio

Las plataformas de streaming y los sellos discográficos deben decidir en qué canciones invertir en promoción, playlists y publicidad. Hoy esa decisión depende en gran parte del criterio del equipo y se toma cuando la canción ya está circulando.

**Problema a resolver:** *clasificar si una canción será popular* (`popularity > 70`), usando solo atributos propios de la canción.

El modelo no reemplaza al equipo de promoción: entrega un **ranking de candidatas** para priorizar su atención.

## 2. Objetivos del proyecto

**Objetivo general:** construir un modelo supervisado que identifique canciones con alta probabilidad de ser populares.

**Objetivos específicos**
1. Definir la variable objetivo binaria `is_popular` (`popularity > 70`).
2. Analizar la calidad de los datos y los patrones que distinguen a las canciones populares.
3. Entrenar un modelo **XGBoost** y evaluarlo con métricas adecuadas a un problema desbalanceado.
4. Complementar con una versión de regresión para medir el error en puntos de popularidad.
5. Identificar qué variables influyen más en la popularidad.

## 3. Definición de KPIs

| Tipo | KPI | Qué mide | Por qué importa al negocio |
|---|---|---|---|
| Clasificación (**principal**) | **F1-score** (clase popular) | Equilibrio entre Precision y Recall | Resume la calidad de la decisión en un solo número |
| Clasificación | **Precision** | De las canciones marcadas como populares, cuántas realmente lo son | Evita gastar presupuesto en canciones que no despegarán |
| Clasificación | **Recall** | De las canciones realmente populares, cuántas se detectan | Evita dejar pasar éxitos reales |
| Clasificación (apoyo) | **PR-AUC** | Calidad del ranking sin depender del umbral | Se compara con la prevalencia (~4 %), que es el nivel de un modelo al azar |
| Regresión | **MAE** | Error promedio en puntos de popularidad | Fácil de explicar |
| Regresión | **RMSE** | Igual que MAE, pero castiga más los errores grandes | Detecta errores puntuales grandes |

> **Por qué no se usa *accuracy*:** solo el 4,25 % de las canciones es popular. Un modelo que prediga "nadie será popular" tendría 95,75 % de accuracy y ninguna utilidad.

## 4. Descripción de las fuentes de datos

**Fuente:** dataset público *Spotify Tracks Dataset* (publicado en Kaggle), con atributos de audio obtenidos desde la API de Spotify. Archivo: `data/Spotify_Tracks_Dataset.csv`. _Completar con el enlace exacto usado por el grupo._

**Tamaño:** 114.000 filas × 21 columnas · 114 géneros con exactamente 1.000 canciones cada uno.

| Grupo | Variables | Descripción |
|---|---|---|
| Identificadores | `Unnamed: 0`, `track_id`, `track_name`, `album_name`, `artists` | Identifican la canción; **no se usan** para modelar |
| Target | `popularity` | Puntaje de 0 a 100 (se transforma en `is_popular = popularity > 70`) |
| Audio | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` | Características de audio calculadas por Spotify |
| Estructura | `duration_ms`, `key`, `mode`, `time_signature` | Duración, tonalidad, modo y compás |
| Contenido | `explicit` | Si la canción tiene contenido explícito |
| Género | `track_genre` | Uno de los 114 géneros del dataset |

## 5. Metodología (CRISP-DM)

| Fase | Qué se hizo | Dónde |
|---|---|---|
| 1. Comprensión del negocio | Problema, objetivos y KPIs | Secciones 1–3 de este informe · Notebook §1–2 |
| 2. Comprensión de los datos | Calidad, distribuciones, correlaciones, género y contenido explícito | Sección 6 · Notebook §3–4 |
| 3. Preparación de los datos | Selección de variables, categóricas nativas y split agrupado sin fuga de datos | Sección 7 · Notebook §5 |
| 4. Modelado | XGBoost de clasificación (principal) y de regresión (complementario), con líneas base | Sección 8 · Notebook §6 |
| 5. Evaluación | Precision, Recall, F1, PR-AUC, MAE y RMSE sobre el conjunto de test | Sección 9 · Notebook §6–7 |
| 6. Despliegue | Modelos guardados en `models/`; uso propuesto como ranking de priorización | Sección 10 |

## 6. Preparación y análisis exploratorio de los datos (EDA)

### 6.1 Calidad de los datos

| Hallazgo | Detalle | Decisión |
|---|---|---|
| Nulos | 1 fila sin `artists`, `album_name` ni `track_name` (género k-pop) | Sin impacto: esas columnas no se usan |
| Canciones repetidas | 40.900 filas (35,9 %) comparten `track_id`: la misma canción aparece en varios géneros. Hay 89.741 canciones únicas | El split se hace agrupando por `track_id` para evitar fuga de datos |
| Ceros en `popularity` | 16.020 filas (14,1 %) | Se mantienen; se declaran como limitación (ver 6.2) |
| Ceros en audio | `tempo = 0` (157), `time_signature = 0` (163), `duration_ms = 0` (1) | Marginales; XGBoost los tolera |
| Géneros balanceados | Todos tienen 1.000 filas | La muestra es artificial; no refleja la proporción real de cada género en Spotify |

### 6.2 Variable objetivo: desbalanceada

Solo **4.846 filas (4,25 %)** superan el umbral de 70, y corresponden a **2.734 canciones únicas**.

![Target](images/01_target.png)

Los ceros de `popularity` se concentran en algunos géneros (jazz 68 %, iranian 66 %, romance 64 %, soul 61 %, latin 59 %, country 59 %). Es poco probable que sean canciones realmente sin oyentes: podrían ser datos sin puntuar. Esto introduce ruido en el target.

![Ceros por género](images/02_ceros_por_genero.png)

### 6.3 Variables de audio

Las canciones populares tienden a ser más fuertes (`loudness` −6,6 vs −8,3 dB), algo más bailables (0,62 vs 0,56) y menos acústicas (0,22 vs 0,32) e instrumentales (0,04 vs 0,16). Sin embargo, **las diferencias son pequeñas y ninguna correlación de Spearman con `popularity` supera |0,08|**: ninguna variable de audio predice por sí sola.

![Audio por clase](images/03_audio_por_clase.png)
![Correlaciones](images/04_correlacion.png)

### 6.4 Género y contenido explícito

El género es la variable más informativa: `pop` alcanza 28 % de canciones populares, seguido de `dance` (24 %) y `electro` (22 %), mientras que **26 de los 114 géneros no tienen ninguna canción popular**. Las canciones explícitas son populares más del doble de veces (8,5 % vs 3,9 %).

![Género y explícito](images/05_popular_por_genero.png)

## 7. Preparación de los datos para el modelamiento

- **Variables descartadas:** `Unnamed: 0`, `track_id`, `track_name`, `album_name` y `artists`. Usar `artists` haría que el modelo aprenda *quién* es famoso en lugar de *qué* hace popular a una canción.
- **Variables usadas:** 9 de audio, `duration_ms`, `key`, `mode`, `time_signature`, `explicit` y `track_genre`.
- **Transformaciones:** `explicit` pasa a 0/1 y `track_genre` a categórica nativa de XGBoost. No se requiere escalado ni *one-hot* al usar árboles.
- **Partición 60/20/20** (train / validación / test), **agrupada por `track_id` y estratificada por clase**. El notebook verifica que ninguna canción quede en dos conjuntos.
- **Validación** se usa para detener el entrenamiento (*early stopping*) y elegir el umbral; **test** se usa una sola vez.

## 8. Modelado

**Clasificación (modelo principal) – `XGBClassifier`.** Captura relaciones no lineales entre variables y maneja el género como categórica, algo necesario porque ninguna variable de audio explica sola la popularidad.

| Parámetro | Valor | Motivo |
|---|---|---|
| `scale_pos_weight` | negativos / positivos (~22,5) | Compensa el desbalance |
| `eval_metric` | `aucpr` | Adecuada para clases desbalanceadas |
| `early_stopping_rounds` | 50 | Evita sobreajuste |
| `learning_rate`, `max_depth` | 0,05 · 6 | Valores moderados de partida (sin afinar) |
| `subsample`, `colsample_bytree` | 0,8 · 0,8 | Regularización por muestreo |

**Umbral de decisión:** se elige el que maximiza F1 en validación, porque con `scale_pos_weight` el corte de 0,5 rara vez es el óptimo.

**Regresión (complementario) – `XGBRegressor`.** Misma configuración base, prediciendo `popularity`; reporta el error en puntos.

**Líneas base:** predecir siempre "no popular" (clasificación) y siempre el promedio (regresión).

## 9. Resultados

> Esta sección se completa al ejecutar `notebooks/EDA.ipynb`. Copiar los valores de las tablas del notebook.

| Modelo | Umbral | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Línea base (siempre "no popular") | – | _completar_ | _completar_ | 0,000 | _completar_ |
| XGBoost (umbral 0,5) | 0,5 | _completar_ | _completar_ | _completar_ | _completar_ |
| XGBoost (umbral óptimo) | _completar_ | _completar_ | _completar_ | _completar_ | _completar_ |

| Modelo | MAE | RMSE |
|---|---|---|
| Línea base (promedio) | _completar_ | _completar_ |
| XGBoost | _completar_ | _completar_ |

**Interpretación:** _completar_ (¿supera a la línea base y a la prevalencia de ~4 %?, qué significan Precision y Recall con el umbral elegido, qué variables pesan más).

## 10. Consideraciones éticas, limitaciones y próximos pasos

**Ética y sesgos**
- Se excluye `artists` para no reproducir el sesgo de "los ya famosos siempre serán populares", que perjudicaría a artistas nuevos.
- La popularidad refleja lo que Spotify ya promociona; el modelo podría reforzar géneros dominantes (pop, dance) en desmedro de otros.
- Usar el modelo como ranking de apoyo, no como decisor automático.

**Limitaciones**
- Solo hay atributos de audio: no hay información de artista, marketing, fecha de lanzamiento ni playlists, que explican gran parte de la popularidad.
- Los ceros de `popularity` pueden ser datos sin puntuar y distorsionar el target.
- La muestra por género es artificialmente pareja (1.000 por género).

**Próximos pasos:** segundo modelo supervisado para comparar con XGBoost, técnica no supervisada (clustering de canciones) y afinamiento de hiperparámetros.
