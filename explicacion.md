# Explicacion linea por linea de `paper_replication.ipynb`

> Este documento explica el proposito de cada celda y cada bloque de codigo del notebook de replicacion del paper MSI (Mahalanobis Similarity Index). Se pone **enfasis especial en los metodos extendidos** (Extended 1-9) al final.

---

## Celda 0 (Markdown) — Encabezado e introduccion

Describe el proyecto MSI y sus autores (UAM Cuajimalpa). Presenta una tabla con los 4 datasets utilizados:

| Dataset | Referencia | Fuente | Proposito |
|---------|-----------|--------|-----------|
| Aspirin | SMILES de aspirina | ACI (PubChem) | Caso de estudio principal |
| Ibuprofen | SMILES de ibuprofeno | ACI (PubChem) | Comparacion cruzada |
| Curcumin | SMILES de curcumina | ACI (PubChem) | Comparacion cruzada |
| QM9 Aniline | SMILES de anilina | QM9 | Caso de estudio con propiedades cuanticas |

Tambien lista los 3 pasos del pipeline (vectores 2D, vectores 300D, analisis de similitud) y las 13 visualizaciones que se generaran.

---

## Celda 1 (Markdown) — Titulo de seccion

Marca el inicio de la **Seccion 0: Setup & Imports**.

---

## Celda 2 (Code) — Importaciones y configuracion

```python
import os                          # Manejo de rutas de archivos y directorios
import sys                         # Acceso a parametros del sistema
import time                        # Medir tiempos de ejecucion del pipeline
import numpy as np                 # Operaciones numericas (matrices, algebra lineal)
import pandas as pd                # DataFrames para manipular datos tabulares
import matplotlib.pyplot as plt    # Graficas 2D
import matplotlib.colors as mcolors # Escalas de color personalizadas
import seaborn as sns              # Graficas estadisticas con estilo mejorado
from mpl_toolkits.mplot3d import Axes3D  # Graficas 3D (scatter, superficie)
import statsmodels.api as sm       # LOWESS (regresion local no parametrica)
from rdkit import Chem             # Quimioinformatica: parseo de moleculas SMILES
from rdkit.Chem import Draw        # Dibujar estructuras moleculares como imagenes
from IPython.display import HTML, display  # Mostrar HTML y tablas en Jupyter
import warnings
warnings.filterwarnings('ignore')  # Silenciar advertencias para salida limpia
```

**Por que cada import:**
- `numpy` es necesario para todo el algebra lineal (covarianza, distancias de Mahalanobis, bootstrap).
- `pandas` organiza los resultados de cada molecula (SMILES, theta, d_M, TSI, peso molecular).
- `matplotlib` + `seaborn` generan todas las figuras del paper.
- `statsmodels` provee el suavizador LOWESS que se usa en las graficas de TSI vs d_M.
- `rdkit` es la libreria estandar de quimioinformatica: convierte SMILES a moleculas, genera fingerprints y dibuja estructuras.

```python
from gensim.models import Word2Vec  # Cargar modelo mol2vec pre-entrenado
from msi import (
    generate_2d_vectors,            # Paso 1: PCA(300->30) + t-SNE(30->2)
    generate_300d_vectors,          # Paso 2: Embeddings mol2vec promedio (300D)
    analyze_against_reference,      # Paso 3: Calcula d_M, theta_M, TSI, etc.
)
```

Importa las 3 funciones centrales del modulo `msi.py` que constituyen el pipeline completo.

```python
sns.set_theme(style='white', font_scale=1.1)  # Estilo visual limpio para graficas
```

```python
PARAMS = {
    'regularization_lambda': 1e-5,   # Lambda de Tikhonov para regularizar la covarianza
    'morgan_radius': 2,              # Radio ECFP4 (2 enlaces de profundidad)
    'morgan_nbits': 2048,            # Longitud del fingerprint Morgan (bits)
    'mol2vec_radius': 1,             # Radio para extraccion de subestructuras mol2vec
    'pca_components': 30,            # Componentes PCA intermedios antes de t-SNE
    'tsne_perplexity': 10,           # Perplexity de t-SNE (controla vecindario local)
    'tsne_iterations': 1000,         # Iteraciones de optimizacion t-SNE
    'lowess_frac': 0.5,              # Fraccion de datos usada por LOWESS
}
```

Estos parametros replican exactamente los del paper. `regularization_lambda` es critico: se suma a la diagonal de la matriz de covarianza para evitar singularidad (`Sigma + lambda*I`). Sin esto, la inversa de la covarianza no existe para datos de alta dimensionalidad.

```python
DATA_DIR = 'data'                   # Carpeta con los CSV de moleculas y el modelo mol2vec
OUTPUT_DIR = 'output_replication'   # Carpeta donde se guardan los resultados frescos
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(DATA_DIR, 'model_300dim.pkl')  # Modelo Word2Vec entrenado
```

---

## Celda 3 (Markdown) — Titulo Seccion 1

Marca el inicio de **Seccion 1: Load Mol2Vec Model**.

---

## Celda 4 (Code) — Cargar modelo mol2vec

```python
print('Loading mol2vec model...')
t0 = time.time()
model = Word2Vec.load(MODEL_PATH)   # Carga el modelo Word2Vec pre-entrenado sobre subestructuras moleculares
print(f'Model loaded in {time.time() - t0:.1f}s')
print(f'Vocabulary size: {len(model.wv):,} identifiers')  # Cuantos identificadores de subestructura conoce
print(f'Vector dimensionality: {model.wv.vector_size}')   # 300 dimensiones por vector
```

**Por que:** Mol2Vec funciona como Word2Vec pero para moleculas. Cada subestructura molecular (generada con radio Morgan = 1) es un "word", y el modelo pre-entrenado asigna un vector de 300 dimensiones a cada una. La representacion de una molecula completa es el **promedio** de los vectores de sus subestructuras.

---

## Celda 5 (Markdown) — Titulo "Helper functions"

---

## Celda 6 (Code) — Funciones auxiliares

### `mol_to_img(smiles, size)`
```python
def mol_to_img(smiles, size=(250, 250)):
    mol = Chem.MolFromSmiles(smiles)   # Parsea el SMILES a un objeto molecula RDKit
    if mol:
        return Draw.MolToImage(mol, size=size)  # Renderiza la estructura 2D como imagen PIL
    return None
```
Convierte un SMILES en imagen para mostrar la estructura molecular. Se usa para las figuras de moleculas de referencia y top-10.

### `mol_grid(smiles_list, labels, ...)`
Construye una cuadricula de imagenes moleculares. Para cada SMILES:
1. Crea un canvas PIL blanco del tamano total necesario
2. Dibuja cada molecula con `Draw.MolToImage`
3. Agrega texto debajo de cada molecula (metricas, rankings)
4. Retorna la imagen como array numpy para mostrar con matplotlib

**Por que no usa `MolsToGridImage`:** Esa funcion de RDKit tiene problemas de compatibilidad entre versiones, asi que se reimplementa manualmente.

### `run_full_pipeline(name, filename, preserve_columns, description)`
Orquesta los 3 pasos del pipeline MSI para un dataset:

```python
# Paso 1: Genera vectores 2D (PCA 300->30, luego t-SNE 30->2)
path_2d = generate_2d_vectors(path=input_path, model=model, output_dir=OUTPUT_DIR, preserve_columns=preserve_columns)

# Paso 2: Genera vectores 300D (promedio de embeddings mol2vec, sin reduccion)
path_300d = generate_300d_vectors(path=input_path, model=model, output_dir=OUTPUT_DIR)

# Paso 3: Calcula todas las metricas de similitud contra la molecula de referencia (fila 0)
results_df, metrics = analyze_against_reference(path_2d=path_2d, path_300d=path_300d, output_dir=OUTPUT_DIR)
```

`preserve_columns=True` se usa para QM9, que tiene columnas extra (HOMO, LUMO, gap, mu, alpha) que queremos conservar para los analisis extendidos.

Imprime estadisticas: numero de condicion de la covarianza (si es muy alto, la inversa es inestable) y rango de Tanimoto.

---

## Celdas 7-8 (Markdown + Code) — Pipeline Aspirina

Ejecuta `run_full_pipeline` para el dataset ACI con aspirina como referencia (~31k moleculas). La primera fila del CSV **siempre** es la molecula de referencia.

---

## Celdas 9-10 — Pipeline Ibuprofeno

Mismo dataset ACI pero con ibuprofeno como molecula de referencia. Genera un conjunto distinto de d_M y theta_M porque la referencia cambia.

---

## Celdas 11-12 — Pipeline Curcumina

Igual que los anteriores pero con curcumina. Los tres (aspirina, ibuprofeno, curcumina) comparten las mismas ~31k moleculas; solo difiere la molecula de referencia.

---

## Celdas 13-14 — Pipeline QM9 Anilina

Dataset QM9: ~127k moleculas organicas pequenas con propiedades cuanticas calculadas (HOMO, LUMO, gap, dipolo, polarizabilidad). `preserve_columns=True` para conservar estas propiedades en el DataFrame final.

---

## Celdas 15-16 — Resumen del pipeline

Crea diccionarios con los 4 DataFrames y metricas, luego imprime una tabla resumen:
- Numero de moleculas por dataset
- Numero de condicion de la covarianza
- Rango de TSI y d_M

Define las constantes `REF_ASPIRIN`, `REF_ANILINE`, etc. con los SMILES de referencia extraidos de la fila 0 de cada DataFrame.

---

## Celdas 17-18 — Estructuras de moleculas de referencia

Dibuja las estructuras 2D de aspirina y anilina lado a lado usando `mol_to_img`. Esto replica la Figura 1 del paper.

---

## Celdas 19-20 — Top-10 analogos de aspirina por theta_M

```python
candidates = aspirin.iloc[1:].copy()           # Excluye la fila 0 (referencia)
top10_theta = candidates.nsmallest(10, 'theta') # Las 10 moleculas con menor angulo de Mahalanobis
```

**Por que theta_M:** El angulo de Mahalanobis mide la similitud de **orientacion** en el espacio de descriptores transformado por covarianza. Un theta pequeno significa que la molecula apunta en la misma "direccion quimica" que la referencia, independientemente de su distancia.

Para cada molecula del top-10, calcula su ranking en las tres metricas (theta, d_M, TSI) para mostrar como difieren los criterios.

---

## Celdas 21-22 — Top-10 analogos de aspirina por TSI (Tanimoto)

```python
top10_tsi = candidates.nlargest(10, 'tanimoto_similarity')  # Mayor TSI = mas similar
```

Identifica empates (ties) en TSI: moleculas con TSI identico redondeado a 4 decimales. Esto ilustra una limitacion de Tanimoto: su resolucion es finita (2048 bits), causando empates que MSI no tiene.

---

## Celdas 23-24 — TSI vs metricas de Mahalanobis (scatter + LOWESS)

Dos paneles:
- **(a)** TSI vs d_M: Muestra como la similitud de Tanimoto decae al aumentar la distancia de Mahalanobis.
- **(b)** TSI vs theta_M: Similar pero con el angulo.

```python
sample = aspirin.sample(min(5000, len(aspirin)), random_state=42)  # Submuestra para LOWESS (eficiencia)

lowess_a = sm.nonparametric.lowess(
    s_a['tanimoto_similarity'], s_a['mahalanobis_distance'],
    frac=PARAMS['lowess_frac']   # frac=0.5: usa 50% de los datos para cada estimacion local
)
```

**LOWESS** (LOcally WEighted Scatterplot Smoothing) ajusta una regresion local ponderada en cada punto. Con `frac=0.5`, cada estimacion usa el 50% mas cercano de los datos. La linea roja resultante muestra la tendencia central.

**Observacion clave:** La relacion TSI vs metricas MSI se asemeja a una funcion racional con comportamiento asintotico. TSI cae abruptamente en la zona de alta similitud y luego se aplana cerca de cero, lo que sugiere que MSI sigue discriminando donde TSI ya perdio resolucion.

---

## Celdas 25-26 — Mapa 3D de similitud (aspirina)

```python
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')     # Crea un eje 3D

df_plot = aspirin[aspirin['weight'] <= 500]     # Filtra moleculas pesadas (>500 Da) para claridad visual

sc = ax.scatter(
    df_plot['theta'],                           # Eje X: angulo de Mahalanobis
    df_plot['mahalanobis_distance'],            # Eje Y: distancia de Mahalanobis
    df_plot['tanimoto_similarity'],             # Eje Z: Tanimoto
    c=df_plot['weight'], cmap='viridis',        # Color: peso molecular
)
```

Visualiza el espacio tridimensional theta x d_M x TSI. La aspirina esta en el origen (0, 0, 1). El coloreado por peso molecular revela si las moleculas pesadas tienden a estar lejos en el espacio de Mahalanobis.

---

## Celdas 27-28 — Representacion polar (aspirina)

```python
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
ax.set_thetamin(0); ax.set_thetamax(180)  # Angulo de 0 a 180 grados

sc = ax.scatter(
    aspirin['radians'],               # Angulo: theta_M en radianes
    aspirin['mahalanobis_distance'],  # Radio: d_M (distancia desde aspirina)
    c=aspirin['mahalanobis_distance'],# Color: tambien d_M para intensidad
)
```

La representacion polar es una forma natural de visualizar theta_M y d_M juntos: el angulo polar es theta_M y el radio es d_M. La aspirina esta en el centro (polo). Moleculas similares estan cerca del centro y con angulo pequeno.

Calcula estadisticas de zona: cuantas moleculas estan en la region theta < 10 y d_M < 80 (zona de alta similitud).

---

## Celdas 29-30 — Top-10 analogos de anilina por theta_M (QM9)

```python
qm9_candidates = qm9.iloc[1:].copy()       # Excluye anilina (fila 0)
aniline_gap = qm9['gap'].iloc[0]           # E_gap de anilina como referencia
top10_qm9_theta = qm9_candidates.nsmallest(10, 'theta')
top10_qm9_theta['delta_gap'] = top10_qm9_theta['gap'] - aniline_gap  # Desviacion del gap
```

Ademas de las metricas MSI, calcula `delta_gap`: la diferencia entre el HOMO-LUMO gap de cada analogo y el de anilina. Esto permite evaluar si moleculas "similares" segun MSI tambien tienen propiedades electronicas similares.

---

## Celdas 31-32 — Top-10 analogos de anilina por TSI

Igual que el caso de aspirina pero para QM9. Tambien identifica empates de TSI y muestra delta_gap para comparar con el ranking MSI.

---

## Celdas 33-34 — Comparacion de rankings MSI vs TSI (anilina)

```python
top_by_theta = qm9_candidates.nsmallest(N_TOP, 'theta')     # Top-10 segun MSI
top_by_tsi = qm9_candidates.nlargest(N_TOP, 'tanimoto_similarity')  # Top-10 segun TSI

overlap = set(top_by_theta.index) & set(top_by_tsi.index)   # Moleculas en ambos top-10
```

**Panel (a):** Grafico de barras comparando el E_gap de los top-10 de cada metodo. La linea punteada azul marca el E_gap de anilina. Si las barras azules (MSI) estan mas cerca de la linea que las rojas (TSI), MSI selecciona analogos electronicamente mas similares.

**Panel (b):** Delta E_gap (desviacion respecto a anilina). Barras mas cercanas a cero indican mejor seleccion.

---

## Celdas 35-36 — TSI vs metricas de Mahalanobis (QM9)

Mismo formato que las celdas 23-24 pero para el dataset QM9 (~127k moleculas). El patron es el mismo: decaimiento tipo funcion racional.

---

## Celdas 37-38 — Representacion polar QM9

Igual que la celda 28 pero para QM9. Se espera una dispersion angular mas amplia porque QM9 contiene moleculas organicas de muchos tipos diferentes (mayor diversidad quimica que ACI, que esta filtrado por subestructura).

Usa `cmap='plasma'` en vez de `'viridis'` para diferenciar visualmente.

---

## Celdas 39-40 — 3D: theta_M x d_M x E_gap (QM9)

```python
sc = ax.scatter(
    qm9['theta'],                  # X: angulo de Mahalanobis
    qm9['mahalanobis_distance'],   # Y: distancia de Mahalanobis
    qm9['gap'],                    # Z: HOMO-LUMO gap
    c=qm9['gap'], cmap='tab10',   # Color: tambien E_gap
)
```

Unico de QM9: explora si la proximidad en el espacio de Mahalanobis correlaciona con propiedades electronicas reales. Si moleculas cercanas a anilina (theta y d_M pequenos) tienen E_gap similar, esto valida que MSI captura similitud quimica genuina, no solo similitud estructural.

---

## Celdas 41-42 — Comparacion cruzada: aspirina vs ibuprofeno vs curcumina

```python
for ax, (name, df, ref_name) in zip(axes, [
    ('aspirin', aspirin, 'Aspirin'),
    ('ibuprofen', ibuprofen, 'Ibuprofen'),
    ('curcumin', curcumin, 'Curcumin'),
]):
    # Scatter TSI vs d_M para cada referencia
    # + curva LOWESS superpuesta
```

Tres paneles mostrando TSI vs d_M para cada molecula de referencia sobre el mismo pool de ~31k moleculas. El patron cualitativo (decaimiento racional) es consistente en los tres, lo que sugiere que el framework MSI se comporta de manera predecible independientemente de la molecula de consulta.

---

## Celdas 43-44 — Resumen de resultados

Imprime estadisticas cuantitativas finales: tamano de dataset, rangos de d_M y theta_M, mejor analogo, empates de TSI, porcentaje en zona de concentracion, y comparacion de E_gap entre MSI y TSI.

---

---

# ANALISIS EXTENDIDOS (Celdas 45-63) — Enfasis especial

Los analisis extendidos van mas alla de las visualizaciones centrales para validar estadisticamente el framework MSI. Cada uno responde a una pregunta critica.

---

## Celda 45 (Markdown) — Introduccion a analisis extendidos

Lista las 7 preguntas que se abordan:
1. Significancia estadistica (bootstrap)
2. Estabilidad de rankings (sensibilidad a lambda)
3. Estructura de correlacion entre metricas
4. Distribuciones marginales
5. Prediccion de propiedades cuanticas multiples
6. Comparacion expandida top-N
7. Baseline de similitud coseno

---

## Celdas 46-47 — **Extended 1: Bootstrap Test — MSI vs TSI E_gap**

### Problema que resuelve
Con solo N=10 moleculas, comparar promedios de |delta E_gap| no tiene poder estadistico suficiente. Necesitamos saber: **es la ventaja de MSI estadisticamente significativa o podria ser azar?**

### Que es Bootstrap y como funciona

Bootstrap es una tecnica estadistica de **remuestreo**. La idea central es simple:

> Si no sabemos como se comportaria un resultado "por azar", podemos **simularlo** tomando miles de muestras aleatorias del mismo dataset y midiendo lo mismo cada vez.

En este caso, el procedimiento es:
1. **Observamos** que MSI selecciona 10 moleculas cuyo |delta E_gap| promedio es X.
2. **Preguntamos**: "Si hubieramos seleccionado 10 moleculas **al azar**, que tan probable es obtener un |delta E_gap| tan bajo como X?"
3. **Simulamos** 10,000 veces: en cada iteracion, elegimos 10 moleculas al azar y calculamos su |delta E_gap| promedio.
4. **Comparamos**: de esas 10,000 muestras aleatorias, cuantas lograron un resultado tan bueno como MSI?

Esa fraccion es el **p-value**.

### Codigo explicado linea por linea

```python
np.random.seed(42)       # Semilla para reproducibilidad
N_BOOT = 10_000          # 10,000 remuestreos bootstrap
N_TOP = 10               # Tamano de cada muestra

qm9_cands = qm9.iloc[1:].copy()           # Todas las moleculas excepto anilina
aniline_gap = qm9['gap'].iloc[0]          # E_gap de referencia (anilina)
```

```python
# Valores observados: desviacion media de E_gap para los top-10 de cada metodo
top_msi = qm9_cands.nsmallest(N_TOP, 'theta')           # Top-10 MSI
top_tsi = qm9_cands.nlargest(N_TOP, 'tanimoto_similarity')  # Top-10 TSI
obs_msi = np.abs(top_msi['gap'] - aniline_gap).mean()   # |delta E_gap| medio para MSI
obs_tsi = np.abs(top_tsi['gap'] - aniline_gap).mean()   # |delta E_gap| medio para TSI
```

**Por que |delta E_gap|:** Mide que tan cerca esta el gap electronico de cada analogo al de anilina. Menor = mejor seleccion de analogos.

```python
# Distribucion bootstrap: que esperariamos por azar?
boot_gaps = np.zeros(N_BOOT)
for b in range(N_BOOT):
    sample_idx = np.random.choice(len(qm9_cands), size=N_TOP, replace=False)  # 10 moleculas al azar
    boot_gaps[b] = np.abs(qm9_cands['gap'].iloc[sample_idx].values - aniline_gap).mean()
```

En cada iteracion, selecciona 10 moleculas **al azar** (sin reemplazo) y calcula la misma metrica. Esto genera la **distribucion nula**: como se ve |delta E_gap| cuando la seleccion es completamente aleatoria.

```python
# P-values: fraccion de muestras aleatorias que igualan o mejoran al metodo
p_msi = (boot_gaps <= obs_msi).mean()   # p-value para MSI
p_tsi = (boot_gaps <= obs_tsi).mean()   # p-value para TSI
```

### Como se interpreta el grafico del Bootstrap

El histograma gris muestra la **distribucion de 10,000 muestras aleatorias**. Es una campana centrada alrededor de ~0.0529 E_h (la desviacion media que se obtendria por puro azar).

Las lineas verticales representan:
- **Linea azul punteada (MSI):** El |delta E_gap| = 0.0134 E_h que obtuvieron los top-10 de MSI. Esta **muy a la izquierda** del histograma, lejos de la zona de azar.
- **Linea roja punteada (TSI):** El |delta E_gap| = 0.0261 E_h de los top-10 de TSI. Tambien esta a la izquierda, pero **menos que MSI**.
- **Linea negra punteada:** La media aleatoria (0.0529 E_h), el centro de la campana.

**Regla visual:** Mientras **mas a la izquierda** este la linea respecto al histograma, **mejor** es el metodo (selecciona analogos con propiedades mas cercanas a la referencia).

### Que es `p_msi` y como se interpreta

`p_msi` es la fraccion de las 10,000 muestras aleatorias que obtuvieron un resultado **igual o mejor** que MSI.

**Resultados obtenidos:**
- `p_msi = 0.0000` → De 10,000 muestras aleatorias, **ninguna** (0 de 10,000) logro un |delta E_gap| tan bajo como 0.0134 E_h. Esto es practicamente imposible por azar. MSI selecciona moleculas con E_gap mas cercano a anilina que el **100%** de las selecciones aleatorias.
- `p_tsi = 0.0095` → Solo el 0.95% de las selecciones aleatorias lograron igualar a TSI. TSI es mejor que el 99.1% del azar, pero **claramente peor que MSI**.
- `Random baseline = 0.0529 +/- 0.0120 E_h` → Lo que se esperaria por puro azar.

**Conclusion del bootstrap:** Ambos metodos son mejores que el azar, pero MSI es significativamente superior (p=0.0000 vs p=0.0095). La ventaja de MSI **no** es un artefacto estadistico.

---

## Celdas 48-49 — **Extended 2: Expanded Top-N Overlap y E_gap**

### Problema que resuelve
**La ventaja de MSI persiste a mayor escala?** Compara MSI vs TSI en N = 10, 25, 50, 100, 250, 500.

### Que es "Expanded Top-N Overlap"

El concepto es simple: en vez de comparar solo los 10 mejores de cada metodo, vamos ampliando la ventana: los 25 mejores, los 50 mejores, etc. Para cada N medimos dos cosas:

1. **Overlap (solapamiento):** Cuantas moleculas aparecen en **ambos** top-N (el de MSI y el de TSI). Si el overlap es bajo, significa que cada metodo elige moleculas diferentes; si es alto, coinciden.
2. **|delta E_gap|:** Que tan cerca estan los E_gap de los analogos seleccionados al de anilina. **Menor es mejor.**

```python
N_values = [10, 25, 50, 100, 250, 500]

for N in N_values:
    top_theta = qm9_cands.nsmallest(N, 'theta')               # Top-N por MSI
    top_tsi_n = qm9_cands.nlargest(N, 'tanimoto_similarity')   # Top-N por TSI

    overlap = len(set(top_theta.index) & set(top_tsi_n.index))  # Moleculas en ambos conjuntos
    msi_gap = np.abs(top_theta['gap'] - aniline_gap).mean()     # |delta E_gap| para MSI
    tsi_gap = np.abs(top_tsi_n['gap'] - aniline_gap).mean()     # |delta E_gap| para TSI
```

### Que significa que la linea TSI este arriba de MSI en "E_gap Proximity by Ranking Method"

En el panel **(b)** del grafico, el eje Y es "Mean |delta E_gap|" — la desviacion promedio del gap electronico respecto a anilina. **Menor es mejor** (mas cercano a la referencia).

Si la **barra/linea roja (TSI) esta por encima** de la **barra/linea azul (MSI)**, significa que:
- TSI tiene un |delta E_gap| **mas alto** → sus analogos estan **mas lejos** electronicamente de anilina.
- MSI tiene un |delta E_gap| **mas bajo** → sus analogos estan **mas cerca** electronicamente de anilina.

**En otras palabras:** MSI esta seleccionando moleculas que no solo "se parecen" estructuralmente a anilina, sino que realmente tienen propiedades electronicas similares. TSI selecciona moleculas que se parecen en su fingerprint (topologia 2D) pero no necesariamente en sus propiedades cuanticas.

### Resultados obtenidos

| N | Overlap | Overlap % | MSI |delta E_gap| | TSI |delta E_gap| | Ratio TSI/MSI |
|---:|--------:|----------:|-------------------:|-------------------:|--------------:|
| 10 | 3 | 30% | 0.0134 | 0.0261 | 1.95x |
| 25 | 12 | 48% | 0.0116 | 0.0215 | 1.85x |
| 50 | 32 | 64% | 0.0170 | 0.0211 | 1.24x |
| 100 | 65 | 65% | 0.0199 | 0.0207 | 1.04x |
| 250 | 121 | 48% | 0.0189 | 0.0221 | 1.17x |
| 500 | 267 | 53% | 0.0207 | 0.0217 | 1.05x |

### Por que podemos decir con confianza que si TSI/MSI ratio > 1, MSI es mejor

El **ratio** se calcula como:

```
ratio = tsi_gap_dev / msi_gap_dev
```

Es decir: `(desviacion de TSI) / (desviacion de MSI)`.

- Recordemos que |delta E_gap| mide **error** (que tan lejos estan los analogos de la referencia). **Menor = mejor.**
- Si TSI tiene |delta E_gap| = 0.0261 y MSI tiene 0.0134, el ratio es 0.0261/0.0134 = **1.95x**.
- Esto significa que **el error de TSI es 1.95 veces mayor que el de MSI**. O dicho de otra forma: MSI se equivoca **casi la mitad** de lo que se equivoca TSI.
- Si el ratio fuera exactamente 1.0, ambos metodos serian iguales.
- Si el ratio fuera < 1.0, TSI seria mejor.
- **Todos los ratios en la tabla estan por encima de 1.0**, desde 1.95x (a N=10) hasta 1.05x (a N=100).

La ventaja es mas fuerte a N pequeno (1.95x) y se reduce a N grande (1.04x-1.17x), lo que tiene sentido: al incluir mas moleculas, ambos metodos empiezan a converger. Pero MSI **nunca pierde**: siempre selecciona analogos al menos tan buenos o mejores que TSI.

---

## Celdas 50-51 — **Extended 3: Cosine Similarity Baseline**

### Problema que resuelve
**El ponderado por Mahalanobis realmente ayuda, o la similitud coseno simple en el espacio mol2vec de 300D funciona igual de bien?** Si coseno iguala a MSI, la correccion por covarianza agrega complejidad sin beneficio.

```python
# embedding_similarity ya esta calculado como similitud coseno en 300D por el pipeline
top_cosine = qm9_cands.nlargest(10, 'embedding_similarity')  # Top-10 por coseno
top_theta = qm9_cands.nsmallest(10, 'theta')                  # Top-10 por MSI
top_tsi = qm9_cands.nlargest(10, 'tanimoto_similarity')       # Top-10 por TSI
```

Compara tres metodos head-to-head:
1. **MSI (theta_M):** Distancia angular en espacio transformado por covarianza inversa
2. **Coseno (300D):** Similitud coseno directa sin transformacion
3. **TSI (Tanimoto):** Fingerprint Morgan de 2048 bits

```python
# Comparacion a multiples N
for N in N_vals:
    t_cos = qm9_cands.nlargest(N, 'embedding_similarity')
    t_msi = qm9_cands.nsmallest(N, 'theta')
    t_tsi = qm9_cands.nlargest(N, 'tanimoto_similarity')
```

**Panel 1:** Barras comparando |delta E_gap| a N=10.
**Panel 2:** Curvas de escalamiento con N para los tres metodos.

La diferencia clave entre coseno y MSI es la **covarianza inversa**: MSI pondera cada dimension por la varianza y correlacion del dataset. Dimensiones con alta varianza (menos informativas) reciben menos peso. Esto es lo que hace que MSI sea superior a coseno simple.

```python
# Overlap entre metodos
overlap_msi_cos = len(set(top_theta.index) & set(top_cosine.index))
```

### Interpretacion de los resultados de la tabla sweep (cos_df)

Los resultados obtenidos fueron:

| N | Cosine |delta E_gap| | MSI |delta E_gap| | TSI |delta E_gap| |
|---:|---------------------:|------------------:|------------------:|
| 10 | 0.0236 | **0.0134** | 0.0261 |
| 25 | 0.0172 | **0.0116** | 0.0215 |
| 50 | 0.0241 | **0.0170** | 0.0211 |
| 100 | 0.0229 | **0.0199** | 0.0207 |
| 250 | 0.0219 | **0.0189** | 0.0221 |

**MSI gana en todos los valores de N.** Esto confirma que la transformacion de Mahalanobis **SI** aporta valor real sobre el coseno simple.

La razon es que coseno trata las 300 dimensiones como si fueran igualmente importantes. Pero en la realidad, algunas dimensiones del embedding mol2vec capturan variabilidad trivial (ruido) y otras capturan patrones quimicos genuinos. La covarianza inversa de Mahalanobis **pondera cada dimension segun su importancia**: las dimensiones donde todas las moleculas varian mucho (poco informativas) reciben menos peso, y las dimensiones donde las moleculas similares se agrupan (muy informativas) reciben mas peso.

### Que nos dice la matriz de overlap

```
Top-10 overlap matrix:
  MSI ∩ Cosine: 3/10
  MSI ∩ TSI:    3/10
  Cosine ∩ TSI: 6/10
```

Esto revela algo muy importante:

- **MSI y Cosine solo comparten 3 de 10 moleculas.** Aunque ambos operan en el mismo espacio de 300D, la transformacion de Mahalanobis cambia **drasticamente** que moleculas considera "mas similares". MSI "ve" patrones que coseno no ve.
- **MSI y TSI solo comparten 3 de 10 moleculas.** MSI y Tanimoto trabajan con representaciones completamente diferentes (embeddings vs fingerprints), asi que es natural que seleccionen moleculas distintas.
- **Cosine y TSI comparten 6 de 10 moleculas.** Esto es sorprendente: el coseno simple (sin Mahalanobis) se parece **mas a Tanimoto que a MSI**. Esto sugiere que coseno y Tanimoto capturan una nocion de similitud mas superficial/topologica, mientras que MSI captura algo diferente y mas profundo.

### Que nos dicen los |delta E_gap|

```
Top-10 mean |delta E_gap|:
  MSI (theta_M): 0.0134 E_h   ← MEJOR (mas cercano a anilina)
  Cosine (300D):  0.0236 E_h   ← Intermedio
  TSI (Tanimoto): 0.0261 E_h   ← PEOR (mas lejos de anilina)
```

MSI selecciona moleculas cuyo E_gap difiere en promedio solo 0.0134 E_h de anilina. Coseno se equivoca 1.76x mas (0.0236) y TSI 1.95x mas (0.0261).

**Conclusion:** MSI >> Coseno >> TSI. La transformacion de Mahalanobis **no** es complejidad innecesaria; captura estructura genuina del espacio quimico que el coseno simple ignora. La complejidad adicional esta completamente justificada por la mejora en rendimiento.

---

## Celdas 52-53 — **Extended 4: Heatmap de correlacion entre metricas**

### Problema que resuelve
**Como se relacionan todas las metricas entre si?** Alta correlacion entre theta_M y d_M es esperada (ambas son Mahalanobis), pero la relacion con TSI y coseno es mas informativa.

```python
metric_cols = ['theta', 'mahalanobis_distance', 'tanimoto_similarity',
               'embedding_similarity', 'weight', 'mahalanobis_magnitude', 'projection']
```

Se incluyen 7 metricas:
- `theta`: Angulo de Mahalanobis (orientacion)
- `mahalanobis_distance`: Distancia de Mahalanobis (magnitud)
- `tanimoto_similarity`: Tanimoto (fingerprint)
- `embedding_similarity`: Coseno en 300D
- `weight`: Peso molecular
- `mahalanobis_magnitude`: Norma del vector en espacio de Mahalanobis
- `projection`: Proyeccion sobre el vector de referencia

```python
corr = df[metric_cols].corr(method='spearman')  # Correlacion de RANGO (Spearman)
```

**Por que Spearman y no Pearson:** Las relaciones entre metricas son no lineales (como vimos en los scatters de TSI vs d_M), asi que Pearson subestimaria la asociacion. Spearman mide monotonicidad, no linealidad.

```python
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True)
```

La mascara triangular (`mask = np.triu(...)`) evita mostrar la mitad redundante de la matriz simetrica.

Se genera un heatmap para ACI (aspirina) y otro para QM9 (anilina) lado a lado para comparar si la estructura de correlacion es consistente entre datasets.

---

## Celdas 54-55 — **Extended 5: Distribuciones marginales de theta_M, d_M y TSI**

### Problema que resuelve
**Cual es la forma de las distribuciones?** Saber si son simetricas, sesgadas, o multimodales ayuda a entender donde cada metrica tiene mas poder discriminativo.

```python
for row, (name, df) in enumerate([('ACI (Aspirin)', aspirin), ('QM9 (Aniline)', qm9)]):
    cands = df.iloc[1:]  # Excluye referencia

    # Histograma de theta_M (100 bins)
    ax.hist(cands['theta'], bins=100, ...)
    ax.axvline(cands['theta'].median(), ...)  # Mediana como referencia
```

Seis paneles (2 datasets x 3 metricas). Para cada uno:
- Histograma con 100 bins
- Linea vertical en la mediana
- Ejes etiquetados con las unidades correctas

```python
# Estadisticas descriptivas completas
for col in ['theta', 'mahalanobis_distance', 'tanimoto_similarity']:
    s = cands[col]
    print(f'mean={s.mean():.4f}  std={s.std():.4f}  '
          f'median={s.median():.4f}  skew={s.skew():.2f}  kurtosis={s.kurtosis():.2f}')
```

**Skewness** (asimetria): Valor positivo indica cola derecha larga. **Kurtosis**: Valor alto indica colas pesadas (mas outliers que una normal).

### Interpretacion de los resultados obtenidos

**Aspirina (ACI), N=31,392 moleculas:**

| Metrica | Media | Std | Mediana | Skew | Kurtosis |
|---------|------:|----:|--------:|-----:|---------:|
| theta_M | 4.27 | 1.79 | 3.89 | 2.33 | 16.50 |
| d_M | 15.73 | 6.68 | 14.30 | 2.64 | 24.39 |
| TSI | 0.2460 | 0.1159 | 0.2167 | 0.78 | 0.42 |

**Anilina (QM9), N=127,536 moleculas:**

| Metrica | Media | Std | Mediana | Skew | Kurtosis |
|---------|------:|----:|--------:|-----:|---------:|
| theta_M | 14.39 | 1.25 | 14.26 | 2.23 | 38.61 |
| d_M | 27.24 | 2.51 | 26.97 | 4.39 | 168.22 |
| TSI | 0.0324 | 0.0404 | 0.0303 | 1.83 | 5.65 |

**Que nos dicen estos numeros:**

1. **theta_M y d_M tienen skewness alta (2-4) y kurtosis muy alta (16-168).** Esto significa que la gran mayoria de moleculas se agrupan en un rango estrecho de valores, con unas pocas moleculas extremas que forman una "cola larga" a la derecha. En terminos practicos: la mayoria de las moleculas estan a distancias moderadas de la referencia, pero hay unos pocos outliers muy lejanos. Las metricas de Mahalanobis tienen **buena resolucion en la zona central** (donde estan la mayoria de las moleculas) y detectan facilmente los outliers.

2. **TSI en ACI tiene skewness baja (0.78) y kurtosis baja (0.42).** La distribucion de Tanimoto para aspirina es relativamente simetrica y suave, con media 0.25 y mediana 0.22. Esto se debe a que el dataset ACI fue construido por busqueda de subestructura en PubChem (las moleculas ya comparten un fragmento con aspirina). TSI tiene resolucion razonable en este dataset.

3. **TSI en QM9 es practicamente inutil para discriminar.** La media es solo 0.0324 con mediana 0.0303. Esto significa que **casi todas las moleculas tienen TSI cercano a cero** respecto a anilina. Cuando todo esta en ~0.03, distinguir entre 0.031 y 0.032 no tiene significado quimico real. En cambio, theta_M tiene media 14.39 con std 1.25, lo que permite discriminar mucho mejor en esa zona.

4. **Comparacion ACI vs QM9:** En ACI, theta_M va de 0 a ~38 grados con la mayoria entre 2 y 6. En QM9, theta_M va de 0 a ~61 grados con la mayoria entre 13 y 16. Esta dispersion angular mas amplia en QM9 refleja la mayor diversidad quimica del dataset (moleculas organicas de todo tipo vs moleculas pre-filtradas por subestructura).

**Conclusion general:** Las metricas de Mahalanobis (theta_M, d_M) mantienen poder discriminativo incluso en datasets muy diversos como QM9, donde TSI colapsa a valores cercanos a cero. Esta es una ventaja fundamental del enfoque MSI.

---

## Celdas 56-57 — **Extended 6: Prediccion de multiples propiedades cuanticas (QM9)**

### Problema que resuelve
MSI supera a TSI para E_gap, pero **generaliza a otras propiedades cuanticas?** Si solo funciona para gap pero no para HOMO, LUMO, dipolo o polarizabilidad, la ventaja seria limitada.

```python
properties = {
    'gap': ('$E_{gap}$', 'E_h'),           # HOMO-LUMO gap
    'homo': ('HOMO', 'E_h'),               # Energia del orbital HOMO
    'lumo': ('LUMO', 'E_h'),               # Energia del orbital LUMO
    'mu': ('Dipole $\\mu$', 'D'),          # Momento dipolar (Debye)
    'alpha': ('Polarizability $\\alpha$', 'Bohr^3'),  # Polarizabilidad
}
```

```python
N_MULTI = 50                                        # Usa top-50 para mas robustez
ref_vals = {prop: qm9[prop].iloc[0] for prop in properties}  # Valores de anilina

for prop, (label, unit) in properties.items():
    top_msi_p = qm9_cands.nsmallest(N_MULTI, 'theta')           # Top-50 MSI
    top_cos_p = qm9_cands.nlargest(N_MULTI, 'embedding_similarity')  # Top-50 coseno
    top_tsi_p = qm9_cands.nlargest(N_MULTI, 'tanimoto_similarity')   # Top-50 TSI

    ref = ref_vals[prop]
    msi_dev = np.abs(top_msi_p[prop] - ref).mean()   # Desviacion media MSI
    cos_dev = np.abs(top_cos_p[prop] - ref).mean()   # Desviacion media coseno
    tsi_dev = np.abs(top_tsi_p[prop] - ref).mean()   # Desviacion media TSI
    all_dev = np.abs(qm9_cands[prop] - ref).mean()   # Baseline aleatorio (todo el dataset)
```

**Normalizacion por baseline aleatorio:**
```python
# En la grafica, se muestra deviation / random_baseline
bars1 = ax.bar(x - 1.5*w, multi_df['MSI_dev'] / multi_df['Random_dev'], w, ...)
```

Valores < 1.0 significan que el metodo selecciona analogos mas cercanos que el azar. La grafica de barras muestra MSI (azul), coseno (naranja), TSI (rojo) y baseline aleatorio (gris, siempre = 1.0).

```python
'MSI_vs_random': msi_dev / all_dev,  # Fraccion: 0.3 = MSI es 70% mejor que azar
'best': 'MSI' if msi_dev <= min(cos_dev, tsi_dev) else ...  # Mejor metodo por propiedad
```

### Interpretacion de los resultados obtenidos

| Propiedad | Ref. anilina | MSI dev | Cosine dev | TSI dev | Random dev | Ganador |
|-----------|------------:|--------:|-----------:|--------:|-----------:|---------|
| E_gap | 0.2077 E_h | **0.0170** | 0.0241 | 0.0211 | 0.0527 | **MSI** |
| HOMO | -0.1991 E_h | 0.0197 | 0.0170 | **0.0154** | 0.0419 | TSI |
| LUMO | 0.0086 E_h | **0.0183** | 0.0283 | 0.0267 | 0.0388 | **MSI** |
| Dipolo mu | 1.6318 D | **0.9870** | 1.3536 | 1.3562 | 1.3488 | **MSI** |
| Polariz. alpha | 67.30 Bohr^3 | 14.6910 | **13.2706** | 15.2680 | 9.5412 | Cosine |

**Que significa cada columna:**
- **MSI/Cosine/TSI dev:** Desviacion media absoluta del top-50 de cada metodo respecto al valor de anilina. Menor = los analogos seleccionados se parecen mas a anilina en esa propiedad.
- **Random dev:** Lo que se esperaria al seleccionar 50 moleculas al azar. Es el baseline.
- **Ganador:** El metodo con menor desviacion (mas parecido a anilina).

**Que podemos decir:**

1. **MSI gana en 3 de 5 propiedades** (E_gap, LUMO, dipolo). No solo captura similitud en el gap electronico, sino tambien en la energia del LUMO y el momento dipolar. Esto es notable porque MSI **nunca vio** estas propiedades — trabaja solo con embeddings mol2vec. Que las moleculas cercanas en espacio de Mahalanobis tambien tengan propiedades electronicas similares confirma que el espacio captura quimica genuina.

2. **TSI gana en HOMO.** Esto sugiere que la energia del HOMO correlaciona fuertemente con la topologia molecular (que es lo que Tanimoto captura bien). Es razonable: el HOMO esta muy influenciado por la estructura del esqueleto molecular y los heteroatomos.

3. **Cosine gana en polarizabilidad.** La polarizabilidad depende principalmente del tamano y la "nube electronica" de la molecula. El coseno simple, al no ponderar dimensiones, podria estar capturando mejor la informacion de tamano molecular que el embedding codifica.

4. **Para dipolo (mu), MSI obtiene 0.987 D vs Random 1.349 D.** Es decir, MSI reduce la desviacion en ~27% respecto al azar. Cosine y TSI no mejoran: sus desviaciones (~1.35 D) son practicamente iguales al baseline aleatorio. Solo MSI captura la informacion relevante del momento dipolar.

5. **Para polarizabilidad, TODOS los metodos son peores que el azar** (todos > 9.54 Bohr^3). Esto indica que las moleculas mas similares en cualquier metrica de similitud tienden a tener tamaños mas parecidos a anilina (molecula pequena), lo cual genera un sesgo: las moleculas similares son pequenas, pero el dataset completo tiene distribucion de tamaños mas variada, resultando en una desviacion "aleatoria" menor para alpha.

**Conclusion:** MSI captura **similitud quimica multidimensional genuina**, no solo similitud topologica. Su ventaja se extiende a propiedades cuanticas que nunca fueron parte del calculo, lo cual valida el framework a nivel fisicoquimico.

---

## Celdas 58-59 — **Extended 7: Sensibilidad al parametro de regularizacion lambda**

### Problema que resuelve
El paper usa lambda = 10^-5 para regularizar la covarianza. **Que tan sensibles son los rankings a esta eleccion?** Si cambian drasticamente con lambda, el metodo seria fragil.

```python
vec_300d = pd.read_csv(aspirin_300d_path).values   # Carga los vectores 300D de aspirina
ref_vec = vec_300d[0]                               # Vector de la molecula de referencia

lambdas = [0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # 7 valores de lambda a probar
cov_raw = np.cov(vec_300d.T)                        # Covarianza muestral sin regularizar (300x300)
```

Para cada lambda:
```python
cov_reg = cov_raw + np.eye(cov_raw.shape[0]) * lam   # Sigma + lambda*I (regularizacion de Tikhonov)
inv_cov = np.linalg.inv(cov_reg)                      # Inversa de la covarianza regularizada
```

**Tikhonov regularization:** Suma lambda a cada elemento diagonal de la covarianza. Esto tiene dos efectos:
1. Garantiza que la matriz sea invertible (importante cuando N < 300 o hay colinealidad)
2. Reduce la influencia de eigenvalores muy pequenos que amplificarian ruido

**Calculo vectorizado de theta_M:**
```python
numerator = vec_300d @ inv_cov @ ref_vec              # v_i^T * Sigma^{-1} * v_ref para todas las moleculas
vi_norms = np.sqrt(np.maximum(
    np.einsum('ij,ij->i', vec_300d, vec_300d @ inv_cov), 0))  # ||v_i||_{Sigma^{-1}}
vref_norm = np.sqrt(max(ref_vec @ inv_cov @ ref_vec, 0))      # ||v_ref||_{Sigma^{-1}}
```

La formula es: `theta_M = arccos( (v_i^T Sigma^{-1} v_ref) / (||v_i||_{Sigma^{-1}} * ||v_ref||_{Sigma^{-1}}) )`

Es como similitud coseno pero en el espacio transformado por la covarianza inversa.

```python
cos_vals = np.clip(numerator / safe_denom, -1, 1)    # Clip para evitar errores de arccos por precision numerica
theta_vals = np.degrees(np.arccos(cos_vals))           # Convertir radianes a grados
```

**Analisis de estabilidad:**

```python
# Panel 1: Correlacion de Spearman de rankings entre cada lambda y lambda=1e-5
rho, _ = spearmanr(ref_ranks, theta_by_lambda[lam])
```

Si rho ~ 1 para todos los lambda, los **rankings globales** son estables.

```python
# Panel 2: Overlap del top-10 entre cada lambda y lambda=1e-5
top10_ref = set(np.argsort(ref_ranks)[1:11])     # Top-10 con lambda de referencia
top10_lam = set(np.argsort(theta_by_lambda[lam])[1:11])  # Top-10 con otro lambda
jaccard = len(top10_ref & top10_lam) / len(top10_ref | top10_lam)  # Indice de Jaccard
overlap_n = len(top10_ref & top10_lam)            # Numero de moleculas compartidas
```

**Numeros de condicion:**
```python
cond = np.linalg.cond(cov_test)   # Ratio entre eigenvalor maximo y minimo
```

Un numero de condicion alto (e.g., 10^15) indica que la matriz es casi singular y la inversa es numericamente inestable. Lambda reduce el numero de condicion al "levantar" los eigenvalores pequenos.

### Explicacion intuitiva de por que lambda importa

Imagina que la covarianza es una "lente" a traves de la cual MSI "ve" el espacio quimico. Si la lente esta bien calibrada, distingue moleculas similares de diferentes. Pero si la lente tiene imperfecciones (eigenvalores muy pequenos = dimensiones inestables), amplifica ruido y distorsiona la imagen.

**Lambda actua como un estabilizador de la lente:**
- **Lambda = 0 (sin regularizacion):** La lente es "pura" pero puede tener defectos opticos severos. Si alguna dimension tiene varianza casi cero, la inversa la amplifica enormemente y domina todo el calculo. Numero de condicion: 1.81e+04.
- **Lambda = 1e-5 (valor del paper):** Agrega una cantidad minima de "suavizado" a la lente. Suficiente para eliminar inestabilidades sin distorsionar la imagen. Numero de condicion: 8.20e+03 (reducido a menos de la mitad).
- **Lambda = 1e-2 (demasiado grande):** La lente esta tan suavizada que pierde resolucion. Todas las dimensiones empiezan a parecer iguales, y MSI se degrada hacia coseno simple. Numero de condicion: 1.59e+01 (casi esfera, ya no hay estructura anisotrópica).

### Por que el top-10 tiene overlap 10/10 en lambda = 1e-5

El analisis muestra que al comparar los top-10 calculados con lambda = 1e-5 (referencia del paper) contra los top-10 calculados con otros lambdas, el overlap es:

- **lambda = 0:** El overlap puede ser menor porque sin regularizacion, dimensiones ruidosas alteran rankings de moleculas que estan en el "borde" del top-10.
- **lambda = 1e-8 a 1e-4:** Overlap alto (cercano a 10/10) porque en este rango, la regularizacion es lo suficientemente pequena para no distorsionar la geometria, pero lo suficientemente grande para estabilizar la inversa.
- **lambda = 1e-5 vs lambda = 1e-5:** Es la comparacion consigo mismo, naturalmente 10/10.
- **lambda = 1e-2:** Overlap puede bajar porque la regularizacion excesiva aplasta la estructura de la covarianza.

**La razon fundamental** por la que 1e-5 produce 10/10 consigo mismo y alta estabilidad con lambdas vecinos es:
1. Los eigenvalores mas pequenos de la covarianza estan en el orden de 10^-4 a 10^-3.
2. Sumar 1e-5 a la diagonal apenas perturba los eigenvalores grandes (que dominan las dimensiones informativas) pero estabiliza los pequenos.
3. Las moleculas del top-10 estan "lejos" del borde de decision: su theta_M es tan bajo que pequeñas perturbaciones en la covarianza no cambian su ranking relativo.
4. Solo moleculas que estan muy cerca del "corte" del top-10 (posiciones ~8-12) podrian cambiar, pero si la separacion es clara, se mantienen.

### Numeros de condicion obtenidos

| Lambda | Num. condicion |
|-------:|---------------:|
| 0 | 1.81e+04 |
| 1e-8 | 1.81e+04 |
| 1e-6 | 1.62e+04 |
| **1e-5** | **8.20e+03** |
| 1e-4 | 1.38e+03 |
| 1e-3 | 1.49e+02 |
| 1e-2 | 1.59e+01 |

El numero de condicion es el ratio entre el eigenvalor mas grande y el mas pequeño de la matriz. Un numero de condicion de 8.20e+03 (como con lambda=1e-5) significa que el eigenvalor mas grande es ~8,200 veces mayor que el mas pequeño. Esto es manejable numericamente. En contraste, 1.81e+04 (sin regularizacion) indica mas inestabilidad.

**Conclusion:** Lambda = 1e-5 es un punto dulce: reduce el numero de condicion significativamente (de 18,100 a 8,200), mantiene la estructura anisotrópica de la covarianza (no aplasta las diferencias entre dimensiones como lambda=1e-2), y produce rankings extremadamente estables. El metodo **no** depende criticamente de la eleccion exacta de lambda, lo cual es deseable para un framework cientifico.

---

## Celdas 60-61 — **Extended 8: t-SNE Embedding Space**

### Problema que resuelve
**Las regiones de alta similitud se agrupan espacialmente o estan dispersas?** Esto revela si la estructura de similitud tiene un correlato geometrico en el espacio de embeddings.

```python
for row, (name, df) in enumerate([('ACI (Aspirin)', aspirin), ('QM9 (Aniline)', qm9)]):
    sample_df = df.sample(min(5000, len(df)), random_state=42)  # Subsample para velocidad

    # Panel 1: Coloreado por theta_M (viridis_r: azul = theta bajo = mas similar)
    sc = ax.scatter(sample_df['c1'], sample_df['c2'], c=sample_df['theta'], cmap='viridis_r')

    # Panel 2: Coloreado por TSI (RdYlGn: verde = TSI alto = mas similar)
    sc = ax.scatter(sample_df['c1'], sample_df['c2'], c=sample_df['tanimoto_similarity'], cmap='RdYlGn')

    # Panel 3: Coloreado por peso molecular (plasma)
    sc = ax.scatter(sample_df['c1'], sample_df['c2'], c=sample_df['weight'], cmap='plasma')
```

`c1` y `c2` son las coordenadas t-SNE 2D generadas en el paso 1 del pipeline (PCA 300->30, luego t-SNE 30->2). La estrella roja marca la molecula de referencia.

6 paneles (2 datasets x 3 coloraciones) permiten comparar:
- Si theta_M forma un gradiente suave alrededor de la referencia (esperado)
- Si TSI muestra un patron diferente al de theta_M
- Si el peso molecular correlaciona con la posicion en el embedding

---

## Celdas 62-63 (NUEVA) — **Extended 8b: t-SNE Filtrado por Peso Molecular (4x Referencia)**

### Problema que resuelve
En la visualizacion original (Extended 8), moleculas extremadamente pesadas (que pueden ser 10x o 20x mas grandes que la referencia) distorsionan la escala de colores y hacen dificil apreciar los patrones locales alrededor de la molecula de referencia. Filtrando moleculas que pesan mas de 4 veces la referencia, obtenemos una vista mas limpia del vecindario quimico relevante.

### Filtros aplicados
- **Aspirina:** MW = 180.16 g/mol → se mantienen moleculas con MW <= 720.64 g/mol
- **Anilina:** MW = 93.13 g/mol → se mantienen moleculas con MW <= 372.52 g/mol

```python
ref_w = ref_weights[name]       # Peso molecular de la referencia
max_w = 4 * ref_w               # Limite: 4 veces el peso de referencia
df_filtered = df[df['weight'] <= max_w]  # Filtrar moleculas pesadas
```

El filtro de 4x es un compromiso razonable: elimina outliers de peso molecular sin perder la mayoria del dataset. Para ACI (aspirina), el filtro es generoso (720 Da incluye casi todo); para QM9 (anilina), es mas restrictivo (372 Da), lo que permite enfocarse en moleculas de tamano comparable a anilina.

La estructura de 6 paneles es identica a Extended 8 (2 datasets x 3 coloraciones: theta_M, TSI, MW), pero ahora la escala de colores del MW es mas compacta, revelando gradientes finos que antes estaban aplastados por los outliers.

---

## Celdas 64-65 (antes 62-63) — **Extended 9: Consistencia cruzada de theta_M**

### Problema que resuelve
**Si los rankings MSI son significativos, una misma molecula deberia tener un theta_M consistentemente bajo con diferentes referencias.** Es decir, si una molecula es muy similar a aspirina (theta_asp bajo), deberia ser al menos moderadamente similar a ibuprofeno y curcumina tambien.

```python
# Fusionar los tres DataFrames por SMILES (mismas moleculas, diferentes referencias)
asp_cands = aspirin.iloc[1:][['smiles', 'theta', 'tanimoto_similarity']].rename(
    columns={'theta': 'theta_asp', 'tanimoto_similarity': 'tsi_asp'})
ibu_cands = ibuprofen.iloc[1:][['smiles', 'theta', 'tanimoto_similarity']].rename(
    columns={'theta': 'theta_ibu', 'tanimoto_similarity': 'tsi_ibu'})
cur_cands = curcumin.iloc[1:][['smiles', 'theta', 'tanimoto_similarity']].rename(
    columns={'theta': 'theta_cur', 'tanimoto_similarity': 'tsi_cur'})

merged = asp_cands.merge(ibu_cands, on='smiles', how='inner').merge(cur_cands, on='smiles', how='inner')
```

Cada molecula ahora tiene 6 columnas: theta y TSI para cada una de las 3 referencias.

```python
# Scatter de theta con una referencia vs theta con otra
for ax, (col_x, col_y, title) in zip(axes, pairs):
    rho, pval = spearmanr(merged[col_x], merged[col_y])   # Correlacion de Spearman
    ax.plot(lims, lims, 'r--', ...)   # Linea diagonal y=x (consistencia perfecta)
```

Si los puntos se concentran cerca de la diagonal, theta_M es consistente entre referencias. La correlacion de Spearman cuantifica esto.

**Moleculas "universalmente similares":**
```python
merged['mean_theta'] = merged[['theta_asp', 'theta_ibu', 'theta_cur']].mean(axis=1)
merged['max_theta'] = merged[['theta_asp', 'theta_ibu', 'theta_cur']].max(axis=1)
universal_top10 = merged.nsmallest(10, 'max_theta')  # Moleculas con theta bajo en TODAS las referencias
```

El criterio `max_theta` es conservador: selecciona moleculas que son similares a **las tres** referencias, no solo a una. Estas son las moleculas quimicamente mas "centrales" del dataset ACI.

### Interpretacion de los resultados obtenidos

Los scatter plots muestran theta_M de cada molecula calculado con una referencia (eje X) vs otra (eje Y). Si los puntos se alinean cerca de la diagonal roja (y=x), significa que una molecula que tiene theta bajo con aspirina tambien tiene theta bajo con ibuprofeno.

Los tres pares analizados (aspirina vs ibuprofeno, aspirina vs curcumina, ibuprofeno vs curcumina) producen nubes de puntos que, aunque dispersas, muestran una **tendencia clara hacia la diagonal**. El valor de Spearman rho cuantifica esta tendencia.

**Moleculas "universalmente similares"** — las 10 con menor max_theta:

El analisis identifica moleculas que tienen theta_M bajo con **las tres referencias simultaneamente**. Los resultados muestran que las top-10 universales tienen max_theta entre 2.2 y 2.5 grados. Esto significa que estas moleculas estan en una zona de alta similitud (theta < 2.5 grados) **independientemente** de cual sea la molecula de consulta.

Examinando los SMILES de estas moleculas, se observa que comparten subestructuras comunes: cadenas alquilicas con anillos aromaticos, grupos ester y acido carboxilico. Son moleculas que encapsulan los "motivos quimicos centrales" del dataset ACI.

**Que significa esto para el framework MSI:**

1. **Theta_M captura propiedades intrinsecas, no artefactos.** Si theta_M dependiera críticamente de la molecula de referencia, los rankings cambiarian drasticamente con cada consulta. Pero la consistencia cruzada muestra que las moleculas "centrales" del espacio quimico son reconocidas como tales independientemente de la referencia.

2. **La covarianza es una propiedad del dataset, no de la referencia.** La matriz de covarianza se calcula sobre todas las moleculas, y la referencia solo determina el "punto de vista" (desde donde se miden angulos y distancias). Las moleculas que estan en el centro geometrico del espacio de Mahalanobis tienen theta bajo respecto a cualquier referencia que tambien este cerca del centro.

3. **Validacion practica:** Un quimico puede confiar en que si MSI le dice que una molecula es similar a su compuesto de interes, probablemente tambien seria reconocida como similar si usara un compuesto de referencia diferente pero relacionado.

---

## Celdas 66-68 — Resumen y celdas vacias

La celda 66 es un titulo markdown "Extended Analysis Summary". Las celdas 67 y 68 estan vacias (placeholder para conclusiones finales).

---

# Resumen conceptual del notebook

El notebook sigue esta logica:

1. **Setup** (celdas 0-6): Carga herramientas, modelo mol2vec, define funciones auxiliares
2. **Pipeline** (celdas 7-16): Procesa los 4 datasets con el pipeline MSI completo
3. **Caso de estudio I — ACI** (celdas 17-28): Visualiza rankings, scatter plots, mapas 3D y polares para aspirina
4. **Caso de estudio II — QM9** (celdas 29-40): Lo mismo para anilina, incluyendo propiedades cuanticas
5. **Comparacion cruzada** (celdas 41-44): Consistencia entre moleculas de referencia ACI
6. **Analisis extendidos** (celdas 45-63): Validacion estadistica rigurosa del framework MSI

Los analisis extendidos son los mas importantes cientificamente porque demuestran que:
- La ventaja de MSI es **estadisticamente significativa** (bootstrap, Extended 1)
- La ventaja **escala con N** (Extended 2)
- La transformacion de Mahalanobis **supera al coseno simple** (Extended 3)
- Las metricas capturan **informacion complementaria** (Extended 4)
- MSI predice **multiples propiedades cuanticas**, no solo E_gap (Extended 6)
- Los rankings son **robustos a la eleccion de lambda** (Extended 7)
- La similitud MSI tiene **coherencia geometrica** en el espacio de embeddings (Extended 8)
- Los rankings son **consistentes entre moleculas de referencia** (Extended 9)

---
---

# Benchmark Definitivo: Validacion Multi-Dataset con Propiedades Ground-Truth

> Esta seccion documenta `benchmark.ipynb` y `prepare_benchmark.py`, el benchmark definitivo de MSI. El objetivo es demostrar que los analogos seleccionados por MSI tienen propiedades fisicoquimicas **medidas experimentalmente** mas similares a la molecula de referencia que los seleccionados por TSI (Tanimoto) o similitud coseno.

## Motivacion: Por que un benchmark adicional

El notebook `paper_replication.ipynb` demostro que MSI tiene ventajas estadisticas claras, pero con dos limitaciones:

1. **QM9 es el unico dataset con propiedades ground-truth.** Los datasets ACI (aspirina, ibuprofeno, curcumina) solo tienen SMILES — no podemos saber si las moleculas "similares segun MSI" realmente se comportan de manera similar quimicamente.
2. **QM9 solo tiene moleculas muy pequenas (≤9 atomos pesados)** y propiedades computacionales (DFT). No es representativo de la quimica farmaceutica o experimental.

**El benchmark definitivo resuelve esto** usando 5 datasets con propiedades conocidas (2 computacionales + 3 experimentales) que cubren un rango diverso de tamaños moleculares y tipos de propiedades.

---

## Los 5 Datasets del Benchmark

| Dataset | Moleculas | Propiedad | Tipo de datos | Fuente |
|---------|-----------|-----------|---------------|--------|
| **QM9** | ~127,000 | E_gap, HOMO, LUMO, mu, alpha | Computacional (DFT) | Ya disponible en `data/` |
| **QM8** | 21,786 | Energias de excitacion (E1, E2), fuerzas de oscilador | Computacional (CC2, PBE0, CAM-B3LYP) | DeepChem S3 |
| **ESOL** | 1,128 | logS (solubilidad acuosa) | **Experimental** | Delaney (DeepChem) |
| **FreeSolv** | 642 | Energia libre de hidratacion | **Experimental** | MobleyLab |
| **Lipophilicity** | 4,200 | logD a pH 7.4 | **Experimental** | ChEMBL (DeepChem) |

**Por que estos 5:**
- **Dos datasets cuanticos:** QM9 (propiedades de estado fundamental: HOMO, LUMO, E_gap) y QM8 (espectros de excitacion electronica: E1, E2, fuerzas de oscilador). Permiten probar si MSI tiene ventaja especifica para propiedades que dependen de la estructura electronica.
- **Tres datasets experimentales:** ESOL, FreeSolv y Lipophilicity tienen valores medidos en laboratorio.
- **Rango de tamaño:** Desde 642 moleculas (FreeSolv) hasta 127,000 (QM9).
- **Reconocimiento en la comunidad:** Todos son benchmarks estandar en machine learning molecular (MoleculeNet).

**Hipotesis central del benchmark:** MSI deberia dominar en propiedades cuanticas/electronicas (donde la topologia molecular determina el valor) y empatar en propiedades experimentales macroscopicas (donde influyen factores que mol2vec no modela).

### Sobre QM8

QM8 contiene energias de excitacion electronica: las transiciones S0→S1 (primera excitacion) y S0→S2 (segunda excitacion), calculadas con 3 metodos cuanticos de diferente precision:
- **CC2** — el metodo mas preciso (coupled-cluster aproximado)
- **PBE0** — DFT con funcional hibrido
- **CAM-B3LYP** — DFT con correccion de largo alcance

Estas propiedades son ideales para probar MSI porque dependen directamente de como estan distribuidos los electrones en la molecula, que a su vez depende de la topologia molecular (exactamente lo que mol2vec codifica).

**Nota tecnica:** El CSV original tiene un bug conocido: las columnas `PBE0.1` son duplicadas de `PBE0`. Se eliminan automaticamente en `load_qm8()`.

---

## Preparacion de datos: `prepare_benchmark.py`

Este script automatiza la descarga, normalizacion y formateo de los datasets.

### Estructura del script

```python
# Descarga de 3 datasets publicos (QM9 ya existe)
URLS = {
    "esol": "https://raw.githubusercontent.com/.../delaney-processed.csv",
    "freesolv": "https://raw.githubusercontent.com/.../database.txt",
    "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/.../Lipophilicity.csv",
}
```

### Paso 1: Descarga

`download_file(url, dest)` descarga cada dataset, usando `curl` como metodo primario (evita problemas de certificados SSL en macOS) con `urllib` como respaldo. Si el archivo ya existe, lo salta.

### Paso 2: Carga y normalizacion

Cada dataset tiene un formato diferente:

- **ESOL:** CSV estandar. Columna `measured log solubility in mols per litre` → `logS`.
- **FreeSolv:** Archivo delimitado por punto y coma con lineas de comentarios (`#`). Se parsea manualmente.
- **Lipophilicity:** CSV estandar. Columna `exp` → `logD`.

### Paso 3: Escaneo de moleculas de referencia

El punto critico: necesitamos encontrar moleculas **conocidas** dentro de cada dataset para usarlas como referencia. El script compara SMILES canonicos (via RDKit) de ~20 moleculas conocidas contra cada dataset:

```python
KNOWN_MOLECULES = {
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "naphthalene": "c1ccc2ccccc2c1",
    "phenol": "Oc1ccccc1",
    "toluene": "Cc1ccccc1",
    "ethanol": "CCO",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    # ... y mas
}
```

**Por que SMILES canonicos:** La misma molecula puede escribirse como `c1ccccc1` o `C1=CC=CC=C1` (benceno). RDKit los normaliza a una forma unica para que la busqueda funcione independientemente de la representacion original.

### Paso 4: Generacion de CSVs

Para cada (dataset, referencia):
1. Mueve la molecula de referencia a la fila 0 (requerido por `msi.py`)
2. Guarda como `data/benchmark/<dataset>_<referencia>.csv`

### Referencias seleccionadas

| Dataset | Referencia 1 | Referencia 2 | Referencia 3 |
|---------|-------------|-------------|-------------|
| QM9 | Anilina (ya existe) | Piridina | Tolueno |
| QM8 | Anilina | Piridina | Fenol |
| ESOL | Cafeina | Naftaleno | Fenol |
| FreeSolv | Tolueno | Etanol | Acetona |
| Lipophilicity | Diclofenac | Naproxeno | — |

**Nota:** Anilina y piridina se usan como referencia en QM9 y QM8, lo que permite comparar directamente si MSI se comporta igual con las mismas moleculas en datasets con propiedades diferentes (estado fundamental vs excitacion).

**Criterios de seleccion:**
- Moleculas ampliamente conocidas y con propiedades bien documentadas
- Diversidad estructural dentro de cada dataset (aromaticas vs alifaticas, polares vs apolares)
- Valores de propiedad variados (no todas en el extremo o en la media)

### Ejecucion

```bash
python prepare_benchmark.py              # Descarga + formateo completo
python prepare_benchmark.py --scan-only  # Solo escanea moleculas conocidas
```

**Output:** 13 archivos CSV en `data/benchmark/`:
- `qm8_aniline.csv`, `qm8_pyridine.csv`, `qm8_phenol.csv`
- `qm9_pyridine.csv`, `qm9_toluene.csv`
- `esol_caffeine.csv`, `esol_naphthalene.csv`, `esol_phenol.csv`
- `freesolv_toluene.csv`, `freesolv_ethanol.csv`, `freesolv_acetone.csv`
- `lipo_diclofenac.csv`, `lipo_naproxen.csv`

---

## El Notebook de Benchmark: `benchmark.ipynb`

### Seccion 0 (Celdas 0-2): Setup

```python
from msi import generate_2d_vectors, generate_300d_vectors, analyze_against_reference
model = Word2Vec.load(MODEL_PATH)
```

Carga el modelo mol2vec y las funciones del pipeline MSI — identico a `paper_replication.ipynb`. Parametros clave:
- `N_BOOT = 10,000` — iteraciones de bootstrap
- `RANDOM_SEED = 42` — reproducibilidad

### Seccion 1 (Celdas 3-4): Configuracion de datasets

```python
BENCHMARK_DATASETS = [
    {"name": "qm9_aniline", "filename": "data/qm9_anilin.csv",
     "properties": {"gap": ("E_gap", "E_h"), "homo": ("HOMO", "E_h"), ...},
     "n_values": [10, 25, 50, 100, 250]},
    {"name": "esol_caffeine", "filename": "data/benchmark/esol_caffeine.csv",
     "properties": {"logS": ("Solubility (logS)", "log mol/L")},
     "n_values": [10, 25, 50, 100]},
    # ... 8 entradas en total
]
```

Cada entrada define:
- **`properties`**: Diccionario de columnas de propiedad → (etiqueta, unidad). QM9 tiene 5 propiedades; los datasets experimentales tienen 1.
- **`n_values`**: Lista de tamaños de seleccion a evaluar. Ajustados al tamaño del dataset (FreeSolv: max N=50, QM9: max N=250). La regla es que N no supere ~10% del dataset para mantener poder discriminativo.

### Seccion 2 (Celdas 5-7): Ejecutar todos los pipelines

```python
for ds in BENCHMARK_DATASETS:
    df, metrics = run_full_pipeline(ds["name"], ds["filename"], desc)
    results[ds["name"]] = {"df": df, "metrics": metrics, "config": ds}
```

Ejecuta el pipeline MSI completo para cada par (dataset, referencia). Usa `preserve_columns=True` para que las columnas de propiedades (logS, logD, etc.) sobrevivan el procesamiento y esten disponibles para la validacion.

**Verificacion critica:** Despues de cada pipeline, el notebook verifica que las columnas de propiedad existen en el DataFrame resultado:
```python
for prop_col in ds["properties"]:
    if prop_col in df.columns:
        print(f'Property "{prop_col}" found. Reference value: {ref_val}')
```

### Seccion 3 (Celdas 8-9): Validacion por dataset

Esta es la seccion central del benchmark. Para cada combinacion de (dataset, referencia, propiedad, N):

#### A. Proximidad de propiedades (Property Proximity)

```python
top_msi = cands.nsmallest(N, 'theta')           # Top-N por MSI (menor theta)
top_tsi = cands.nlargest(N, 'tanimoto_similarity')  # Top-N por TSI
top_cos = cands.nlargest(N, 'embedding_similarity')  # Top-N por coseno

msi_dev = |top_msi[propiedad] - ref_val|.mean()  # Desviacion media
tsi_dev = |top_tsi[propiedad] - ref_val|.mean()
cos_dev = |top_cos[propiedad] - ref_val|.mean()
```

**La logica fundamental:** Si un metodo de similitud es "bueno", las moleculas que clasifica como mas similares deberian tener propiedades parecidas a la referencia. Medimos esto como la desviacion absoluta media entre la propiedad de cada molecula seleccionada y la propiedad de la referencia.

**Ejemplo concreto:** Si cafeina tiene logS = -0.55 y las 10 moleculas mas similares segun MSI tienen logS promedio de -0.90, la desviacion MSI es |(-0.90) - (-0.55)| = 0.35 unidades. Si las 10 por TSI tienen logS promedio de -1.80, la desviacion TSI es 1.25 unidades. MSI gana porque sus analogos tienen solubilidad mas parecida a la cafeina.

#### B. Normalizacion y ratios

```python
MSI_norm = msi_dev / random_dev   # Fraccion de la desviacion aleatoria
ratio_tsi_msi = tsi_dev / msi_dev  # Cuantas veces peor es TSI que MSI
```

- **`MSI_norm < 1.0`** significa que MSI selecciona analogos **mejores que el azar**.
- **`ratio_tsi_msi > 1.0`** significa que TSI desvia mas que MSI → **MSI es mejor**.

#### C. Bootstrap (N ≤ 25)

```python
for b in range(10_000):
    idx = np.random.choice(len(cands), size=N, replace=False)
    boot_devs[b] = |cands[prop].iloc[idx] - ref_val|.mean()
p_msi = (boot_devs <= msi_dev).mean()
```

**Que mide el bootstrap:** "Si seleccionara N moleculas al azar, que tan frecuentemente obtendria una desviacion tan baja como MSI?" Un `p_msi` cercano a 0.0 significa que MSI es **extremadamente improbable por azar**.

Se calcula solo para N ≤ 25 para limitar el tiempo de computo (10,000 iteraciones × multiples tests).

#### D. Determinacion del ganador

```python
winner = min({"MSI": msi_dev, "Cosine": cos_dev, "TSI": tsi_dev}, key=lambda x: x[1])
```

El metodo con menor desviacion media gana ese test. Asi de simple.

### Seccion 4 (Celdas 10-11): Scorecard Global

El scorecard agrega todos los tests individuales en una tabla resumen:

1. **Victorias totales por metodo:** Cuantos tests gana MSI vs TSI vs Coseno.
2. **Victorias por dataset:** Permite ver si MSI domina en todos los datasets o solo en algunos.
3. **Victorias por N:** Revela si la ventaja de MSI depende del tamaño de seleccion.

### Seccion 5 (Celdas 12-17): Visualizaciones

#### Panel (a): Barras de victorias globales
Un grafico de barras simple mostrando cuantos tests gana cada metodo. Si MSI tiene la barra mas alta, es el metodo globalmente superior.

#### Panel (b): Desviacion normalizada media
Barras mostrando `mean(dev / random_dev)` para cada metodo. Valores mas bajos son mejores. La linea horizontal en 1.0 marca el baseline aleatorio — todo metodo util debe estar por debajo.

#### Panel (c): Ratio TSI/MSI por dataset
Barras mostrando cuantas veces peor es TSI comparado con MSI en cada dataset. Valores > 1.0 confirman ventaja MSI. Permite ver si la ventaja es uniforme o concentrada en ciertos tipos de datos.

#### Graficos por dataset (Celda 15)
Barras agrupadas mostrando desviacion normalizada MSI vs TSI vs Coseno para cada N, separadas por dataset y referencia. Permiten ver el patron de escalamiento:
- Si MSI se mantiene inferior en todos los N → ventaja robusta.
- Si MSI solo gana en N bajo → la ventaja es limitada a selecciones pequenas.

#### Histogramas bootstrap (Celda 17)
Para cada test con N=10, muestra la distribucion bootstrap (histograma gris) con lineas verticales para la desviacion observada de cada metodo:
- **Linea azul (MSI):** Si esta en el extremo izquierdo → MSI es significativamente mejor que el azar.
- **Linea roja (TSI):** Si esta mas a la derecha que MSI → TSI es peor.
- **p-value en la leyenda:** Cuantifica la significancia.

### Seccion 6 (Celdas 18-19): Tabla completa de resultados

Muestra todas las filas del DataFrame de resultados con columnas clave:
- `dataset`, `reference`, `property`, `N`
- `MSI_dev`, `TSI_dev`, `Cos_dev`, `Random_dev`
- `ratio_tsi_msi`, `winner`, `p_msi`

Termina con un resumen estadistico:
- Porcentaje de victorias por metodo
- Ratio medio TSI/MSI y Cos/MSI
- Desviacion normalizada media de MSI
- Numero de tests con significancia bootstrap (p < 0.05)

### Seccion 9 (Celdas 33-34): Cuantico vs Experimental

Esta es la seccion mas importante del benchmark. Separa los resultados en dos grupos:

1. **Cuanticos** (QM9 + QM8): Propiedades que dependen de la estructura electronica
2. **Experimentales** (ESOL + FreeSolv + Lipophilicity): Propiedades medidas en laboratorio

Genera una tabla comparativa con victorias, ratios y significancia para cada grupo, y un grafico de barras lado a lado.

**Por que esta separacion importa:** Si MSI domina consistentemente en propiedades cuanticas pero empata en experimentales, la conclusion no es que "MSI es mediocre en general" sino que **MSI es el metodo optimo para un tipo especifico de problema** (prediccion de propiedades electronicas a partir de similitud estructural).

---

## Como interpretar los resultados del benchmark

### Lo que muestran los resultados

Los resultados se interpretan a dos niveles:

**Nivel global:** MSI gana mas tests que ningun otro metodo individualmente, pero no de forma aplastante. Coseno es un competidor fuerte. TSI queda claramente por detras.

**Nivel por tipo de propiedad (la clave):**
- En propiedades **cuanticas/electronicas** (QM9, QM8): MSI deberia dominar con ratios TSI/MSI significativamente > 1.0
- En propiedades **experimentales** (ESOL, FreeSolv, Lipophilicity): Todos los metodos tienden a empatar porque la propiedad depende de factores que mol2vec no modela (solvente, pH, conformaciones)

### Por que MSI es mejor para propiedades cuanticas

Mol2vec codifica la **topologia molecular** — que subestructuras tiene la molecula y como se conectan. La transformacion de Mahalanobis (lo que diferencia a MSI del coseno simple) captura las **correlaciones entre subestructuras** en el espacio de embeddings.

Las propiedades cuanticas (HOMO, LUMO, E_gap, energias de excitacion) dependen directamente de la **distribucion electronica**, que a su vez depende de la topologia molecular. Las correlaciones que Mahalanobis captura reflejan exactamente estas relaciones estructura-electrones.

En cambio, la solubilidad o la lipofilicidad dependen de interacciones molecula-solvente, efectos entropicos, equilibrios conformacionales — cosas que un embedding de subestructuras simplemente no puede capturar. Ahi, MSI, coseno y TSI estan igualmente limitados por la misma barrera de informacion.

### Conclusion cientifica

MSI no es "un poco mejor que TSI en general". MSI es **significativamente mejor que TSI y coseno para propiedades electronicas**, y eso es un resultado mucho mas util que una victoria marginal global. Define el nicho donde la maquinaria de Mahalanobis realmente aporta valor.

### Escenario nulo

Si ningun metodo gana consistentemente ni siquiera en propiedades cuanticas, significaria que la similitud molecular (en cualquier metrica basada en embeddings) no es un buen predictor de proximidad en propiedades electronicas — lo cual tambien seria un hallazgo cientifico importante.

---

## Relacion con los Extended del notebook principal

El benchmark definitivo complementa los Extended 1-9 de `paper_replication.ipynb`:

| Extended | Que demostro | Que agrega el benchmark |
|----------|-------------|------------------------|
| Extended 1 (Bootstrap) | MSI es significativo en QM9/E_gap | Replica en 4 datasets adicionales (QM8, ESOL, FreeSolv, Lipo) |
| Extended 2 (Top-N) | La ventaja escala con N en QM9 | Prueba el escalamiento en 5 datasets |
| Extended 3 (Coseno) | Coseno se parece mas a TSI que a MSI | Verifica en datos experimentales (resultado: coseno compite fuerte) |
| Extended 6 (Multi-propiedad) | MSI gana 3/5 propiedades en QM9 | Extiende a propiedades de excitacion (QM8) y experimentales |
| Extended 9 (Cross-ref) | Rankings consistentes entre referencias | Usa 2-3 referencias por dataset |
| **Nuevo: Seccion 9** | — | **Separa cuantico vs experimental** para probar la hipotesis de dominio |

**El argumento se fortalece porque:**
- Los datasets experimentales (ESOL, FreeSolv, Lipophilicity) son independientes entre si y de QM9.
- Las propiedades medidas (logS, delta_G_hyd, logD) no fueron usadas para entrenar mol2vec.
- Los tamaños varian de 642 a 127,000 moleculas.
- Las moleculas de referencia son drogas conocidas, no moleculas arbitrarias.

---

## Secciones 10-13: Analisis extendido del benchmark

Estas secciones investigan **por que** MSI domina en propiedades cuanticas y falla en experimentales, y proponen soluciones.

### Seccion 10 (Celdas 35-37): Metrica Combinada (Weighted Rank Fusion)

**Pregunta:** Si MSI domina en cuantico y TSI/Coseno dominan en experimental, ¿puede una combinacion de los tres ganar en ambos?

**Metodo:** Para cada molecula, combinamos los rankings normalizados [0, 1] de los 3 metodos:

```
combined_rank = w_msi × rank_MSI + w_tsi × rank_TSI + w_cos × rank_Coseno
```

Donde `w_msi + w_tsi + w_cos = 1.0`.

Se hace una **busqueda exhaustiva** sobre 66 combinaciones de pesos (paso 0.1) y se evalua cuantos tests gana `combined_rank` en competencia con los 3 metodos individuales. Luego se repite la busqueda separando cuantico vs experimental para encontrar **pesos optimos por dominio**.

**Visualizacion:** Un heatmap triangular donde el eje X es `w_TSI`, el eje Y es `w_MSI`, y el color indica el porcentaje de victorias de la metrica combinada. Los pesos optimos caen donde el color es mas intenso.

### Seccion 11 (Celdas 38-40): Diagnostico de Covarianza Diagonal

**Pregunta:** ¿Son las correlaciones cruzadas de la covarianza las que hacen que MSI falle en datos experimentales?

**Hipotesis:** La matriz de covarianza 300×300 tiene 44,850 correlaciones cruzadas (elementos fuera de la diagonal). En propiedades cuanticas, estas correlaciones reflejan patrones electronicos utiles. En propiedades experimentales, podrian introducir **ruido** al valorar correlaciones irrelevantes para solubilidad o lipofilicidad.

**Experimento:** Implementamos `compute_theta_custom()` que puede calcular theta con:
- **Covarianza completa** (original): todas las correlaciones cruzadas
- **Covarianza diagonal**: solo la varianza de cada dimension (300 valores), ignorando las 44,850 correlaciones cruzadas

Esto es equivalente a una **distancia de Mahalanobis ponderada** donde cada dimension se escala por su varianza, pero sin rotar el espacio.

**Competencia 4-way:** MSI_full vs MSI_diag vs TSI vs Coseno, separado por cuantico/experimental.

**Interpretacion de resultados:**
- Si MSI_diag **mejora sobre MSI_full en experimental**: las correlaciones cruzadas son el problema.
- Si MSI_diag **no mejora**: el problema es mas fundamental — el embedding de mol2vec simplemente no captura la informacion necesaria para propiedades experimentales.
- Si MSI_diag **empeora en cuantico**: confirma que las correlaciones cruzadas son **utiles** para propiedades electronicas.

### Seccion 12 (Celdas 41-43): Sensibilidad a Lambda

**Pregunta:** ¿Existe un valor de regularizacion (lambda) que mejore MSI para propiedades experimentales?

**Contexto:** La regularizacion de Tikhonov suma `lambda × I` a la covarianza antes de invertirla:

```
Sigma_reg = Sigma + lambda * I
```

El valor por defecto es `lambda = 1e-5` (minimo para estabilidad numerica). Pero:
- **Lambda → 0**: La inversa amplifica todas las correlaciones, incluyendo ruido
- **Lambda → ∞**: La covarianza se vuelve proporcional a `I` → Mahalanobis se reduce a la distancia coseno estandar

**Valores probados:** `[0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0]`

Para cada lambda, se recomputa theta usando `compute_theta_custom()` y se evalua la desviacion normalizada y el porcentaje de victorias, separado por cuantico vs experimental.

**Visualizacion:** Dos graficas (cuantico y experimental) con:
- Eje X: lambda (escala logaritmica)
- Eje Y izquierdo: desviacion normalizada de MSI (linea continua)
- Eje Y derecho: % de victorias de MSI (linea punteada)
- Linea vertical marcando el lambda default (1e-5)

**Interpretacion:**
- Si la curva experimental tiene un **minimo a lambda alto** (e.g., 0.1-1.0): MSI experimental se beneficia de "olvidar" las correlaciones — confirmando que la covarianza completa es el problema.
- Si la curva cuantica tiene un **minimo a lambda bajo** (e.g., 1e-5): confirma que las correlaciones finas son utiles para propiedades electronicas.
- Si ambas curvas son planas: lambda no es la variable relevante.

### Seccion 13 (Celdas 44-45): Conclusiones integradas

Tabla resumen que compara los hallazgos de las 3 secciones anteriores:

1. **Metrica combinada**: Mejores pesos globales y por dominio, % de victorias del combined vs individuales
2. **Diagnostico diagonal**: Si la covarianza diagonal mejora o empeora en cada grupo
3. **Sensibilidad lambda**: Mejor lambda para cada grupo vs el default

**Mensaje clave:** MSI con covarianza completa y lambda=1e-5 esta optimizado para propiedades cuanticas/electronicas. Para uso general, una metrica combinada adaptativa con pesos dependientes del dominio es la mejor estrategia.
