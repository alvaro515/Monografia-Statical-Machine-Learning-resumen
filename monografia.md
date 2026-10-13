# Statical Machine Learning

## K-Nearest Neighbors

## Tree  Models

### 1. Introducción

Un árbol de decisión es un modelo que predice haciendo preguntas de sí o no sobre las variables, una tras otra, hasta llegar a una respuesta. Se parece a un diagrama de flujo: "¿los puertos distintos son más de 14? Si es así, la conexión es sospechosa". Cada pregunta divide los datos en dos grupos y el proceso se repite dentro de cada grupo. Este es el algoritmo CART (*Classification and Regression Trees*), propuesto por Breiman, Friedman, Olshen y Stone en 1984.

| Elemento | Qué es |
| --- | --- |
| Nodo raíz | La primera pregunta, que usa todos los datos. |
| Nodo interno | Una pregunta intermedia que divide los datos en dos ramas. |
| Hoja | Un grupo final, sin más preguntas. Aquí se da la predicción. |
| Profundidad | Cantidad máxima de preguntas seguidas desde la raíz hasta una hoja. |

Lo que hace el árbol es partir el espacio de las variables en rectángulos y dar una predicción por rectángulo. Sirve para dos tareas:

- **Clasificación**: la hoja predice la clase más frecuente de sus observaciones, y la proporción de esa clase da la probabilidad.
- **Regresión**: la hoja predice el promedio de $y$ de sus observaciones.

Entre sus ventajas están que se lee fácilmente, no necesita escalar las variables y capta relaciones no lineales e interacciones sin que haya que indicarlas. Su principal debilidad es que es inestable: un cambio pequeño en los datos puede producir un árbol muy distinto, es decir, tiene varianza alta. Si se deja crecer sin límite, memoriza los datos. Esa debilidad es el punto de partida del bagging y del bosque aleatorio, que se ven en la sección siguiente.

---

### 2. Comprenderlo matemáticamente

Esta parte explica cómo el árbol elige sus preguntas. La idea de fondo: en cada nodo se prueban muchos cortes posibles y se elige el que deja los dos grupos más "puros", es decir, con observaciones de una sola clase.

#### 2.1 Notación

| Símbolo | Significado |
| --- | --- |
| $N$ | Número de filas, o sea, observaciones |
| $P$ | Número de variables predictoras |
| $t$ | Un nodo del árbol; $N_t$ es cuántas observaciones tiene |
| $t_L$, $t_R$ | Los nodos hijos de la izquierda y de la derecha |
| $j$, $s$ | Variable elegida para el corte y valor de corte: $X_j \le s$ va a la izquierda |
| $K$ | Número de clases (en clasificación) |
| $p_k$ | Proporción de observaciones de la clase $k$ dentro del nodo |
| $I(t)$ | Impureza del nodo $t$: qué tan mezcladas están sus clases |
| $\Delta I$ | Reducción de impureza que produce un corte |
| $\lvert T \rvert$ | Número de hojas del árbol $T$ |
| $\alpha$ | Penalización por hoja en la poda, un número mayor o igual a 0 |

#### 2.2 Medir qué tan mezclado está un nodo

En clasificación hay dos medidas comunes:

$$G(t) = \sum_{k=1}^{K} p_k (1 - p_k) = 1 - \sum_{k=1}^{K} p_k^2 \qquad \text{(Impureza Gini)}$$

$$H(t) = -\sum_{k=1}^{K} p_k \log_2 p_k \qquad \text{(Entropía)}$$

Las dos valen 0 cuando el nodo es puro, es decir, cuando todas sus observaciones son de la misma clase, y crecen cuanto más mezcladas están. Con dos clases, el Gini máximo es 0.5, cuando hay mitad y mitad. En regresión se usa la varianza de $y$ dentro del nodo:

$$I(t) = \frac{1}{N_t} \sum_{i \in t} (y_i - \bar{y}_t)^2$$

donde $\bar{y}_t$ es el promedio de $y$ en el nodo.

#### 2.3 Elegir el mejor corte

Para cada variable $j$ y cada valor de corte $s$, se calcula cuánto baja la impureza al dividir el nodo en dos:

$$\Delta I = I(t) - \frac{N_L}{N_t} I(t_L) - \frac{N_R}{N_t} I(t_R)$$

Cada hijo cuenta en proporción a su tamaño, porque un hijo pequeño y puro importa menos que uno grande y puro. El corte elegido es el $(j, s)$ con mayor $\Delta I$. Como los valores de corte posibles son los puntos medios entre valores consecutivos de la variable, el árbol los revisa todos.

El algoritmo completo, llamado partición binaria recursiva, es:

1. Se parte del nodo raíz con todas las observaciones.
2. En el nodo actual se prueban todos los cortes $(j, s)$ y se escoge el de mayor $\Delta I$.
3. Se divide el nodo en dos hijos y se repite el paso 2 en cada uno.
4. Se detiene un nodo cuando es puro o cuando se cumple una regla de parada, como una profundidad máxima o un mínimo de observaciones por hoja.

Es un algoritmo **voraz** (*greedy*): en cada nodo toma el mejor corte de ese momento, sin mirar si otro corte peor ahora permitiría mejores cortes después.

#### 2.4 Ejemplo con números

Con las 16 conexiones de la sección 3.1, hay 8 normales y 8 sospechosas. El nodo raíz está totalmente mezclado:

$$G(\text{raíz}) = 1 - \left(\tfrac{8}{16}\right)^2 - \left(\tfrac{8}{16}\right)^2 = 0.5$$

Se prueban dos cortes. Primero, intentos por minuto $\le 5.5$: a la izquierda quedan 7 conexiones, todas normales ($G = 0$), y a la derecha 9, de las cuales 1 es normal y 8 sospechosas:

$$G(t_R) = 1 - \left(\tfrac{1}{9}\right)^2 - \left(\tfrac{8}{9}\right)^2 = 0.198$$

$$\Delta G = 0.5 - \tfrac{7}{16}(0) - \tfrac{9}{16}(0.198) = 0.5 - 0.111 = 0.389$$

Segundo, puertos distintos $\le 14$: a la izquierda quedan las 8 conexiones normales y a la derecha las 8 sospechosas. Ambos hijos son puros:

$$\Delta G = 0.5 - \tfrac{8}{16}(0) - \tfrac{8}{16}(0) = 0.5$$

El segundo corte reduce más la impureza (0.5 frente a 0.389), así que el árbol lo elige. Como los dos hijos ya son puros, no hay nada más que dividir.

#### 2.5 Predicción

Para predecir una observación nueva se la hace bajar por el árbol según las respuestas a cada pregunta, hasta llegar a una hoja. En clasificación, la probabilidad de la clase $k$ es la proporción $p_k$ de esa hoja. En regresión, la predicción es el promedio $\bar{y}$ de la hoja.

#### 2.6 Cuándo parar: la poda

Un árbol que crece hasta que todas sus hojas son puras clasifica sin error los datos de entrenamiento, pero memoriza el ruido. Hay dos formas de controlarlo. La primera es detener el crecimiento antes, con una profundidad máxima o un mínimo de observaciones por hoja. La segunda, la **poda por costo-complejidad**, deja crecer el árbol completo y luego elimina ramas. Para un árbol $T$ se minimiza:

$$C_\alpha(T) = \text{error}(T) + \alpha \, \lvert T \rvert$$

Es decir, el error del árbol más una penalización por cada hoja. Con $\alpha = 0$ se conserva el árbol completo y, a medida que $\alpha$ crece, se podan más ramas hasta quedar solo la raíz. El valor de $\alpha$ se elige con validación cruzada.

#### 2.7 Importancia de las variables

La importancia de una variable es la suma de las reducciones de impureza $\Delta I$ de todos los cortes que la usan, ponderadas por el tamaño de cada nodo y normalizadas para que todas las importancias sumen 1.

---

### 3. Aplicación en el código

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from itertools import product
```

- **numpy** y **pandas** sirven, como antes, para los cálculos y las tablas.
- **DecisionTreeClassifier** es el modelo de árbol para clasificación. Para regresión existe DecisionTreeRegressor, que se usa igual.
- **export_text** imprime el árbol como reglas escritas, lo que permite leerlo.
- **train_test_split** separa los datos en una parte para entrenar y otra para probar.
- **itertools** arma las combinaciones de valores con product, útil para probar varios hiperparámetros.

#### 3.1 Ejemplo

Con las mismas 16 conexiones de la sección de boosting:

```python
datos = pd.DataFrame({
    'intentos_por_minuto': [2, 3, 5, 4, 6, 3, 2, 50, 80, 120, 90, 150, 70, 5, 60, 110],
    'puertos_distintos':   [1, 1, 2, 1, 2, 1, 1, 40, 60, 200, 35, 150, 80, 3, 25, 90],
    'sospechoso':          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
})

X = datos[['intentos_por_minuto', 'puertos_distintos']]
y = datos['sospechoso']

arbol = DecisionTreeClassifier(max_depth=2, random_state=1)
arbol.fit(X, y)

nuevos = pd.DataFrame({'intentos_por_minuto': [90, 4], 'puertos_distintos': [45, 2]})

print(arbol.predict_proba(nuevos)[:, 1])
print(arbol.predict(nuevos))
print(export_text(arbol, feature_names=list(X.columns)))
```

DecisionTreeClassifier crea el modelo vacío. max_depth=2 permite como máximo dos preguntas seguidas y random_state=1 fija el desempate cuando dos cortes sirven igual. fit entrena el modelo. predict_proba da la probabilidad de que sea sospechosa, predict da la clase y export_text escribe las reglas del árbol.

Se ejecutó esto y salió:

```
[1. 0.]
[1 0]
|--- puertos_distintos <= 14.00
|   |--- class: 0
|--- puertos_distintos >  14.00
|   |--- class: 1
```

Aunque se permitieron dos niveles, el árbol hizo una sola pregunta: con puertos distintos $\le 14$ la conexión es normal y con más es sospechosa. Es el mismo corte que ganó en la sección 2.4. El valor 14 es el punto medio entre 3, el mayor número de puertos de una conexión normal, y 25, el menor de una sospechosa. Como las dos hojas ya eran puras, el árbol no siguió dividiendo. Con ese corte, la conexión de 45 puertos queda como sospechosa con probabilidad 1 y la de 2 puertos como normal con probabilidad 0.

---

### 4. Ejemplo 1: árbol con dos variables

Se usan los mismos 400 datos inventados de la sección de boosting:

```python
np.random.seed(1)
n = 400

datos = pd.DataFrame({
    'intentos_por_minuto': np.random.randint(1, 200, n),
    'puertos_distintos': np.random.randint(1, 100, n),
    'hora': np.random.randint(0, 24, n),
    'bytes_promedio': np.random.randint(40, 1500, n)
})

z = 0.05 * datos['intentos_por_minuto'] + 0.1 * datos['puertos_distintos'] - 10
probabilidad = 1 / (1 + np.exp(-z))
datos['sospechoso'] = (np.random.uniform(size=n) < probabilidad).astype(int)
```

Ahora se entrena un árbol de profundidad 3 con las dos columnas que sí importan:

```python
predictores = ['intentos_por_minuto', 'puertos_distintos']
X = datos[predictores]
y = datos['sospechoso']

arbol = DecisionTreeClassifier(max_depth=3, random_state=1)
arbol.fit(X, y)

print(export_text(arbol, feature_names=predictores, show_weights=True))

nuevos = pd.DataFrame({'intentos_por_minuto': [150, 10], 'puertos_distintos': [80, 5]})
print(arbol.predict_proba(nuevos)[:, 1])
print(dict(zip(predictores, arbol.feature_importances_.round(3))))
```

show_weights=True añade a cada hoja cuántas conexiones normales y sospechosas quedaron en ella. Se ejecutó esto y salió:

```
|--- puertos_distintos <= 51.50
|   |--- intentos_por_minuto <= 122.50
|   |   |--- intentos_por_minuto <= 108.50
|   |   |   |--- weights: [125.00, 4.00] class: 0
|   |   |--- intentos_por_minuto >  108.50
|   |   |   |--- weights: [10.00, 3.00] class: 0
|   |--- intentos_por_minuto >  122.50
|   |   |--- puertos_distintos <= 10.50
|   |   |   |--- weights: [16.00, 1.00] class: 0
|   |   |--- puertos_distintos >  10.50
|   |   |   |--- weights: [22.00, 48.00] class: 1
|--- puertos_distintos >  51.50
|   |--- intentos_por_minuto <= 79.00
|   |   |--- puertos_distintos <= 79.00
|   |   |   |--- weights: [29.00, 10.00] class: 0
|   |   |--- puertos_distintos >  79.00
|   |   |   |--- weights: [6.00, 19.00] class: 1
|   |--- intentos_por_minuto >  79.00
|   |   |--- puertos_distintos <= 52.50
|   |   |   |--- weights: [1.00, 4.00] class: 1
|   |   |--- puertos_distintos >  52.50
|   |   |   |--- weights: [1.00, 101.00] class: 1

[0.99019608 0.03100775]
{'intentos_por_minuto': 0.406, 'puertos_distintos': 0.594}
```

El árbol se lee de arriba abajo. La primera pregunta es si los puertos distintos son más de 51.5. Si no lo son, la mayoría de las conexiones es normal, salvo la hoja de más de 122.5 intentos por minuto y más de 10.5 puertos, donde 48 de 70 son sospechosas. Si pasan de 51.5 puertos, la mayoría son sospechosas, salvo las de pocos intentos por minuto y menos de 79 puertos.

La conexión con 150 intentos a 80 puertos cae en la última hoja, que tiene 101 sospechosas y 1 normal, y por eso su probabilidad es 101/102 = 0.99. La de 10 intentos a 5 puertos cae en la primera, con 4 sospechosas y 125 normales, y su probabilidad es 4/129 = 0.03. Las importancias muestran que el árbol usa más los puertos distintos (59 %) que los intentos por minuto (41 %).

---

### 5. Ejemplo 2: sobreajuste

Ahora el árbol recibe las cuatro columnas, incluidas hora y bytes_promedio, que no influyen en nada, y se deja crecer sin límite:

```python
predictores = ['intentos_por_minuto', 'puertos_distintos', 'hora', 'bytes_promedio']
X = datos[predictores]
y = datos['sospechoso']

X_entren, X_prueba, y_entren, y_prueba = train_test_split(X, y, test_size=100, random_state=0)

completo = DecisionTreeClassifier(random_state=1)
completo.fit(X_entren, y_entren)

print('entrenamiento: ', np.mean(completo.predict(X_entren) != y_entren))
print('prueba: ', np.mean(completo.predict(X_prueba) != y_prueba))
print('profundidad: ', completo.get_depth())
print('hojas: ', completo.get_n_leaves())
print(dict(zip(predictores, completo.feature_importances_.round(3))))
```

La separación es la misma del boosting: 300 conexiones para entrenar y 100 para probar. El error es la proporción de predicciones distintas de la clase real. Sin max_depth, el árbol sigue dividiendo hasta que todas las hojas son puras. Se ejecutó esto y salió:

```
entrenamiento:  0.0
prueba:  0.14
profundidad:  9
hojas:  43
{'intentos_por_minuto': 0.415, 'puertos_distintos': 0.495, 'hora': 0.045, 'bytes_promedio': 0.046}
```

En entrenamiento no falla nunca, pero con conexiones que no vio falla el 14 %. Para lograr esos cero errores necesitó 9 niveles y 43 hojas, y hasta usó las columnas hora y bytes_promedio, que son puro ruido, con importancias cercanas al 4.5 % cada una. Eso es sobreajuste: el árbol memorizó los datos, incluido el azar.

---

### 6. Ejemplo 3: controlar el tamaño del árbol

La primera forma de evitar el sobreajuste es limitar la profundidad. Se entrenan árboles de distinta profundidad y se compara su error en entrenamiento y en prueba:

```python
filas = []
for prof in [1, 2, 3, 4, 6, 8, 10]:
    m = DecisionTreeClassifier(max_depth=prof, random_state=1)
    m.fit(X_entren, y_entren)
    filas.append({
        'max_depth': prof,
        'hojas': m.get_n_leaves(),
        'error_entren': np.mean(m.predict(X_entren) != y_entren),
        'error_prueba': np.mean(m.predict(X_prueba) != y_prueba)
    })

print(pd.DataFrame(filas).round(3))
```

Se ejecutó esto y salió:

```
   max_depth  hojas  error_entren  error_prueba
0          1      2         0.220          0.28
1          2      4         0.190          0.25
2          3      8         0.117          0.16
3          4     16         0.093          0.14
4          6     33         0.043          0.15
5          8     42         0.003          0.14
6         10     43         0.000          0.14
```

El error de entrenamiento siempre baja al crecer el árbol, pero el de prueba deja de mejorar desde profundidad 4. Con un árbol de 16 hojas se logra el mismo 14 % de error que con uno de 43. Un árbol más grande no predice mejor, solo memoriza más. Con solo 100 conexiones de prueba, una diferencia de 1 punto equivale a una conexión, así que estas diferencias pequeñas no son concluyentes.

La segunda forma es la poda por costo-complejidad de la sección 2.6. En scikit-learn se controla con ccp_alpha, el $\alpha$ de la fórmula:

```python
ruta = DecisionTreeClassifier(random_state=1).cost_complexity_pruning_path(X_entren, y_entren)
alphas = ruta.ccp_alphas

filas = []
for a in alphas[::3]:
    m = DecisionTreeClassifier(random_state=1, ccp_alpha=a)
    m.fit(X_entren, y_entren)
    filas.append({
        'ccp_alpha': a,
        'hojas': m.get_n_leaves(),
        'error_prueba': np.mean(m.predict(X_prueba) != y_prueba)
    })

print(pd.DataFrame(filas).round(4))
```

cost_complexity_pruning_path calcula los valores de $\alpha$ en los que el árbol pierde una rama, y aquí se prueba uno de cada tres. Se ejecutó esto y salió:

```
   ccp_alpha  hojas  error_prueba
0     0.0000     43          0.14
1     0.0031     36          0.14
2     0.0033     28          0.15
3     0.0044     22          0.14
4     0.0057     17          0.13
5     0.0064     13          0.14
6     0.0093      7          0.16
7     0.0251      4          0.25
8     0.1550      1          0.54
```

Con $\alpha = 0$ queda el árbol completo de 43 hojas. Al subir $\alpha$ el árbol se poda hasta 13 o 17 hojas sin perder precisión. Desde alrededor de 0.009 empieza a empeorar, y con $\alpha = 0.155$ queda solo la raíz, que predice siempre la misma clase y falla el 54 %.

---

### 7. Ejemplo 4: validación cruzada

Para elegir los hiperparámetros sin depender de una sola separación de los datos se usa validación cruzada con $K = 5$ pliegues, como en la sección 2.5 de boosting. Se prueban 9 combinaciones de profundidad máxima y mínimo de observaciones por hoja:

```python
np.random.seed(1)
grupo = np.random.choice(5, size=len(datos), replace=True)

errores = []
for prof, hoja in product([2, 4, 8], [1, 10, 25]):
    errores_pliegues = []
    for k in range(5):
        entren = grupo != k
        prueba = grupo == k
        m = DecisionTreeClassifier(max_depth=prof, min_samples_leaf=hoja, random_state=1)
        m.fit(X[entren], y[entren])
        errores_pliegues.append(np.mean(m.predict(X[prueba]) != y[prueba]))
    errores.append({
        'max_depth': prof,
        'min_samples_leaf': hoja,
        'error_promedio': np.mean(errores_pliegues)
    })

tabla_errores = pd.DataFrame(errores)
print(tabla_errores.round(3))
```

np.random.choice reparte las conexiones al azar en 5 grupos. Cada combinación se entrena 5 veces, apartando un grupo distinto para probar, y se promedian los 5 errores. min_samples_leaf=10 obliga a que cada hoja tenga al menos 10 conexiones. Se ejecutó esto y salió:

```
   max_depth  min_samples_leaf  error_promedio
0          2                 1           0.221
1          2                10           0.221
2          2                25           0.221
3          4                 1           0.185
4          4                10           0.156
5          4                25           0.192
6          8                 1           0.169
7          8                10           0.138
8          8                25           0.192
```

La mejor combinación es max_depth=8 con min_samples_leaf=10, con 13.8 % de error. Con profundidad 2 el árbol es demasiado simple y falla el 22 % sin importar la otra opción. Con min_samples_leaf=25 las hojas son demasiado grandes y también empeora. El mejor resultado aparece con un árbol profundo, pero con hojas de tamaño razonable: ambas opciones se complementan.

---

### 8. Hiperparámetros de un árbol de decisión

| Hiperparámetro | Qué controla |
| --- | --- |
| max_depth | Profundidad máxima del árbol. Es la forma más directa de limitar la complejidad. Si no se indica, el árbol crece sin límite. |
| min_samples_split | Mínimo de observaciones que debe tener un nodo para poder dividirse. |
| min_samples_leaf | Mínimo de observaciones por hoja. Evita hojas con muy pocos casos, que suelen ser ruido. |
| max_features | Cuántas variables se consideran en cada corte. Con todas es un árbol normal; con menos se usa en el bosque aleatorio. |
| criterion | Medida de impureza: gini o entropy en clasificación; squared_error en regresión. |
| ccp_alpha | Penalización por hoja en la poda por costo-complejidad. Con 0 no se poda. |

---

### 9. Conclusiones

- Un árbol de decisión predice con una secuencia de preguntas de sí o no sobre las variables y da una predicción por cada hoja: la clase más frecuente en clasificación y el promedio en regresión.
- En cada nodo el árbol elige el corte que más reduce la impureza, medida con Gini, entropía o varianza. Lo hace de forma voraz, un nodo a la vez.
- Es fácil de leer y no requiere escalar las variables, pero es inestable y sobreajusta si crece sin límite: en el ejemplo, error 0 en entrenamiento y 14 % en prueba.
- La complejidad se controla con la profundidad máxima, el mínimo de observaciones por hoja o la poda por costo-complejidad, y sus valores se eligen con validación cruzada.
- La varianza alta de un solo árbol es el motivo del bagging y del bosque aleatorio, que combinan muchos árboles para estabilizar la predicción.

---

## Bagging and the Ramdom Forest

### 1. Introducción

Los métodos de ensamble (ensemble learning) parten de la analogía del experimento de Sir Francis Galton (1906) sobre la "sabiduría de las multitudes" (Wisdom of Crowds), donde el promedio de múltiples estimaciones individuales supera en precisión a cualquier estimación aislada. En árboles de decisión, combinar las predicciones de múltiples modelos reduce significativamente la varianza sin aumentar el sesgo.

### 2. Bagging (Bootstrap Aggregating)

Introducido por Leo Breiman en 1996, el Bagging consiste en entrenar múltiples modelos predictivos de forma independiente utilizando diferentes submuestras de datos generadas mediante muestreo bootstrap (muestreo aleatorio con reemplazo)

#### 2.1. Definición matemática y algoritmo

Dado un conjunto de entrenamiento con $$N$$ registros y $$P$$ variables predictoras $$X = (X_1, X_2, \dots, X_P)$$:

1. Se fijan $$M$$ (número de modelos) y $$n$$ ($$n < N$$, tamaño del subsistema).
2. Para cada iteración $$m = 1, \dots, M$$:
   - Se extrae una muestra bootstrap de tamaño $$n$$ con reemplazo para formar la submuestra $$(Y_m, X_m)$$.
   - Se entrena un modelo individual para obtener la regla de decisión $$f_m(X)$$.
3. **Predicción Agregada**:
   - **Regresión o Estimación de Probabilidades**:

$$\hat{f}(X) = \frac{1}{M} \sum_{m=1}^{M} f_m(X)$$

- **Clasificación**: Se aplica **voto mayoritario** (*majority vote*) sobre las predicciones de las $$M$$ reglas $$f_m(X)$$.

#### 2.2. Ejemplo

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Definición del modelo de Bagging con árboles de decisión
model_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    random_state=1
)

# Entrenamiento y predicción
[se quitó el vínculo sospechoso](X_train, y_train)
y_pred = model_bagging.predict(X_test)
```

### 3. Random Forest

El Random Forest extiende el principio de bagging sobre árboles de decisión agregando una segunda capa de aleatorización: la selección aleatoria de variables (*feature sampling*) en cada división de nodo (*split*). Esto descorrelaciona los árboles entre sí, reduciendo la varianza global del ensemble de manera más efectiva que el bagging simple.

#### 3.1 Definición matemática y algoritmo

1. Se extrae una muestra bootstrap con reemplazo del conjunto de datos.
2. En cada división de nodo, en lugar de evaluar los $$P$$ predictores disponibles, el algoritmo selecciona al azar un subconjunto de $$p < P$$ variables sin reemplazo.
   - **Regla empírica recomendada**: $$p = \sqrt{P}$$ para problemas de clasificación.
3. Se evalúa el criterio de división (p. ej., minimización de la **Impureza Gini** o **Entropía**) únicamente dentro de las $$p$$ variables elegidas para determinar el corte óptimo $$s_{j_k}$$.
4. Se hace crecer el árbol completamente sin podar.
5. Se repite el proceso para los $$M$$ árboles y se promedian/votan los resultados:

$$\hat{f}_{rf}(X) = \frac{1}{M} \sum_{m=1}^{M} f_m(X; \theta_m)$$

(donde $$\theta_m$$ representa la aleatorización tanto de datos como de variables en el árbol $$m$$ ).

#### 3.2. Ejemplo

```python
from sklearn.ensemble import RandomForestClassifier

# Crear y entrenar el modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',  # mtry = sqrt(P)
    random_state=1,
    oob_score=True
)
[se quitó el vínculo sospechoso](X_train, y_train)

# Evaluación fuera de bolsa (Out-of-Bag)
print(f"Precisión Out-of-Bag (OOB): {rf.oob_score_:.4f}")
```

## Boosting

### 1. Introducción

El boosting es una técnica para crear un ensamble de modelos, es decir, un conjunto de modelos cuyas predicciones se combinan. Se usa casi siempre con árboles de decisión. A diferencia del bagging y del bosque aleatorio, donde los modelos se entrenan por separado, en el boosting se entrenan en secuencia: cada modelo nuevo trata de reducir el error del anterior. Por eso requiere más ajuste de parámetros que el bagging.

Existen tres variantes principales:

| Variante | Descripción |
| --- | --- |
| AdaBoost | Aumenta el peso de las observaciones mal clasificadas para que el siguiente modelo se concentre en ellas. |
| Gradient boosting | Plantea el problema como la minimización de una función de costo. Cada modelo nuevo se ajusta al pseudo-residuo, es decir, a lo que aún no explica el conjunto. |
| Stochastic gradient boosting | Igual que el anterior, pero en cada ronda usa una muestra aleatoria de filas y de columnas. Es la variante más general y la más usada. |

### 2. Comprenderlo matemáticamente

Esta parte explica las fórmulas con palabras y con números sencillos. Lo importante es la idea de fondo: en cada ronda se entrena un modelo, se ve dónde se equivocó y el siguiente modelo se encarga de corregir eso.

#### 2.1 Notación

| Símbolo | Significado |
| --- | --- |
| $N$ | Número de filas, o sea, observaciones |
| $i$ | Número de una fila, $i = 1, 2, \dots, N$ |
| $M$ | Número máximo de modelos |
| $m$ | Número del modelo actual, $m = 1, 2, \dots, M$ |
| $w_i$ | Peso de la fila $i$, que indica qué tan importante es esa fila |
| $f_m$ | Modelo número $m$ |
| $F_m$ | Ensamble formado por los primeros $m$ modelos |
| $e_m$ | Error ponderado del modelo $m$ |
| $\alpha_m$ | Peso o confianza del modelo $m$ |
| $y_i$ | Valor real de la fila $i$ |
| $\eta$ | Factor de contracción, un número entre 0 y 1 |
| $K$ | Número de pliegues de la validación cruzada |

#### 2.2 AdaBoost

La idea de AdaBoost en pocas palabras: se entrena un modelo, se mira qué filas clasificó mal, se les sube la importancia, es decir, el peso, y se entrena el siguiente modelo con esos pesos nuevos. Así el siguiente modelo se esfuerza más en las filas difíciles. Al final se combinan todos los modelos, y los que se equivocaron menos tienen más voz.

Se explica paso a paso con un ejemplo de 5 observaciones llamadas A, B, C, D y E.

**Paso 1. Todas las filas empiezan con el mismo peso.**

$$w_i = \frac{1}{N} \qquad F_0 = 0$$

Si hay $N$ filas, cada una pesa $1/N$, así ninguna tiene ventaja al principio. $F_0 = 0$ quiere decir que el ensamble empieza vacío y todavía no predice nada.

Ejemplo: con 5 observaciones, cada una pesa $1/5 = 0.2$.

**Paso 2. Medir cuánto se equivoca el modelo.** Se entrena el modelo $f_m$ usando los pesos y se calcula su error ponderado:

$$e_m = \sum_{i\ \text{mal clasificadas}} w_i$$

Se suman solo los pesos de las filas que el modelo clasificó mal. Las que acertó no cuentan.

Ejemplo: el primer modelo solo falla en D, que pesa 0.2, así que $e_1 = 0.2$. Si hubiera fallado en D y en A, sería $0.2 + 0.2 = 0.4$.

**Paso 3. Decidir cuánta confianza se le da al modelo.**

$$\alpha_m = \ln\!\left(\frac{1 - e_m}{e_m}\right)$$

Aquí $\ln$ es el logaritmo natural. Lo que importa es cómo se comporta: si el error es minimo, $\alpha_m$ sale grande, y si el error es grande, sale minimo.

Ejemplo con $e_1 = 0.2$: $\alpha_1 = \log(0.8 / 0.2) = \log(4) \approx 1.39$.

| Error $e_m$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| Confianza $\alpha_m$ | 2.20 | 1.39 | 0.85 | 0.41 | 0 |

Con un error de 0.5 la confianza es 0: el modelo acierta tanto como una moneda al aire, así que no aporta nada.

**Paso 4. Agregar el modelo al ensamble.**

$$F_m = F_{m-1} + \alpha_m f_m$$

El ensamble nuevo es el ensamble anterior más el modelo nuevo, y el modelo nuevo entra multiplicado por su confianza.

**Paso 5. Subir el peso de las filas que falló.** Los pesos de las filas mal clasificadas aumentan según $\alpha_m$. La forma habitual de hacerlo es multiplicar el peso de esas filas por $e^{\alpha_m}$ y después reescalar:

$$w_i \leftarrow w_i \cdot e^{\alpha_m} \quad \text{(solo si la fila } i \text{ fue mal clasificada)}$$

Reescalar quiere decir dividir cada peso entre la suma de todos, para que los pesos vuelvan a sumar 1.

Ejemplo: $e^{1.39} \approx 4$, así que el peso de D se multiplica por 4.

| Observación | A | B | C | D mal clasificada | E | Suma |
| --- | --- | --- | --- | --- | --- | --- |
| Peso antes | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 1 |
| Peso multiplicado | 0.2 | 0.2 | 0.2 | 0.8 | 0.2 | 1.6 |
| Peso reescalado entre 1.6 | 0.125 | 0.125 | 0.125 | 0.5 | 0.125 | 1 |

Ahora D pesa la mitad de todo, y el segundo modelo se esfuerza sobre todo por acertar D.

**Paso 6. Repetir** hasta tener los $M$ modelos. La predicción final es la suma de todos, cada uno multiplicado por su confianza:

$$F = \alpha_1 f_1 + \alpha_2 f_2 + \cdots + \alpha_M f_M$$

**Ejemplo con dos modelos.** Supongamos que el segundo modelo, entrenado con los pesos nuevos, solo falla en A:

| | A | B | C | D | E | Error | Confianza |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pesos con que entrena el modelo 1 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | $e_1 = 0.2$, falla D | $\alpha_1 = 1.39$ |
| Pesos con que entrena el modelo 2 | 0.125 | 0.125 | 0.125 | 0.5 | 0.125 | $e_2 = 0.125$, falla A | $\alpha_2 = \log(7) \approx 1.95$ |

Para un caso nuevo, con las dos clases codificadas como $+1$ y $-1$, el modelo 1 predice $+1$ y el modelo 2 predice $-1$:

$$F = 1.39 \times (+1) + 1.95 \times (-1) = -0.56$$

El resultado es negativo, así que la predicción final es $-1$. Gana el modelo 2 porque tiene más confianza, porque se equivocó menos. Es como una votación donde no todos los votos valen lo mismo.

#### 2.3 Gradient boosting

AdaBoost corrige cambiando los pesos de las filas. Gradient boosting hace otra cosa: mira cuánto se está equivocando el ensamble en cada fila, que es el residuo, entrena el siguiente modelo para predecir justamente ese error y lo suma al ensamble.

$$r_i = y_i - F_{m-1}(x_i) \qquad\qquad F_m = F_{m-1} + \eta\, h_m$$

- La primera fórmula dice: residuo = valor real − lo que predice el ensamble hasta ahora. Es lo que todavía falta.
- La segunda dice: el ensamble nuevo es el anterior más el modelo nuevo $h_m$, que aprendió a predecir ese residuo, multiplicado por $\eta$.
- $\eta$ es un número entre 0 y 1 que hace que cada modelo nuevo corrija solo una parte del error. Con $\eta = 0.5$ corrige la mitad, con $\eta = 0.1$ solo el 10 %.

Ejemplo: el valor real es 100, el primer modelo predice 80 y se usa $\eta = 0.5$. Para simplificar, se supone que cada árbol aprende exactamente el residuo.

| Ronda | Predicción del ensamble | Residuo, 100 menos la predicción | Lo que se suma, $0.5 \times$ residuo |
| --- | --- | --- | --- |
| 1 | 80 | 20 | 10 |
| 2 | 90 | 10 | 5 |
| 3 | 95 | 5 | 2.5 |
| 4 | 97.5 | 2.5 | 1.25 |

En la ronda 1 faltan 20, pero el árbol solo aporta la mitad, que es 10, y la predicción sube a 90. En la ronda 2 faltan 10, se suma la mitad, que es 5, y así sucesivamente. La predicción se acerca a 100 en cada ronda. Con un $\eta$ más pequeño el avance es más lento, pero el modelo se sobreajusta menos.

El pseudo-residuo es la versión general del residuo, para cuando el costo no es el error cuadrático. Se calcula con la derivada de la función de costo $L$:

$$r_i = -\frac{\partial L\big(y_i,\, F_{m-1}(x_i)\big)}{\partial F}$$

La derivada indica en qué dirección cambia el costo si se mueve la predicción, y el signo menos hace que se avance hacia donde el costo baja. Con el error cuadrático, $L = \tfrac{1}{2}(y - F)^2$, el resultado es justo $r_i = y_i - F_{m-1}(x_i)$, el residuo de arriba. En la práctica basta con entender el pseudo-residuo como "lo que le falta al ensamble".

#### 2.4 Regularización

Un modelo puede ajustarse tanto a los datos de entrenamiento que termina memorizando hasta el ruido, y luego falla con datos nuevos. Para evitarlo, la regularización suma al error un castigo por ser complicado:

$$\text{costo total} = \text{error del modelo} + \text{castigo por complejidad}$$

Hay dos tipos de castigo, y se escriben aquí para regresión, que es donde nació la idea.

Ridge o L2, castiga la suma de los coeficientes al cuadrado:

$$\sum_{i=1}^{n}\left(Y_i - b_0 - b_1 X_{i1} - \cdots - b_p X_{ip}\right)^2 + \lambda\left(b_1^2 + \cdots + b_p^2\right)$$

Lasso o L1, castiga la suma de los valores absolutos:

$$\sum_{i=1}^{n}\left(Y_i - b_0 - b_1 X_{i1} - \cdots - b_p X_{ip}\right)^2 + \alpha\left(|b_1| + \cdots + |b_p|\right)$$

Cómo leerlas:

- **Primer término, el error.** $b_0 + b_1 X_{i1} + \cdots$ es lo que predice el modelo para la fila $i$. Al restárselo al valor real $Y_i$ queda el error de esa fila; se eleva al cuadrado y se suman todas las filas.
- **Segundo término, el castigo.** Los $b$ son los coeficientes, los números que multiplican a cada variable. Mientras más grandes sean, más castigo.
- **$\lambda$ y $\alpha$** controlan qué tan fuerte es el castigo. Si valen 0, no hay castigo. Si son muy grandes, el modelo se ve obligado a usar coeficientes casi nulos.
- En XGBoost la idea es la misma y los parámetros se llaman lambda y alpha. Este $\alpha$ no tiene que ver con el $\alpha_m$ de AdaBoost.

Ejemplo con $b_1 = 3$, $b_2 = -2$ y un parámetro de castigo igual a 10:

| Tipo | Cálculo | Castigo |
| --- | --- | --- |
| L2 ridge | $10 \times (3^2 + (-2)^2) = 10 \times 13$ | 130 |
| L1 lasso | $10 \times (\lvert 3 \rvert + \lvert -2 \rvert) = 10 \times 5$ | 50 |

El modelo busca el menor costo total, así que le conviene bajar los coeficientes para reducir el castigo, aunque pierda un poco de precisión en el entrenamiento.

#### 2.5 Error y validación cruzada

**Error.** El código de los ejemplos mide el error así, con $p_i$ la probabilidad que predice el modelo y $y_i$ igual a 0 o 1:

$$\text{error} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\big(\,|y_i - p_i| > 0.5\,\big)$$

- $\mathbf{1}(\cdot)$ vale 1 si lo que está dentro se cumple y 0 si no.
- Para cada fila se calcula la diferencia entre el valor real y la probabilidad. Si pasa de 0.5, el modelo estaba más cerca de la clase equivocada y se cuenta como error.
- Al final se promedia: errores entre total de filas.

Ejemplo con 4 filas:

| Fila | Real $y$ | Predicción $p$ | $\lvert y - p \rvert$ | ¿Error? |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.9 | 0.1 | No |
| 2 | 0 | 0.2 | 0.2 | No |
| 3 | 0 | 0.6 | 0.6 | Sí |
| 4 | 1 | 0.4 | 0.6 | Sí |

Hay 2 errores en 4 filas, así que el error es $2/4 = 0.5$, es decir, 50 %.

**Validación cruzada.** Los datos se dividen en $K$ grupos llamados pliegues. Se entrena $K$ veces y en cada una se deja fuera un pliegue distinto para probar. Luego se promedian los errores de las $K$ pruebas:

$$\text{CV} = \frac{1}{K}\sum_{k=1}^{K} E_k$$

donde $E_k$ es el error medido en el pliegue $k$. Por ejemplo, con 5 pliegues y errores de 0.33, 0.32, 0.34, 0.33 y 0.33, la suma es 1.65 y el error de validación cruzada es $1.65 / 5 = 0.33$. Se repite para cada combinación de hiperparámetros y se elige la que da el menor CV.

---

### 3. Aplicación en el código

```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from itertools import product
```

- **numpy**,  sirve para hacer cálculos con números, como promedios o números al azar.
- **pandas**,  sirve para trabajar con tablas de datos.
- **xgboost** trae el modelo de boosting. De ahí se importa la clase XGBClassifier.
- **scikit-learn**, ayuda a separar los datos en una parte para entrenar y otra para probar, con train_test_split.
- **itertools** permite armar todas las combinaciones posibles de valores con product, algo útil cuando se prueban varios hiperparámetros.

#### 3.1 Ejemplo

Detectar conexiones sospechosas en una red. De cada una se sabe cuántos intentos hace por minuto y a cuántos puertos distintos intenta entrar: una conexión normal usa uno o dos puertos, y quien escanea la red toca decenas. Hay 16 conexiones ya revisadas:

```python
datos = pd.DataFrame({
    'intentos_por_minuto': [2, 3, 5, 4, 6, 3, 2, 50, 80, 120, 90, 150, 70, 5, 60, 110],
    'puertos_distintos':   [1, 1, 2, 1, 2, 1, 1, 40, 60, 200, 35, 150, 80, 3, 25, 90],
    'sospechoso':          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
})

X = datos[['intentos_por_minuto', 'puertos_distintos']]
y = datos['sospechoso']

modelo = XGBClassifier(objective='binary:logistic', n_estimators=50,
                       max_depth=2, learning_rate=0.3)
modelo.fit(X, y)

nuevos = pd.DataFrame({'intentos_por_minuto': [90, 4], 'puertos_distintos': [45, 2]})

print(modelo.predict_proba(nuevos)[:, 1])
print(modelo.predict(nuevos))
```

La columna sospechoso vale 1 si la conexión era un ataque y 0 si era normal. Es la que se quiere predecir, así que va en y, y las otras dos columnas van en X.

XGBClassifier crea el modelo vacío. objective='binary:logistic' indica que hay dos respuestas posibles, n_estimators=50 que se construyen 50 árboles en secuencia, max_depth=2 que cada árbol es pequeño y learning_rate=0.3 es el $\eta$ de las fórmulas. fit entrena el modelo.

Luego se prueban dos conexiones nuevas: una con 90 intentos por minuto a 45 puertos y otra con 4 intentos a 2 puertos. predict_proba da la probabilidad de que sean sospechosas y predict da la clase, 1 o 0.

Se ejecutó esto y salió:

```
[0.8663118 0.1336882]
[1 0]
```

La primera tiene 87 % de probabilidad de ser sospechosa y el modelo predice 1. La segunda tiene 13 % y predice 0. Los demás ejemplos siguen estos mismos pasos y solo cambian los datos y los hiperparámetros.

---

### 4. Ejemplo 1: XGBoost con dos variables

Con 16 conexiones no se puede comprobar mucho, así que ahora se crean 400 con datos inventados:

```python
np.random.seed(1)
n = 400

datos = pd.DataFrame({
    'intentos_por_minuto': np.random.randint(1, 200, n),
    'puertos_distintos': np.random.randint(1, 100, n),
    'hora': np.random.randint(0, 24, n),
    'bytes_promedio': np.random.randint(40, 1500, n)
})

z = 0.05 * datos['intentos_por_minuto'] + 0.1 * datos['puertos_distintos'] - 10
probabilidad = 1 / (1 + np.exp(-z))
datos['sospechoso'] = (np.random.uniform(size=n) < probabilidad).astype(int)

print(datos.head())
```

Se ejecutó esto y salió:

```
   intentos_por_minuto  puertos_distintos  hora  bytes_promedio  sospechoso
0                   38                 98    21            1269           1
1                  141                 39    16            1426           0
2                   73                 64    19            1480           0
3                  138                 47    22             560           1
4                  134                 90     5             944           1
```

np.random.seed(1) hace que los números al azar sean siempre los mismos. Los datos se armaron para que una conexión sea sospechosa según sus intentos por minuto y sus puertos distintos, con algo de azar. Las columnas hora y bytes_promedio no influyen en nada y se dejan para el ejemplo 2.

Ahora se entrena con las dos columnas que sí importan:

```python
predictores = ['intentos_por_minuto', 'puertos_distintos']
resultado = 'sospechoso'

X = datos[predictores]
y = datos[resultado]

modelo = XGBClassifier(objective='binary:logistic', subsample=0.63)
modelo.fit(X, y)

nuevos = pd.DataFrame({'intentos_por_minuto': [150, 10], 'puertos_distintos': [80, 5]})
print(modelo.predict_proba(nuevos)[:, 1])
```

Se ejecutó esto y salió:

```
[0.9986865 0.0018378]
```

Es el mismo procedimiento. La novedad es subsample=0.63: cada árbol se construye con un 63 % de las filas elegidas al azar, y eso lo convierte en stochastic gradient boosting. La conexión con 150 intentos a 80 puertos casi seguro es sospechosa y la de 10 intentos a 5 puertos casi seguro es normal.

En Python el factor de contracción se llama learning_rate. Si se escribe eta no da error, pero no tiene efecto.

---

### 5. Ejemplo 2: sobreajuste

Ahora el modelo recibe las cuatro columnas, incluidas hora y bytes_promedio, y se deja aprender sin freno:

```python
predictores = ['intentos_por_minuto', 'puertos_distintos', 'hora', 'bytes_promedio']
X = datos[predictores]
y = datos['sospechoso']

X_entren, X_prueba, y_entren, y_prueba = train_test_split(X, y, test_size=100, random_state=0)

modelo_defecto = XGBClassifier(objective='binary:logistic', n_estimators=250,
                               max_depth=6, reg_lambda=0, learning_rate=0.3,
                               subsample=1)
modelo_defecto.fit(X_entren, y_entren)

pred_entren = modelo_defecto.predict_proba(X_entren)[:, 1]
pred_prueba = modelo_defecto.predict_proba(X_prueba)[:, 1]

print('entrenamiento: ', np.mean(abs(y_entren - pred_entren) > 0.5))
print('prueba: ', np.mean(abs(y_prueba - pred_prueba) > 0.5))
```

train_test_split aparta 100 conexiones para probar y deja 300 para entrenar. random_state=0 fija la separación. El modelo usa 250 árboles de profundidad 6 sin penalización (reg_lambda=0) y el error se mide con la regla de la sección 2.5.

Se ejecutó esto y salió:

```
entrenamiento:  0.0
prueba:  0.18
```

En entrenamiento no falla nunca, pero con conexiones que no vio falla el 18 %. Eso es sobreajuste: el modelo memorizó las 300 conexiones, ruido incluido, y no generaliza.

---

### 6. Ejemplo 3: regularización

Es el mismo ejemplo y los mismos datos, pero con tres cambios para frenar el sobreajuste:

```python
modelo_penalizado = XGBClassifier(objective='binary:logistic', n_estimators=250,
                                  max_depth=6, reg_lambda=10, learning_rate=0.1,
                                  subsample=0.63)
modelo_penalizado.fit(X_entren, y_entren)

pred_entren = modelo_penalizado.predict_proba(X_entren)[:, 1]
pred_prueba = modelo_penalizado.predict_proba(X_prueba)[:, 1]

print('entrenamiento: ', np.mean(abs(y_entren - pred_entren) > 0.5))
print('prueba: ', np.mean(abs(y_prueba - pred_prueba) > 0.5))
```

| Argumento | Antes | Ahora | Efecto |
| --- | --- | --- | --- |
| reg_lambda | 0 | 10 | Aplica una penalización L2 a los árboles complejos. |
| learning_rate | 0.3 | 0.1 | Reduce el tamaño de cada paso. |
| subsample | 1 | 0.63 | Cada árbol usa solo una parte de las filas. |

reg_lambda vale 10 porque son pocos datos; con tablas grandes se usan valores mucho más altos. La penalización L1 se activa con reg_alpha. En R los parámetros se llaman lambda y alpha.

Se ejecutó esto y salió:

```
entrenamiento:  0.04666666666666667
prueba:  0.15
```

Ahora el error es 4.7 % en entrenamiento y 15 % en prueba. Pierde algo de precisión en entrenamiento, pero con conexiones nuevas mejora (antes 18 %) y la diferencia entre los dos errores baja de 18 a unos 10 puntos. Ya no memoriza.

---

### 7. Ejemplo 4: validación cruzada

Hasta ahora eta y max_depth se elegían a ojo. Con validación cruzada se prueban varios valores y se elige el mejor:

```python
grupos = np.random.choice(range(5), size=len(X), replace=True)
errores = []

for eta, profundidad in product([0.1, 0.5, 0.9], [2, 4, 8]):
    modelo = XGBClassifier(objective='binary:logistic', n_estimators=100,
                           max_depth=profundidad, learning_rate=eta)
    errores_pliegues = []
    for k in range(5):
        en_prueba = grupos == k
        X_entren = X.loc[~en_prueba]; y_entren = y[~en_prueba]
        X_prueba = X.loc[en_prueba]; y_prueba = y[en_prueba]

        modelo.fit(X_entren, y_entren)
        pred = modelo.predict_proba(X_prueba)[:, 1]
        errores_pliegues.append(np.mean(abs(y_prueba - pred) > 0.5))
    errores.append({
        'eta': eta,
        'profundidad': profundidad,
        'error_promedio': np.mean(errores_pliegues)
    })

tabla_errores = pd.DataFrame(errores)
print(tabla_errores.round(3))
```

np.random.choice reparte las conexiones al azar en 5 grupos y product arma las 9 combinaciones de eta y profundidad. Cada combinación se entrena 5 veces, apartando un grupo distinto para probar, y se promedian los 5 errores. Son 45 modelos: con 400 filas es rápido, con tablas grandes puede tardar.

Se ejecutó esto y salió:

```
   eta  profundidad  error_promedio
0  0.1            2           0.146
1  0.1            4           0.166
2  0.1            8           0.159
3  0.5            2           0.190
4  0.5            4           0.160
5  0.5            8           0.159
6  0.9            2           0.184
7  0.9            4           0.166
8  0.9            8           0.159
```

La mejor es eta=0.1 con max_depth=2, con 14.6 % de error. Las demás quedan entre 15.9 % y 19 %. Con estos datos conviene un eta pequeño y árboles poco profundos.

---

### 8. Hiperparámetros de XGBoost

| Hiperparámetro | Qué controla |
| --- | --- |
| eta / learning_rate | Factor de contracción entre 0 y 1. Controla el tamaño de cada paso. Con datos ruidosos se recomiendan valores pequeños, como 0.1. |
| nrounds / n_estimators | Número de rondas, es decir, de árboles. Si eta es pequeño, se necesitan más rondas. |
| max_depth | Profundidad máxima del árbol. En boosting se usan árboles poco profundos, lo que evita interacciones complejas producidas por el ruido. |
| subsample y colsample_bytree | Fracción de filas y de columnas que usa cada árbol. Ayudan a evitar el sobreajuste. |
| lambda / reg_lambda y alpha / reg_alpha | Parámetros de regularización. Controlan el sobreajuste mediante penalizaciones L2 y L1. |

---

### 9. Conclusiones

- El boosting ajusta una secuencia de modelos y da más peso a las observaciones con errores grandes en las rondas siguientes.
- Stochastic gradient boosting con árboles es la variante más general, y XGBoost la implementa.
- El boosting es propenso al sobreajuste, por lo que sus hiperparámetros deben ajustarse.
- La regularización, tanto L1 como L2, reduce el sobreajuste al penalizar la complejidad del modelo.
- La validación cruzada permite elegir los hiperparámetros, y es importante porque XGBoost tiene muchos.
