## Boosting

### 1. Introducción

El boosting es una técnica para crear un ensamble de modelos, es decir, un conjunto de modelos cuyas predicciones se combinan. Se usa casi siempre con árboles de decisión. A diferencia del bagging (y del bosque aleatorio), donde los modelos se entrenan por separado, en el boosting se entrenan en secuencia: cada modelo nuevo trata de reducir el error del anterior. Por eso requiere más ajuste de parámetros que el bagging.

Existen tres variantes principales:

| Variante | Descripción |
|---|---|
| AdaBoost | Aumenta el peso de las observaciones mal clasificadas para que el siguiente modelo se concentre en ellas. |
| Gradient boosting | Plantea el problema como la minimización de una función de costo. Cada modelo nuevo se ajusta al pseudo-residuo, es decir, a lo que aún no explica el conjunto. |
| Stochastic gradient boosting | Igual que el anterior, pero en cada ronda usa una muestra aleatoria de filas y de columnas. Es la variante más general y la más usada. |

#### Términos básicos

| Término | Qué significa | Ejemplo |
|---|---|---|
| Observación (fila) | Un caso individual dentro de la tabla de datos. | Un préstamo. |
| Variable predictora (columna) | Característica que se usa para predecir. | `borrower_score`, el puntaje del cliente. |
| Variable respuesta (`outcome`) | Lo que se quiere predecir. | Si el préstamo se pagó o no. |
| Modelo | Regla que se aprende de los datos para hacer predicciones. | Un árbol de decisión. |
| Entrenar (`fit`) | Hacer que el modelo aprenda a partir de los datos. | `xgb.fit(X, y)`. |
| Conjunto de entrenamiento | Datos con los que el modelo aprende. | Todas las filas menos las 10,000 de prueba. |
| Conjunto de prueba | Datos que el modelo no vio al entrenar, usados para evaluarlo. | 10,000 filas reservadas. |
| Árbol de decisión | Serie de preguntas de sí/no que termina en una predicción. | ¿`borrower_score` > 0.5? |
| Ensamble | Conjunto de modelos cuyas predicciones se combinan. | 250 árboles trabajando juntos. |
| Peso | Número que indica la importancia de una observación o de un modelo. | Un peso de 0.5 pesa más que uno de 0.125. |
| Error | Proporción de predicciones incorrectas. | 35.3 % de error en prueba. |
| Residuo | Diferencia entre el valor real y el predicho. | Real 100, predicho 80: residuo 20. |
| Pseudo-residuo | Residuo generalizado a cualquier función de costo; es lo que ajusta cada modelo en gradient boosting. | Con error cuadrático coincide con el residuo. |
| Función de costo | Medida de qué tan mal predice el modelo. | El error cuadrático. |
| Probabilidad | Número entre 0 y 1 que expresa qué tan probable es un resultado. | 0.8 = 80 % de probabilidad de incumplimiento. |
| Factor de contracción (`eta`) | Número entre 0 y 1 que reduce el aporte de cada árbol nuevo. | `eta = 0.1`. |
| Sobreajuste | El modelo se ajusta demasiado a los datos de entrenamiento y falla con datos nuevos. | 13.3 % de error en entrenamiento y 35.3 % en prueba. |
| Regularización | Penalización a la complejidad del modelo para evitar el sobreajuste. | `reg_lambda=1000`. |
| Hiperparámetro | Parámetro que se fija antes de entrenar. | `max_depth`, `n_estimators`. |
| Pliegue (fold) | Cada uno de los grupos en que se divide el dato en la validación cruzada. | 5 pliegues. |
| Validación cruzada | Evaluar el modelo varias veces usando un pliegue distinto como prueba. | Entrenar 5 veces y promediar el error. |

---

### 2. Fórmulas

Las fórmulas están escritas en LaTeX. Se visualizan en la vista previa de VS Code y en GitHub.

#### 2.1 Notación

| Símbolo | Significado |
|---|---|
| $N$ | Número de filas (observaciones) |
| $i$ | Índice de una fila, $i = 1, 2, \dots, N$ |
| $M$ | Número máximo de modelos |
| $m$ | Índice del modelo actual, $m = 1, 2, \dots, M$ |
| $w_i$ | Peso de la fila $i$ |
| $f_m$ | Modelo número $m$ |
| $F_m$ | Ensamble formado por los primeros $m$ modelos |
| $e_m$ | Error ponderado del modelo $m$ |
| $\alpha_m$ | Peso (confianza) del modelo $m$ |

#### 2.2 AdaBoost

Inicialización: todas las observaciones tienen el mismo peso y el ensamble parte vacío.

$$w_i = \frac{1}{N} \qquad F_0 = 0$$

Por ejemplo, con 5 observaciones (A, B, C, D, E) cada una pesa $1/5 = 0.2$.

Error ponderado: se entrena $f_m$ usando los pesos y se suman los pesos de las observaciones que clasificó mal.

$$e_m = \sum_{i\ \text{mal clasificadas}} w_i$$

Si el modelo solo falla en D, entonces $e_m = 0.2$.

Peso del modelo:

$$\alpha_m = \log\!\left(\frac{1 - e_m}{e_m}\right)$$

Aquí $\log$ es el logaritmo natural. Con $e_m = 0.2$: $\alpha_m = \log(0.8/0.2) = \log(4) \approx 1.39$. A menor error, mayor $\alpha_m$:

| $e_m$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| $\alpha_m$ | 2.20 | 1.39 | 0.85 | 0.41 | 0 |

Con $e_m = 0.5$ el modelo no aporta nada, porque acierta tanto como una moneda.

Actualización del ensamble:

$$F_m = F_{m-1} + \alpha_m f_m$$

Actualización de pesos: Bruce et al. indican solo que los pesos de las observaciones mal clasificadas aumentan según $\alpha_m$. La forma habitual de hacerlo es multiplicar esos pesos por $e^{\alpha_m}$ y luego reescalar para que sumen 1.

$$w_i \leftarrow w_i \cdot e^{\alpha_m} \quad \text{(solo si la observación } i \text{ fue mal clasificada)}$$

En el ejemplo, $e^{1.39} \approx 4$, de modo que el peso de D pasa de 0.2 a 0.8 antes de reescalar:

| Observación | A | B | C | D (mal clasificada) | E |
|---|---|---|---|---|---|
| Peso antes | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Peso después | 0.125 | 0.125 | 0.125 | 0.5 | 0.125 |

El proceso se repite hasta completar los $M$ modelos y la predicción final es:

$$F = \alpha_1 f_1 + \alpha_2 f_2 + \cdots + \alpha_M f_M$$

Cada modelo contribuye en proporción a su $\alpha_m$, por lo que los modelos con menor error tienen más influencia.

#### 2.3 Gradient boosting

En lugar de modificar pesos, cada modelo nuevo $h_m$ se ajusta al residuo del ensamble actual y se suma con un factor de contracción $\eta$ (entre 0 y 1).

$$r_i = y_i - F_{m-1}(x_i) \qquad\qquad F_m = F_{m-1} + \eta\, h_m$$

La versión general usa una función de costo $L$, y el pseudo-residuo es el negativo de su derivada respecto a la predicción:

$$r_i = -\frac{\partial L\big(y_i,\, F_{m-1}(x_i)\big)}{\partial F}$$

Con el error cuadrático $L = \tfrac{1}{2}(y - F)^2$ se obtiene $r_i = y_i - F_{m-1}(x_i)$, que es el residuo de la primera expresión.

Ejemplo: valor real 100, primera predicción 80 y $\eta = 0.5$. Se supone que cada árbol predice exactamente el residuo.

| Ronda | Predicción del ensamble | Residuo | Aporte del árbol nuevo ($0.5 \times$ residuo) |
|---|---|---|---|
| 1 | 80 | 20 | 10 |
| 2 | 90 | 10 | 5 |
| 3 | 95 | 5 | 2.5 |
| 4 | 97.5 | 2.5 | 1.25 |

La predicción se aproxima a 100 en cada ronda. Un $\eta$ pequeño hace el avance más lento, pero reduce el sobreajuste.

#### 2.4 Regularización

La regularización agrega al costo un término que penaliza la complejidad del modelo. Las dos formas, planteadas para regresión:

Ridge (L2), penaliza la suma de los coeficientes al cuadrado:

$$\sum_{i=1}^{n}\left(Y_i - b_0 - b_1 X_{i1} - \cdots - b_p X_{ip}\right)^2 + \lambda\left(b_1^2 + \cdots + b_p^2\right)$$

Lasso (L1), penaliza la suma de los valores absolutos:

$$\sum_{i=1}^{n}\left(Y_i - b_0 - b_1 X_{i1} - \cdots - b_p X_{ip}\right)^2 + \alpha\left(|b_1| + \cdots + |b_p|\right)$$

El primer término es el error del modelo y el segundo es la penalización. Los parámetros $\lambda$ y $\alpha$ controlan su magnitud y en XGBoost corresponden a `lambda` y `alpha`. Este $\alpha$ no es el $\alpha_m$ de AdaBoost.

Ejemplo con $b_1 = 3$, $b_2 = -2$ y un parámetro de penalización igual a 10:

| Tipo | Cálculo | Penalización |
|---|---|---|
| L2 | $10 \times (3^2 + (-2)^2) = 10 \times 13$ | 130 |
| L1 | $10 \times (\lvert 3 \rvert + \lvert -2 \rvert) = 10 \times 5$ | 50 |

Con coeficientes más grandes la penalización aumenta, por lo que el modelo tiende a valores pequeños.

#### 2.5 Error y validación cruzada

El error que calcula el código de los ejemplos, con $p_i$ la probabilidad predicha y $y_i \in \{0, 1\}$:

$$\text{error} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\big(\,|y_i - p_i| > 0.5\,\big)$$

donde $\mathbf{1}(\cdot)$ vale 1 si la condición se cumple y 0 si no. Es la proporción de observaciones mal predichas. Si $y = 1$ y $p = 0.8$, la diferencia es 0.2 y la predicción es correcta; si $p = 0.3$, la diferencia es 0.7 y es un error.

En la validación cruzada con $K$ pliegues se entrena $K$ veces dejando fuera un pliegue distinto cada vez, y se promedian los errores $E_k$ de cada pliegue:

$$\text{CV} = \frac{1}{K}\sum_{k=1}^{K} E_k$$

Por ejemplo, con errores de 0.33, 0.32, 0.34, 0.33 y 0.33 en cinco pliegues: $\text{CV} = 1.65 / 5 = 0.33$.

---

### 3. Importaciones

Los fragmentos de código de Bruce et al. no muestran las importaciones, pero son necesarias para ejecutarlos:

```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from itertools import product
```

| Elemento | Qué es | Qué hace |
|---|---|---|
| `import ... as ...` | Instrucción de Python | Carga una librería y le asigna un nombre corto. |
| `from ... import ...` | Instrucción de Python | Carga solo una herramienta específica de una librería. |
| `numpy` (`np`) | Librería | Cálculo numérico. |
| `pandas` (`pd`) | Librería | Manejo de tablas de datos. |
| `xgboost` | Librería | Implementación de boosting. |
| `sklearn` | Librería (scikit-learn) | Herramientas de aprendizaje automático. |
| `itertools` | Librería estándar | Herramientas para combinar listas. |

---

### 4. Ejemplo 1: XGBoost básico

```python
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

xgb = XGBClassifier(objective='binary:logistic', subsample=0.63)
xgb.fit(X, y)
```

Datos: `loan3000` es una tabla de 3000 préstamos. `borrower_score` es el puntaje del cliente, `payment_inc_ratio` es la relación entre la cuota y el ingreso, y `outcome` indica si el préstamo se pagó o no.

Clase

| Nombre | Qué es | Qué hace |
|---|---|---|
| `XGBClassifier(...)` | Clase de la librería xgboost | Crea un modelo XGBoost de clasificación, sin entrenar. |

Argumentos de `XGBClassifier`

| Argumento | Qué es | Qué hace |
|---|---|---|
| `objective='binary:logistic'` | Tipo de problema | Define clasificación en dos categorías con salida en forma de probabilidad. |
| `subsample=0.63` | Fracción de filas | Cada árbol usa el 63 % de las filas, elegidas al azar. |

Métodos

| Nombre | Qué es | Qué hace |
|---|---|---|
| `xgb.fit(X, y)` | Método del modelo | Entrena el modelo con las variables predictoras `X` y la respuesta `y`. |

Otros elementos

| Elemento | Qué es | Qué hace |
|---|---|---|
| `predictors = [...]` | Variable que guarda una lista | Almacena los nombres de las columnas predictoras. |
| `loan3000[predictors]` | Selección de columnas | Extrae esas columnas de la tabla. |
| `X`, `y` | Variables | Por convención, `X` son los predictores e `y` la variable a predecir. |

En la API de scikit-learn el parámetro `eta` se llama `learning_rate`. Usar `eta` no genera error, pero no tiene efecto.

---

### 5. Ejemplo 2: sobreajuste

```python
predictors = ['loan_amnt', 'term', 'annual_inc', 'dti', 'payment_inc_ratio',
              'revol_bal', 'revol_util', 'purpose', 'delinq_2yrs_zero',
              'pub_rec_zero', 'open_acc', 'grade', 'emp_length', 'purpose_',
              'home_', 'emp_len_', 'borrower_score']
outcome = 'outcome'

X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = pd.Series([1 if o == 'default' else 0 for o in loan_data[outcome]])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=10000)

xgb_default = XGBClassifier(objective='binary:logistic', n_estimators=250,
                            max_depth=6, reg_lambda=0, learning_rate=0.3,
                            subsample=1)
xgb_default.fit(train_X, train_y)

pred_default = xgb_default.predict_proba(valid_X)[:, 1]
error_default = abs(valid_y - pred_default) > 0.5
print('default: ', np.mean(error_default))
```

Resultado: el error es 13.3 % en el conjunto de entrenamiento y 35.3 % en el de prueba. La diferencia indica sobreajuste: el modelo explica bien los datos con los que se entrenó, pero sus reglas no se generalizan a datos nuevos.

Funciones

| Nombre | Qué es | Qué hace |
|---|---|---|
| `pd.get_dummies(...)` | Función de pandas | Convierte columnas categóricas en columnas de 0 y 1. Con `drop_first=True` elimina una columna redundante. |
| `pd.Series(...)` | Clase de pandas | Crea una columna de datos a partir de una lista. |
| `train_test_split(...)` | Función de scikit-learn | Divide los datos en entrenamiento y prueba. Con `test_size=10000` reserva 10,000 filas para la prueba. Devuelve predictores y respuesta de cada subconjunto. |
| `abs(...)` | Función de Python | Valor absoluto. |
| `np.mean(...)` | Función de numpy | Promedio. Aplicado a valores True/False (1/0) da la proporción de errores. |
| `print(...)` | Función de Python | Muestra un resultado en pantalla. |

Argumentos de `XGBClassifier`

| Argumento | Qué es | Qué hace |
|---|---|---|
| `n_estimators=250` | Número de árboles | Entrena 250 árboles. |
| `max_depth=6` | Profundidad máxima | Cada árbol tiene como máximo 6 niveles de decisión. |
| `reg_lambda=0` | Regularización L2 | Con 0 no hay penalización. |
| `learning_rate=0.3` | Factor de contracción ($\eta$) | Define el tamaño de cada paso. |
| `subsample=1` | Fracción de filas | Usa todas las filas en cada árbol. |

Métodos

| Nombre | Qué es | Qué hace |
|---|---|---|
| `xgb_default.fit(train_X, train_y)` | Método del modelo | Entrena con los datos de entrenamiento. |
| `xgb_default.predict_proba(valid_X)` | Método del modelo | Devuelve la probabilidad de cada clase para los datos de prueba (una columna por clase). |

Otros elementos

| Elemento | Qué es | Qué hace |
|---|---|---|
| `[:, 1]` | Selección | Toma todas las filas y solo la columna 1, la probabilidad de incumplimiento. |
| `[1 if o == 'default' else 0 for o in ...]` | Lista por comprensión | Recorre la columna y asigna 1 si el valor es `'default'` y 0 en otro caso. |
| `==` | Operador de comparación | Compara dos valores y devuelve True o False. No es lo mismo que `=`, que asigna un valor. |
| `> 0.5` | Operador de comparación | Devuelve True si la diferencia supera 0.5, es decir, si hubo error. |

---

### 6. Ejemplo 3: regularización

```python
xgb_penalty = XGBClassifier(objective='binary:logistic', n_estimators=250,
                            max_depth=6, reg_lambda=1000, learning_rate=0.1,
                            subsample=0.63)
xgb_penalty.fit(train_X, train_y)

pred_penalty = xgb_penalty.predict_proba(valid_X)[:, 1]
error_penalty = abs(valid_y - pred_penalty) > 0.5
print('penalty: ', np.mean(error_penalty))
```

Es el mismo código del ejemplo 2 con tres cambios:

| Argumento | Antes | Ahora | Efecto |
|---|---|---|---|
| `reg_lambda` | 0 | 1000 | Aplica una penalización L2 fuerte a los árboles complejos. |
| `learning_rate` | 0.3 | 0.1 | Reduce el tamaño de cada paso. |
| `subsample` | 1 | 0.63 | Cada árbol usa solo una parte de las filas. |

`reg_alpha` es la penalización L1. En R los parámetros se llaman `lambda` y `alpha`.

Resultado (ejemplo en R de Bruce et al.): el error es 31.0 % en entrenamiento y 32.9 % en prueba. Los dos errores quedan cercanos, lo que indica que el modelo ya no se sobreajusta.

---

### 7. Ejemplo 4: validación cruzada

```python
idx = np.random.choice(range(5), size=len(X), replace=True)
error = []
for eta, max_depth in product([0.1, 0.5, 0.9], [3, 6, 9]):
    xgb = XGBClassifier(objective='binary:logistic', n_estimators=250,
                        max_depth=max_depth, learning_rate=eta)
    cv_error = []
    for k in range(5):
        fold_idx = idx == k
        train_X = X.loc[~fold_idx]; train_y = y[~fold_idx]
        valid_X = X.loc[fold_idx]; valid_y = y[fold_idx]

        xgb.fit(train_X, train_y)
        pred = xgb.predict_proba(valid_X)[:, 1]
        cv_error.append(np.mean(abs(valid_y - pred) > 0.5))
    error.append({
        'eta': eta,
        'max_depth': max_depth,
        'avg_error': np.mean(cv_error)
    })
    print(error[-1])
errors = pd.DataFrame(error)
```

Se evalúan 9 combinaciones de `eta` y `max_depth`. Cada una se entrena 5 veces, una por pliegue, así que se ajustan 45 modelos y el proceso puede tardar.

Funciones

| Nombre | Qué es | Qué hace |
|---|---|---|
| `np.random.choice(range(5), size=len(X), replace=True)` | Función de numpy | Asigna a cada fila, al azar, un número de pliegue entre 0 y 4. |
| `range(5)` | Función de Python | Genera los números 0, 1, 2, 3 y 4. |
| `len(X)` | Función de Python | Devuelve el número de filas de `X`. |
| `product(lista1, lista2)` | Función de itertools | Genera todas las combinaciones entre dos listas. Aquí son 9. |
| `pd.DataFrame(error)` | Clase de pandas | Convierte la lista de resultados en una tabla. |

Métodos

| Nombre | Qué es | Qué hace |
|---|---|---|
| `error.append(...)`, `cv_error.append(...)` | Método de lista | Agrega un elemento al final de la lista. |
| `X.loc[...]` | Método de tabla de pandas | Selecciona filas. |
| `xgb.fit(...)`, `xgb.predict_proba(...)` | Métodos del modelo | Entrenan y predicen, como en los ejemplos anteriores. |

Otros elementos

| Elemento | Qué es | Qué hace |
|---|---|---|
| `for ... in ...:` | Bucle | Repite un bloque de código. El bloque repetido va sangrado. |
| `for k in range(5)` | Bucle anidado | Repite 5 veces con `k` = 0, 1, 2, 3, 4. En cada vuelta el pliegue `k` es el de prueba. |
| `idx == k` | Comparación | Devuelve True para las filas del pliegue `k` y False para las demás. |
| `~fold_idx` | Negación | Invierte True y False. Selecciona las filas que no están en el pliegue de prueba. |
| `;` | Separador | Permite escribir dos instrucciones en la misma línea. |
| `{'eta': eta, ...}` | Diccionario | Guarda pares clave-valor. |
| `error[-1]` | Índice de lista | Último elemento de la lista. |

Resultados (error promedio, %):

| eta | max_depth 3 | max_depth 6 | max_depth 12 |
|---|---|---|---|
| 0.1 | 32.90 | 33.08 | 34.56 |
| 0.5 | 33.43 | 35.60 | 36.83 |
| 0.9 | 34.36 | 37.82 | 38.18 |

La mejor combinación es `eta=0.1` con `max_depth=3` (o posiblemente 6): árboles poco profundos y un factor de contracción pequeño dan resultados más exactos y estables.

Nota: el código en Python de Bruce et al. usa `max_depth` 3, 6 y 9, pero la tabla corresponde al código en R, que usa 3, 6 y 12.

---

### 8. Hiperparámetros de XGBoost

| Parámetro | Qué es | Qué hace |
|---|---|---|
| `eta` / `learning_rate` | Factor de contracción entre 0 y 1 | Controla el tamaño de cada paso. Con datos ruidosos se recomiendan valores pequeños, como 0.1. |
| `nrounds` / `n_estimators` | Número de rondas | Cantidad de árboles. Si `eta` es pequeño, se necesitan más rondas. |
| `max_depth` | Profundidad máxima del árbol | En boosting se usan árboles poco profundos, lo que evita interacciones complejas producidas por el ruido. |
| `subsample` y `colsample_bytree` | Fracción de filas y de columnas | Cada árbol usa solo una parte de los datos, lo que ayuda a evitar el sobreajuste. |
| `lambda` / `reg_lambda` y `alpha` / `reg_alpha` | Parámetros de regularización | Controlan el sobreajuste mediante penalizaciones L2 y L1. |

---

### 9. Conclusiones

- El boosting ajusta una secuencia de modelos y da más peso a las observaciones con errores grandes en las rondas siguientes.
- Stochastic gradient boosting con árboles es la variante más general, y XGBoost la implementa.
- El boosting es propenso al sobreajuste, por lo que sus hiperparámetros deben ajustarse.
- La regularización (L1 y L2) reduce el sobreajuste al penalizar la complejidad del modelo.
- La validación cruzada permite elegir los hiperparámetros, y es importante porque XGBoost tiene muchos.