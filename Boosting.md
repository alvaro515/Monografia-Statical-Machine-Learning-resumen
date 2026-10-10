## Boosting

### 1. Introducción

El boosting es una técnica para crear un ensamble de modelos, es decir, un conjunto de modelos cuyas predicciones se combinan. Se usa casi siempre con árboles de decisión. A diferencia del bagging y del bosque aleatorio, donde los modelos se entrenan por separado, en el boosting se entrenan en secuencia: cada modelo nuevo trata de reducir el error del anterior. Por eso requiere más ajuste de parámetros que el bagging.

Existen tres variantes principales:

| Variante | Descripción |
|---|---|
| AdaBoost | Aumenta el peso de las observaciones mal clasificadas para que el siguiente modelo se concentre en ellas. |
| Gradient boosting | Plantea el problema como la minimización de una función de costo. Cada modelo nuevo se ajusta al pseudo-residuo, es decir, a lo que aún no explica el conjunto. |
| Stochastic gradient boosting | Igual que el anterior, pero en cada ronda usa una muestra aleatoria de filas y de columnas. Es la variante más general y la más usada. |


### 2. Comprenderlo matemáticamente

Esta parte explica las fórmulas con palabras y con números sencillos. Lo importante es la idea de fondo: en cada ronda se entrena un modelo, se ve dónde se equivocó y el siguiente modelo se encarga de corregir eso.

#### 2.1 Notación

| Símbolo | Significado |
|---|---|
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

$$\alpha_m = \log\!\left(\frac{1 - e_m}{e_m}\right)$$

Aquí $\log$ es el logaritmo natural, la tecla ln de la calculadora. Lo que importa es cómo se comporta: si el error es chico, $\alpha_m$ sale grande, y si el error es grande, sale chico.

Ejemplo con $e_1 = 0.2$: $\alpha_1 = \log(0.8 / 0.2) = \log(4) \approx 1.39$.

| Error $e_m$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| Confianza $\alpha_m$ | 2.20 | 1.39 | 0.85 | 0.41 | 0 |

Con un error de 0.5 la confianza es 0: el modelo acierta tanto como una moneda al aire, así que no aporta nada.

**Paso 4. Agregar el modelo al ensamble.**

$$F_m = F_{m-1} + \alpha_m f_m$$

Se lee así: el ensamble nuevo es el ensamble anterior más el modelo nuevo, y el modelo nuevo entra multiplicado por su confianza.

**Paso 5. Subir el peso de las filas que falló.** Bruce et al. indican solo que los pesos de las filas mal clasificadas aumentan según $\alpha_m$. La forma habitual de hacerlo es multiplicar el peso de esas filas por $e^{\alpha_m}$ y después reescalar:

$$w_i \leftarrow w_i \cdot e^{\alpha_m} \quad \text{(solo si la fila } i \text{ fue mal clasificada)}$$

Reescalar quiere decir dividir cada peso entre la suma de todos, para que los pesos vuelvan a sumar 1.

Ejemplo: $e^{1.39} \approx 4$, así que el peso de D se multiplica por 4.

| Observación | A | B | C | D mal clasificada | E | Suma |
|---|---|---|---|---|---|---|
| Peso antes | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 1 |
| Peso multiplicado | 0.2 | 0.2 | 0.2 | 0.8 | 0.2 | 1.6 |
| Peso reescalado entre 1.6 | 0.125 | 0.125 | 0.125 | 0.5 | 0.125 | 1 |

Ahora D pesa la mitad de todo, y el segundo modelo se esfuerza sobre todo por acertar D.

**Paso 6. Repetir** hasta tener los $M$ modelos. La predicción final es la suma de todos, cada uno multiplicado por su confianza:

$$F = \alpha_1 f_1 + \alpha_2 f_2 + \cdots + \alpha_M f_M$$

**Ejemplo con dos modelos.** Supongamos que el segundo modelo, entrenado con los pesos nuevos, solo falla en A:

| | A | B | C | D | E | Error | Confianza |
|---|---|---|---|---|---|---|---|
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
|---|---|---|---|
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
|---|---|---|
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
|---|---|---|---|---|
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

Librerías que se usan en los ejemplos:

```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from itertools import product
```

- **numpy**, abreviada np, sirve para hacer cálculos con números, como promedios o números al azar.
- **pandas**, abreviada pd, sirve para trabajar con tablas de datos.
- **xgboost** trae el modelo de boosting. De ahí se importa la clase XGBClassifier.
- **scikit-learn**, abreviada sklearn, ayuda a separar los datos en una parte para entrenar y otra para probar, con train_test_split.
- **itertools** permite armar todas las combinaciones posibles de valores con product, algo útil cuando se prueban varios hiperparámetros.

#### 3.1 Ejemplo

Queremos detectar conexiones sospechosas en una red. De cada una se sabe cuántos intentos hace por minuto y a cuántos puertos distintos intenta entrar: una conexión normal usa uno o dos puertos, y quien escanea la red toca decenas. Hay 16 conexiones ya revisadas:

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

XGBClassifier crea el modelo vacío. objective='binary:logistic' indica que hay dos respuestas posibles, n_estimators=50 que se construyen 50 árboles en secuencia (sección 2), max_depth=2 que cada árbol es pequeño y learning_rate=0.3 es el $\eta$ de las fórmulas. fit entrena el modelo.

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
|---|---|---|---|
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
|---|---|
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