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

Dos cosas de notación que aparecen todo el tiempo:

- **Subíndices.** $w_3$ se lee "w sub 3" y es el peso de la fila 3. La letra de abajo solo indica de cuál fila o de cuál modelo se habla.
- **Sumatoria.** El símbolo $\sum$ significa "sumar todos". Por ejemplo, $\sum_{i=1}^{3} w_i = w_1 + w_2 + w_3$. Si cada peso vale 0.2, el resultado es $0.2 + 0.2 + 0.2 = 0.6$.

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

**¿Qué es el pseudo-residuo?** Es la versión general del residuo, para cuando el costo no es el error cuadrático. Se calcula con la derivada de la función de costo $L$:

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

```python
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from itertools import product
```

Se usan cinco librerías:

- **numpy**, abreviada np: cálculos con números.
- **pandas**, abreviada pd: tablas de datos.
- **xgboost**: el modelo de boosting, con la clase XGBClassifier.
- **scikit-learn**, abreviada sklearn: dividir los datos en entrenamiento y prueba, con train_test_split.
- **itertools**: armar combinaciones, con product.

Los nombres de las variables están en español. Los de las librerías y los de las columnas de los datos no se pueden traducir.

---

### 4. Ejemplo 1: XGBoost

```python
predictores = ['borrower_score', 'payment_inc_ratio']
resultado = 'outcome'

X = prestamos_3000[predictores]
y = prestamos_3000[resultado]

modelo = XGBClassifier(objective='binary:logistic', subsample=0.63)
modelo.fit(X, y)
```

prestamos_3000 es una tabla de 3000 préstamos. En el libro se llama loan3000. borrower_score es el puntaje del cliente, payment_inc_ratio es la cuota respecto al ingreso y outcome indica si el préstamo se pagó.

Qué hace: toma dos columnas de los préstamos, el puntaje del cliente y la cuota respecto al ingreso, crea un modelo XGBoost con XGBClassifier y lo entrena con fit para que aprenda a predecir si un préstamo se paga o no. El argumento subsample=0.63 hace que cada árbol use solo el 63 % de las filas.

En Python, eta se llama learning_rate. Usar eta no da error, pero no tiene efecto.

---

### 5. Ejemplo 2: sobreajuste

```python
predictores = ['loan_amnt', 'term', 'annual_inc', 'dti', 'payment_inc_ratio',
               'revol_bal', 'revol_util', 'purpose', 'delinq_2yrs_zero',
               'pub_rec_zero', 'open_acc', 'grade', 'emp_length', 'purpose_',
               'home_', 'emp_len_', 'borrower_score']
resultado = 'outcome'

X = pd.get_dummies(datos_prestamos[predictores], drop_first=True)
y = pd.Series([1 if o == 'default' else 0 for o in datos_prestamos[resultado]])

X_entren, X_prueba, y_entren, y_prueba = train_test_split(X, y, test_size=10000)

modelo_defecto = XGBClassifier(objective='binary:logistic', n_estimators=250,
                               max_depth=6, reg_lambda=0, learning_rate=0.3,
                               subsample=1)
modelo_defecto.fit(X_entren, y_entren)

pred_defecto = modelo_defecto.predict_proba(X_prueba)[:, 1]
error_defecto = abs(y_prueba - pred_defecto) > 0.5
print('defecto: ', np.mean(error_defecto))
```

datos_prestamos es la tabla completa de préstamos. En el libro se llama loan_data.

Qué hace: usa 17 columnas de los préstamos y convierte el texto a números con get_dummies. Separa 10,000 préstamos para probar con train_test_split y entrena con el resto un modelo XGBoost con la configuración por defecto, usando XGBClassifier y fit. Después calcula con predict_proba la probabilidad de incumplimiento de los préstamos de prueba y mide en qué porcentaje se equivoca.

Resultado: 13.3 % de error en entrenamiento y 35.3 % en prueba. Esa diferencia indica sobreajuste: el modelo funciona bien con los datos que ya vio, pero no con datos nuevos.

---

### 6. Ejemplo 3: regularización

```python
modelo_penalizado = XGBClassifier(objective='binary:logistic', n_estimators=250,
                                  max_depth=6, reg_lambda=1000, learning_rate=0.1,
                                  subsample=0.63)
modelo_penalizado.fit(X_entren, y_entren)

pred_penalizado = modelo_penalizado.predict_proba(X_prueba)[:, 1]
error_penalizado = abs(y_prueba - pred_penalizado) > 0.5
print('penalizado: ', np.mean(error_penalizado))
```

Qué hace: lo mismo que el ejemplo 2, pero con regularización. Solo cambian tres argumentos de XGBClassifier:

| | | | |
|---|---|---|---|
| Argumento | Antes | Ahora | Efecto |
| reg_lambda | 0 | 1000 | Aplica una penalización L2 fuerte a los árboles complejos. |
| learning_rate | 0.3 | 0.1 | Reduce el tamaño de cada paso. |
| subsample | 1 | 0.63 | Cada árbol usa solo una parte de las filas. |

reg_alpha es el argumento de la penalización L1. En R los parámetros se llaman lambda y alpha.

Resultado del ejemplo en R de Bruce et al.: 31.0 % de error en entrenamiento y 32.9 % en prueba. Los dos errores quedan cercanos: el modelo ya no se sobreajusta.

---

### 7. Ejemplo 4: validación cruzada

```python
grupos = np.random.choice(range(5), size=len(X), replace=True)
errores = []
for eta, profundidad in product([0.1, 0.5, 0.9], [3, 6, 9]):
    modelo = XGBClassifier(objective='binary:logistic', n_estimators=250,
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
    print(errores[-1])
tabla_errores = pd.DataFrame(errores)
```

Qué hace: reparte los datos al azar en 5 grupos con np.random.choice. Para cada una de las 9 combinaciones de eta y profundidad, que arma product, entrena el modelo 5 veces, dejando fuera un grupo distinto cada vez para probarlo. Con eso obtiene el error promedio de cada combinación y al final los junta en una tabla con pd.DataFrame para compararlos.

Son 9 combinaciones por 5 grupos, es decir, 45 modelos, así que puede tardar.

Resultados, error promedio en porcentaje:

| | | | |
|---|---|---|---|
| eta | max_depth 3 | max_depth 6 | max_depth 12 |
| 0.1 | 32.90 | 33.08 | 34.56 |
| 0.5 | 33.43 | 35.60 | 36.83 |
| 0.9 | 34.36 | 37.82 | 38.18 |

La mejor combinación es eta=0.1 con max_depth=3, o posiblemente 6: árboles poco profundos y un factor de contracción pequeño dan resultados más exactos y estables.

Nota: el código en Python de Bruce et al. usa max_depth 3, 6 y 9, pero la tabla corresponde al código en R, que usa 3, 6 y 12.

---

### 8. Hiperparámetros de XGBoost

| | |
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