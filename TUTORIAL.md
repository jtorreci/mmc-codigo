# Tutorial de la librería `mmc`

`mmc` reúne las operaciones que se repiten en los problemas del libro. En diez minutos podrás calcular un estado tensional, evaluar un criterio de fluencia y dibujar el resultado.

## 1. Preparar el entorno

Descarga el repositorio y abre una terminal en su carpeta principal:

```bash
python3 -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -e .
```

Solo hace falta una vez. La opción `-e` instala la librería de modo que cualquier cambio en `mmc/` se aplica sin reinstalar.

## 2. Unidades y signos

Las funciones no fijan unidades: si entran MPa, salen MPa. La tracción es positiva, salvo en los problemas de geotecnia que indican lo contrario.

## 3. Estado tensional

Tomamos el tensor del problema 1.18, en MPa:

```python
from mmc import tensores

S = tensores.tensor_simetrico(150, 100, 50, s12=40, s13=30, s23=20)

I1, I2, I3 = tensores.invariantes(S)     # 300, 24600, 568000
valores, direcciones = tensores.principales(S)
print(valores)                           # [181.96  78.04  40.  ]
print(direcciones[:, 0])                 # dirección de la mayor, en columna
```

`principales` ordena de mayor a menor y devuelve las direcciones en las columnas de la matriz. El signo de cada dirección es arbitrario: la solución `-n` es igual de válida.

La tensión sobre un plano se obtiene con su normal, que no hace falta normalizar:

```python
t, sigma_n, tau = tensores.componentes_intrinsecas(S, [1, 1, 1])
print(sigma_n, tau)                      # 160.0  48.99
```

## 4. Criterios de fluencia

`von_mises` y `tresca` aceptan el tensor completo o las tres tensiones principales:

```python
print(tensores.von_mises(S))             # 127.3
print(tensores.tresca(S))                # 142.0
print(tensores.von_mises([200, 100, 0])) # 173.2
```

## 5. Elasticidad

```python
import numpy as np
from mmc import elasticidad

c = elasticidad.constantes_elasticas(200e3, 0.3)
print(c.G, c.K, c.lam)                   # 76923  166667  115385

eps = tensores.tensor_simetrico(1e-3, -3e-4, -3e-4)
sigma = elasticidad.hooke(eps, 200e3, 0.3)
```

Muchas funciones admiten vectores, de modo que un perfil completo se calcula sin bucles. Aquí, la tensión circunferencial en la sección transversal a una placa con taladro de radio 1, a tracción de 100 MPa:

```python
radios = np.array([1.0, 2.0, 4.0])
_, stt, _ = elasticidad.kirsch(100.0, radios, np.pi / 2, a=1.0)
print(stt)                               # [300.  121.9  103.7]
```

## 6. Viscoelasticidad

La superposición de Boltzmann recibe la función de fluencia y la historia de saltos de carga. Modelo de Zener del problema 7.5, con 5 MPa aplicados en `t = 0` y retirados en `t = 300 s`:

```python
from mmc import viscoelasticidad as visco

p = visco.parametros_zener(500, 800, 8e4)


def fluencia(t):
    return visco.fluencia_zener(t, p.E0, p.Einf, p.tau_s)


t = np.linspace(0, 600, 601)
deformacion = visco.boltzmann(fluencia, t, [0.0, 300.0], [5.0, -5.0])
print(deformacion[-1])                   # 2.96e-04
```

En el instante exacto de un salto, `boltzmann` devuelve el valor posterior al salto.

## 7. Flujo en tuberías

La ecuación de Colebrook-White es implícita; `colebrook` la resuelve por iteración. Sin rugosidad, da el tubo liso:

```python
from mmc import fluidos

u = fluidos.velocidad_media(0.02, 0.1)
re = fluidos.reynolds(u, 0.1, 1e-6)
f = fluidos.colebrook(re, 0.045 / 100)
print(f)                                 # 0.01816
print(fluidos.darcy_weisbach(f, 500, 0.1, 1000, u) / 1e3)  # 294 kPa
```

## 8. Gráficos

Cada función devuelve `(fig, ax)`, de modo que puedes añadir lo que quieras antes de guardar:

```python
from mmc import graficos

fig, ax = graficos.circulos_mohr(valores, unidad="MPa")
ax.plot(sigma_n, tau, "ko")              # el plano de normal (1, 1, 1)
print(graficos.guardar(fig, "mohr"))     # salida/mohr.png

curvas = {"Zener": deformacion}
fig, ax = graficos.historia_temporal(t, curvas, "t (s)", "deformación")
graficos.guardar(fig, "historia")
```

Las figuras se guardan en `salida/`.

## 9. Los scripts de los problemas

Cada script de `problemas/` resuelve un problema del libro. Se ejecuta desde la carpeta principal del repositorio:

```bash
python3 -m problemas.tema_06.p6_06
```

Todos siguen el mismo esquema:

| Elemento | Para qué sirve |
|---|---|
| Constantes al principio | Los datos del enunciado, con sus unidades |
| `VALORES_LIBRO` | Los resultados publicados, usados como comprobación |
| `resolver()` | El cálculo; devuelve un diccionario |
| `graficar()` | Las figuras, si el problema las tiene |

La mejor forma de aprender es cambiar un dato y volver a ejecutar. ¿Cómo varía la pérdida de carga si duplicas el diámetro? ¿Y si la tubería es lisa?

## 10. Comprobar que todo funciona

```bash
python3 -m unittest discover -s tests -t .
```
