# Código de Mecánica de Medios Continuos

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24092087.svg)](https://doi.org/10.5281/zenodo.24092087)

Librería y scripts que acompañan al libro de problemas. Solo tienen script los problemas cuyo cálculo va más allá de la calculadora o en los que conviene ver un gráfico.

## Inicio rápido

```bash
git clone https://github.com/jtorreci/mmc-codigo.git
cd mmc-codigo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m problemas.tema_06.p6_06
```

El [tutorial](TUTORIAL.md) explica el uso de la librería paso a paso.

## Contenido

| Carpeta | Contenido |
|---|---|
| `mmc/tensores.py` | Invariantes, tensiones principales, criterios de Von Mises y Tresca |
| `mmc/elasticidad.py` | Constantes elásticas, ley de Hooke, solución de Kirsch |
| `mmc/viscoelasticidad.py` | Modelos de Maxwell, Kelvin y Zener; superposición de Boltzmann |
| `mmc/fluidos.py` | Reynolds, Colebrook-White, Darcy-Weisbach, Hagen-Poiseuille |
| `mmc/graficos.py` | Círculos de Mohr, historias temporales, guardado de figuras |
| `problemas/tema_NN/pN_MM.py` | Script del problema N.M del libro |
| `tests/` | Pruebas de la librería y comprobación de cada script con los valores del libro |

Las figuras generadas se guardan en `salida/`, o en la carpeta que indique la variable de entorno `MMC_SALIDA`.

## Pruebas

```bash
python3 -m unittest discover -s tests -t .
```

## Cómo citar

Usa el DOI [10.5281/zenodo.24092087](https://doi.org/10.5281/zenodo.24092087), que representa todas las versiones y remite siempre a la última.

## Licencia

El código se distribuye bajo la licencia [MIT](LICENSE). Los libros tienen su propia licencia, CC BY-SA 4.0.
