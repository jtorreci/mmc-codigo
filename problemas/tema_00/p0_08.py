#!/usr/bin/env python3
"""Problema 0.8: tensiones y direcciones principales en tres dimensiones.

La ecuación característica de este tensor no tiene raíces sencillas, de modo que el cálculo
numérico da a la vez los autovalores, los autovectores y su ortogonalidad.
"""
import numpy as np

from mmc import tensores

SIGMA = tensores.tensor_simetrico(60.0, 40.0, 80.0, s12=30.0, s13=20.0, s23=10.0)  # MPa

VALORES_LIBRO = {
    "I1": 180.0,
    "I2": 9000.0,
    "I3": 110000.0,
    "lambda_1": 102.97,
    "lambda_2": 58.89,
    "lambda_3": 18.14,
    "n1_1": 0.600,
    "n1_2": 0.396,
    "n1_3": 0.695,
    "n2_1": 0.527,
    "n2_2": 0.458,
    "n2_3": -0.716,
    "n3_1": 0.602,
    "n3_2": -0.796,
    "n3_3": -0.066,
    "sigma_m": 60.0,
    "s_doble_contraido": 3600.0,
    "sigma_VM": 73.5,
}


def orientar(n):
    """Devuelve la dirección con la primera componente positiva, como el libro."""
    return n if n[0] > 0 else -n


def resolver():
    """Devuelve los resultados del problema."""
    i1, i2, i3 = tensores.invariantes(SIGMA)
    valores, vectores = tensores.principales(SIGMA)
    direcciones = np.column_stack([orientar(v) for v in vectores.T])
    s = tensores.desviador(SIGMA)
    resultados = {"I1": i1, "I2": i2, "I3": i3}
    for i, valor in enumerate(valores, start=1):
        resultados[f"lambda_{i}"] = valor
    for i, n in enumerate(direcciones.T, start=1):
        for j, componente in enumerate(n, start=1):
            resultados[f"n{i}_{j}"] = componente
    resultados["error_ortogonalidad"] = np.abs(direcciones.T @ direcciones - np.eye(3)).max()
    resultados["sigma_m"] = i1 / 3
    resultados["s_doble_contraido"] = np.sum(s * s)
    resultados["sigma_VM"] = tensores.von_mises(SIGMA)
    return {clave: float(valor) for clave, valor in resultados.items()}


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")


if __name__ == "__main__":
    main()
