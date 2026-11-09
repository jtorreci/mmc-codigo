#!/usr/bin/env python3
"""Problema 1.18: análisis completo de un estado tensional tridimensional.

Reúne en un solo cálculo el problema de autovalores completo y todas las magnitudes del tema,
que a mano exigirían factorizar la ecuación característica y resolver tres sistemas.
"""
import numpy as np

from mmc import tensores

SIGMA = tensores.tensor_simetrico(150.0, 100.0, 50.0, s12=40.0, s13=30.0, s23=20.0)  # MPa

# El libro imprime n2_3 con solo dos cifras significativas (-0.070).
TOLERANCIA = 7e-3

VALORES_LIBRO = {
    "I1": 300.0,
    "I2": 24600.0,
    "I3": 568000.0,
    "sigma_1": 181.96,
    "sigma_2": 78.04,
    "sigma_3": 40.0,
    "n1_1": 0.840,
    "n1_2": 0.474,
    "n1_3": 0.263,
    "n2_1": 0.507,
    "n2_2": -0.859,
    "n2_3": -0.070,
    "n3_1": 0.192,
    "n3_2": 0.192,
    "n3_3": -0.962,
    "sigma_m": 100.0,
    "J2": 5400.0,
    "J3": 108000.0,
    "tau_oct": 60.0,
    "xi": 173.2,
    "rho": 103.9,
    "tau_max": 70.98,
    "sigma_n": 110.98,
    "normal_1": 0.730,
    "normal_2": 0.471,
    "normal_3": -0.495,
}


def orientar(n):
    """Devuelve la dirección con la primera componente positiva, como el libro."""
    return n if n[0] > 0 else -n


def resolver():
    """Devuelve los resultados del problema."""
    i1, i2, i3 = tensores.invariantes(SIGMA)
    valores, vectores = tensores.principales(SIGMA)
    direcciones = np.column_stack([orientar(v) for v in vectores.T])
    j2, j3 = tensores.invariantes_desviador(SIGMA)
    xi, rho, lode = tensores.haigh_westergaard(SIGMA)
    t_max, sigma_n = tensores.tau_max(SIGMA)
    # Normal de un plano de tangencial máxima: bisectriz de n1 y n3.
    normal = (direcciones[:, 0] + direcciones[:, 2]) / np.sqrt(2)
    resultados = {"I1": i1, "I2": i2, "I3": i3}
    for i, valor in enumerate(valores, start=1):
        resultados[f"sigma_{i}"] = valor
    for i, n in enumerate(direcciones.T, start=1):
        for j, componente in enumerate(n, start=1):
            resultados[f"n{i}_{j}"] = componente
    resultados["error_ortogonalidad"] = np.abs(direcciones.T @ direcciones - np.eye(3)).max()
    resultados.update({
        "sigma_m": i1 / 3,
        "J2": j2,
        "J3": j3,
        "tau_oct": tensores.octaedricas(SIGMA)[1],
        "xi": xi,
        "rho": rho,
        "theta_Lode_grados": np.degrees(lode),
        "tau_max": t_max,
        "sigma_n": sigma_n,
        "normal_1": normal[0],
        "normal_2": normal[1],
        "normal_3": normal[2],
    })
    return {clave: float(valor) for clave, valor in resultados.items()}


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")


if __name__ == "__main__":
    main()
