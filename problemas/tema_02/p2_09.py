#!/usr/bin/env python3
"""Problema 2.9: descomposición polar de la deformación de una lámina.

Separa el giro R del estiramiento U construyendo U en la base propia de C, tal como hace el
capítulo; los productos matriciales son largos a mano y aquí se comprueban de una vez.
"""
import numpy as np

from mmc import tensores

F = np.array([[1.1, 0.3, 0.0], [-0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])  # gradiente de deformación

# El libro imprime V12 con solo dos cifras significativas (0.078).
TOLERANCIA = 7e-3

VALORES_LIBRO = {
    "J": 1.02,
    "lambda_1": 1.161,
    "lambda_2": 0.878,
    "alfa_N1_grados": 28.2,
    "theta_grados": -11.3,
    "U11": 1.098,
    "U12": 0.118,
    "U22": 0.941,
    "V11": 1.137,
    "V12": 0.078,
    "V22": 0.902,
    "alfa_n1_grados": 16.8,
}


def angulo_plano(n):
    """Devuelve el ángulo en grados de una dirección del plano, entre -90 y 90."""
    return np.degrees(np.arctan2(n[1], n[0])) if n[0] > 0 else np.degrees(np.arctan2(-n[1], -n[0]))


def descomposicion_polar():
    """Devuelve (R, U, V, estiramientos, direcciones) a partir de C = F^T F."""
    cuadrados, N = tensores.principales(F.T @ F)
    estiramientos = np.sqrt(cuadrados)
    U = N @ np.diag(estiramientos) @ N.T
    R = F @ np.linalg.inv(U)
    return R, U, R @ U @ R.T, estiramientos, N


def resolver():
    """Devuelve los resultados del problema."""
    R, U, V, estiramientos, N = descomposicion_polar()
    # La dirección 3 no se deforma: su autovalor 1 queda entre los dos del plano.
    N1 = N[:, 0]
    resultados = {
        "J": np.linalg.det(F),
        "lambda_1": estiramientos[0],
        "lambda_2": estiramientos[2],
        "alfa_N1_grados": angulo_plano(N1),
        "theta_grados": np.degrees(np.arctan2(R[1, 0], R[0, 0])),
        "U11": U[0, 0],
        "U12": U[0, 1],
        "U22": U[1, 1],
        "V11": V[0, 0],
        "V12": V[0, 1],
        "V22": V[1, 1],
        "alfa_n1_grados": angulo_plano(R @ N1),
        "error_RU_F": np.abs(R @ U - F).max(),
        "error_U2_C": np.abs(U @ U - F.T @ F).max(),
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")


if __name__ == "__main__":
    main()
