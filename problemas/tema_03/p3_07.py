#!/usr/bin/env python3
"""Problema 3.7: material neo-hookeano incompresible en tracción uniaxial.

Las curvas separan la tensión de Cauchy de la nominal y, frente a la recta de la teoría
infinitesimal, muestran que esta deja de valer con estiramientos de este orden.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos

MU = 1.0  # módulo de corte, MPa
LAMBDA_1 = 2.0  # estiramiento axial

VALORES_LIBRO = {
    "lambda_2": 0.707,
    "p": 0.500,
    "sigma_1": 3.50,
    "nominal_1": 1.75,
    "I1": 5.0,
    "W": 1.00,
}


def cauchy(estiramiento):
    """Devuelve la tensión axial de Cauchy en MPa con las caras laterales libres."""
    return MU * (estiramiento**2 - 1 / estiramiento)


def nominal(estiramiento):
    """Devuelve la tensión axial nominal en MPa, referida a la sección inicial."""
    return cauchy(estiramiento) / estiramiento


def resolver():
    """Devuelve los resultados del problema."""
    lambda_2 = 1 / np.sqrt(LAMBDA_1)
    p = MU * lambda_2**2
    i1 = LAMBDA_1**2 + 2 * lambda_2**2
    resultados = {
        "lambda_2": lambda_2,
        "p": p,
        "sigma_1": MU * LAMBDA_1**2 - p,
        "nominal_1": nominal(LAMBDA_1),
        "I1": i1,
        "W": MU / 2 * (i1 - 3),
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja las curvas tensión-estiramiento y el punto del problema."""
    r = resolver()
    estiramientos = np.linspace(1, 3, 201)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(estiramientos, cauchy(estiramientos), color="tab:blue", label=r"Cauchy, $\sigma_1$")
    ax.plot(estiramientos, nominal(estiramientos), color="tab:orange", label=r"Nominal, $\sigma_1/\lambda_1$")
    ax.plot(estiramientos, 3 * MU * (estiramientos - 1), color="0.5", linestyle="--", label=r"Infinitesimal, $3\mu(\lambda_1-1)$")
    ax.plot([LAMBDA_1, LAMBDA_1], [r["sigma_1"], r["nominal_1"]], "o", color="black", markersize=5)
    ax.annotate(f"{r['sigma_1']:.2f} MPa", (LAMBDA_1, r["sigma_1"]), textcoords="offset points", xytext=(-52, 6))
    ax.annotate(f"{r['nominal_1']:.2f} MPa", (LAMBDA_1, r["nominal_1"]), textcoords="offset points", xytext=(6, -14))
    ax.set_xlabel(r"Estiramiento axial $\lambda_1$")
    ax.set_ylabel("Tensión (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ruta = graficos.guardar(fig, "p3_07")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
