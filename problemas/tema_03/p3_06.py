#!/usr/bin/env python3
"""Problema 3.6: tensor de tensiones de Hooke y estado tensional tridimensional.

Aplica la ley de Hooke generalizada y sitúa el resultado en los tres círculos de Mohr, donde
se ve que todas las tensiones principales son de tracción aunque una deformación sea negativa.
"""
import numpy as np

from mmc import elasticidad, graficos, tensores

E = 200e3  # módulo de Young, MPa
NU = 0.3  # coeficiente de Poisson
EPSILON = tensores.tensor_simetrico(0.001, -0.0003, 0.0002, s12=0.0005)  # adimensional

VALORES_LIBRO = {
    "G_GPa": 76.92,
    "lambda_GPa": 115.38,
    "K_GPa": 166.67,
    "s11": 257.7,
    "s22": 57.7,
    "s33": 134.6,
    "s12": 76.9,
    "sigma_1": 284.0,
    "sigma_2": 135.0,
    "sigma_3": 31.5,
    "traza_sigma": 450.0,
    "tres_K_traza_eps": 450.0,
}


def resolver():
    """Devuelve los resultados del problema."""
    c = elasticidad.constantes_elasticas(E, NU)
    sigma = elasticidad.hooke(EPSILON, E, NU)
    valores, _ = tensores.principales(sigma)
    resultados = {
        "G_GPa": c.G / 1e3,
        "lambda_GPa": c.lam / 1e3,
        "K_GPa": c.K / 1e3,
        "s11": sigma[0, 0],
        "s22": sigma[1, 1],
        "s33": sigma[2, 2],
        "s12": sigma[0, 1],
        "sigma_1": valores[0],
        "sigma_2": valores[1],
        "sigma_3": valores[2],
        "traza_sigma": np.trace(sigma),
        "tres_K_traza_eps": 3 * c.K * np.trace(EPSILON),
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja los círculos de Mohr con el diámetro de las facetas del plano 1-2."""
    r = resolver()
    fig, ax = graficos.circulos_mohr([r["sigma_1"], r["sigma_2"], r["sigma_3"]], unidad="MPa")
    ax.plot([r["s11"], r["s22"]], [r["s12"], -r["s12"]], "o-", color="tab:orange", markersize=5, linewidth=1)
    ax.annotate("Cara 1", (r["s11"], r["s12"]), textcoords="offset points", xytext=(-46, 6))
    ax.annotate("Cara 2", (r["s22"], -r["s12"]), textcoords="offset points", xytext=(6, -14))
    ruta = graficos.guardar(fig, "p3_06")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
