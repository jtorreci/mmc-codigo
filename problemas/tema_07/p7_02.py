#!/usr/bin/env python3
"""Problema 7.2: relajación de un mástique con el modelo de Maxwell.

La curva y su tangente en el origen muestran el significado del tiempo de relajación:
la tangente corta el eje en t = tau_M, mucho antes de que la tensión se extinga.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, viscoelasticidad

EPS0 = 0.02  # deformación impuesta, adimensional
SIGMA0 = 10.0  # tensión inicial del ajuste, MPa
TAU_M = 200.0  # tiempo de relajación del ajuste, s
T_MAX = 1100.0  # fin de la gráfica, s

VALORES_LIBRO = {
    "E": 500.0,
    "tau_M": 200.0,
    "eta_MPas": 1.00e5,
    "sigma_200": 3.68,
    "sigma_1000": 0.0674,
    "t10": 461.0,
    "J_por_E_tau": 0.736,
}


def tension(t):
    """Devuelve la tensión medida, en MPa, en el instante t, en segundos."""
    return viscoelasticidad.relajacion_maxwell(t, SIGMA0, TAU_M)


def resolver():
    """Devuelve los resultados del problema."""
    e = SIGMA0 / EPS0
    eta = e * TAU_M
    # J(t) de Maxwell = 1/E + t/eta; E(t) = E exp(-t/tau_M).
    j_por_e = (1 / e + TAU_M / eta) * e * np.exp(-1)
    return {
        "E": e,
        "tau_M": TAU_M,
        "eta_MPas": eta,
        "sigma_200": float(tension(200.0)),
        "sigma_1000": float(tension(1000.0)),
        "t10": float(TAU_M * np.log(10)),
        "J_por_E_tau": float(j_por_e),
    }


def graficar():
    """Dibuja la historia de tensión, la tangente en el origen y el instante t10."""
    resultados = resolver()
    t = np.linspace(0, T_MAX, 400)
    t_tangente = np.array([0.0, TAU_M])

    fig, ax = graficos.historia_temporal(t, {"Tensión medida": tension(t)}, "$t$ (s)", r"$\sigma$ (MPa)")
    ax.plot(t_tangente, SIGMA0 * (1 - t_tangente / TAU_M), "--", color="tab:red",
            label=r"Tangente en el origen (corta en $t = \tau_M$)")
    ax.plot(resultados["t10"], SIGMA0 / 10, "o", color="tab:orange", markersize=7,
            label=rf"$\sigma_0/10$ en $t_{{10}}$ = {resultados['t10']:.0f} s")
    ax.set_ylim(bottom=0)
    ax.legend()
    ruta = graficos.guardar(fig, "p7_02")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
