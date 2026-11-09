#!/usr/bin/env python3
"""Problema 1.22: concentración de tensiones en el borde de un agujero (Kirsch).

Las curvas muestran que la concentración es local: el borde pasa de compresión a tracción
triple según el ángulo, y a un diámetro del borde la perturbación casi ha desaparecido.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from mmc import elasticidad, graficos, tensores

A = 10.0  # radio del agujero, mm
SIGMA_INF = 100.0  # tracción lejana según x, MPa

VALORES_LIBRO = {
    "sigma_A": -100.0,
    "sigma_B": 300.0,
    "Kt": 3.0,
    "tau_max_A": 50.0,
    "tau_max_B": 150.0,
    "sigma_2a": 121.9,
    "sigma_3a": 107.4,
    "exceso_N_mm": 1000.0,
}


def sigma_tt(r, theta):
    """Devuelve la tensión circunferencial en (r, theta), con theta en radianes."""
    return elasticidad.kirsch(SIGMA_INF, r, theta, a=A)[1]


def resolver():
    """Devuelve los resultados del problema."""
    sigma_a = sigma_tt(A, 0.0)
    sigma_b = sigma_tt(A, np.pi / 2)
    # En el borde el estado es uniaxial: la única principal no nula es la circunferencial.
    exceso, _ = quad(lambda r: float(sigma_tt(r, np.pi / 2)) - SIGMA_INF, A, np.inf)
    resultados = {
        "sigma_A": sigma_a,
        "sigma_B": sigma_b,
        "Kt": sigma_b / SIGMA_INF,
        "tau_max_A": tensores.tau_max((sigma_a, 0.0, 0.0))[0],
        "tau_max_B": tensores.tau_max((sigma_b, 0.0, 0.0))[0],
        "sigma_2a": sigma_tt(2 * A, np.pi / 2),
        "sigma_3a": sigma_tt(3 * A, np.pi / 2),
        "exceso_N_mm": exceso,
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja la tensión circunferencial en el borde y su atenuación con r."""
    r = resolver()
    fig, (izquierda, derecha) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    theta = np.linspace(0, 180, 361)
    izquierda.plot(theta, sigma_tt(A, np.radians(theta)), color="tab:blue", label=r"$\sigma_{\theta\theta}(a,\theta)$")
    izquierda.plot([0, 90], [r["sigma_A"], r["sigma_B"]], "o", color="black", markersize=5)
    izquierda.annotate("A", (0, r["sigma_A"]), textcoords="offset points", xytext=(6, 4))
    izquierda.annotate("B", (90, r["sigma_B"]), textcoords="offset points", xytext=(6, -12))
    izquierda.axhline(0, color="0.4", linewidth=0.8)
    izquierda.set_xlabel(r"$\theta$ (grados)")
    izquierda.set_ylabel(r"$\sigma_{\theta\theta}$ (MPa)")
    izquierda.set_title("A lo largo del borde")
    izquierda.legend()
    izquierda.grid(True, alpha=0.3)

    radios = np.linspace(1, 6, 201)
    derecha.plot(radios, sigma_tt(radios * A, np.pi / 2), color="tab:blue", label=r"$\sigma_{\theta\theta}(r,90^\circ)$")
    derecha.axhline(SIGMA_INF, color="tab:orange", linestyle="--", label=r"$\sigma_\infty$")
    derecha.plot([2, 3], [r["sigma_2a"], r["sigma_3a"]], "o", color="black", markersize=5)
    derecha.annotate(f"{r['sigma_2a']:.1f} MPa", (2, r["sigma_2a"]), textcoords="offset points", xytext=(6, 16))
    derecha.annotate(f"{r['sigma_3a']:.1f} MPa", (3, r["sigma_3a"]), textcoords="offset points", xytext=(6, 6))
    derecha.set_xlabel(r"$r/a$")
    derecha.set_title(r"Atenuación en $\theta = 90^\circ$")
    derecha.legend()
    derecha.grid(True, alpha=0.3)

    ruta = graficos.guardar(fig, "p1_22")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
