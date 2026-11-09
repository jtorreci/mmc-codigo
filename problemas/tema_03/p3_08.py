#!/usr/bin/env python3
"""Problema 3.8: relajación de tensiones en un modelo de Maxwell.

La curva sitúa los instantes pedidos sobre la exponencial y muestra que la tensión se relaja
por completo, propio de un fluido viscoelástico.
"""
import numpy as np

from mmc import graficos, viscoelasticidad

E = 100.0  # módulo, MPa
ETA = 1000.0  # viscosidad, MPa·s
EPS_0 = 0.01  # deformación súbita mantenida
FRACCION = 0.10  # fracción de la tensión inicial en el apartado c

VALORES_LIBRO = {"tau": 10.0, "sigma_0": 1.00, "sigma_tau": 0.368, "t_10": 23.0}


def resolver():
    """Devuelve los resultados del problema."""
    tau = ETA / E
    sigma_0 = E * EPS_0
    resultados = {
        "tau": tau,
        "sigma_0": sigma_0,
        "sigma_tau": viscoelasticidad.relajacion_maxwell(tau, sigma_0, tau),
        "t_10": tau * np.log(1 / FRACCION),
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja la relajación con los instantes del problema."""
    r = resolver()
    t = np.linspace(0, 5 * r["tau"], 301)
    sigma = viscoelasticidad.relajacion_maxwell(t, r["sigma_0"], r["tau"])
    fig, ax = graficos.historia_temporal(t, {r"$\sigma(t)$": sigma}, "t (s)", r"$\sigma$ (MPa)")
    puntos = {
        r"$t = 0^+$": (0.0, r["sigma_0"]),
        r"$t = \tau$": (r["tau"], r["sigma_tau"]),
        r"$10\,\%$ de $\sigma_0$": (r["t_10"], FRACCION * r["sigma_0"]),
    }
    for nombre, (x, y) in puntos.items():
        ax.plot(x, y, "o", color="black", markersize=5)
        ax.annotate(f"{nombre}: {y:.3g} MPa", (x, y), textcoords="offset points", xytext=(10, -2))
    ax.axhline(0, color="0.4", linewidth=0.8)
    ruta = graficos.guardar(fig, "p3_08")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
