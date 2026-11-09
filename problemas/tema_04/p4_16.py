#!/usr/bin/env python3
"""Problema 4.16: losa de pavimento calentada por el sol.

Descompone por integración numérica el perfil de temperatura en media, parte lineal y
parte no lineal. Las gráficas muestran qué parte produce cada tensión en el espesor.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from mmc import graficos

H = 0.25  # espesor de la losa, m
DT0 = 15.0  # incremento de temperatura en la cara superior, K
E = 33e3  # módulo de elasticidad, MPa
NU = 0.2  # coeficiente de Poisson
ALFA = 1e-5  # coeficiente de dilatación térmica, 1/K
FCTM = 2.9  # resistencia media a tracción, MPa
N_PUNTOS = 2001  # impar, para que el plano medio sea un punto de la malla

VALORES_LIBRO = {
    "dT_media_K": 5.00,
    "sigma_caras_MPa": -1.03,
    "sigma_plano_medio_MPa": 0.516,
    "kappa_1_m": 6.00e-4,
    "sigma_coaccion_MPa": 3.09,
    "sigma_inf_MPa": 2.06,
    "sigma_sup_MPa": -4.13,
    "fraccion_fctm": 0.71,
}


def perfiles():
    """Devuelve la cota x3 y las partes del perfil: (x3, total, media, lineal, no lineal)."""
    x3 = np.linspace(-H / 2, H / 2, N_PUNTOS)
    total = DT0 * (0.5 + x3 / H) ** 2
    media = simpson(total, x=x3) / H
    pendiente = 12 / H**3 * simpson(total * x3, x=x3)
    lineal = pendiente * x3
    return x3, total, media, lineal, total - media - lineal


def tensiones():
    """Devuelve (x3, autoequilibrada, coaccion, total) en MPa."""
    x3, _, _, lineal, no_lineal = perfiles()
    factor = E * ALFA / (1 - NU)
    autoequilibrada = -factor * no_lineal
    coaccion = -factor * lineal
    return x3, autoequilibrada, coaccion, autoequilibrada + coaccion


def resolver():
    """Devuelve los resultados del problema."""
    x3, _, media, lineal, _ = perfiles()
    _, autoequilibrada, coaccion, total = tensiones()
    pendiente = (lineal[-1] - lineal[0]) / H
    medio = N_PUNTOS // 2
    return {
        "dT_media_K": float(media),
        "sigma_caras_MPa": float(autoequilibrada[-1]),
        "sigma_plano_medio_MPa": float(autoequilibrada[medio]),
        "axil_libre_MN_m": float(simpson(autoequilibrada, x=x3)),
        "momento_libre_MN": float(simpson(autoequilibrada * x3, x=x3)),
        "kappa_1_m": float(ALFA * pendiente),
        "sigma_coaccion_MPa": float(coaccion[0]),
        "sigma_inf_MPa": float(total[0]),
        "sigma_sup_MPa": float(total[-1]),
        "fraccion_fctm": float(total[0] / FCTM),
    }


def graficar():
    """Dibuja la descomposición de la temperatura y las tensiones en el espesor."""
    x3, total_t, media, lineal, no_lineal = perfiles()
    _, autoequilibrada, coaccion, total_s = tensiones()
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    ax_t.plot(total_t, x3, color="0.15", linewidth=2.5, label=r"$\Delta T$ total")
    ax_t.axvline(media, color="tab:blue", linewidth=2, label="Media")
    ax_t.plot(lineal, x3, color="tab:orange", linewidth=2, label="Parte lineal")
    ax_t.plot(no_lineal, x3, color="tab:green", linewidth=2, label="Parte no lineal")
    ax_t.set_xlabel("Incremento de temperatura (K)")
    ax_t.set_ylabel("Cota $x_3$ (m)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax_s.plot(autoequilibrada, x3, color="tab:green", linewidth=2, label="Autoequilibrada (losa libre)")
    ax_s.plot(coaccion, x3, color="tab:orange", linewidth=2, label="Por curvatura impedida")
    ax_s.plot(total_s, x3, color="0.15", linewidth=2.5, label="Total con curvatura impedida")
    ax_s.axvline(FCTM, color="tab:red", linestyle="--", linewidth=1, label=r"$f_{ctm}$")
    ax_s.axvline(0, color="0.4", linewidth=0.8)
    ax_s.set_xlabel("Tensión $\\sigma$ en el plano de la losa (MPa)")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ruta = graficos.guardar(fig, "p4_16")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:24s} {valor:.5g}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
