#!/usr/bin/env python3
"""Problema 4.14: cilindro de un gato hidráulico con las fórmulas de Lamé.

La gráfica muestra cómo se reparten las tensiones en la pared gruesa: la circunferencial
se concentra en la cara interior y la fórmula de pared delgada no recoge ese máximo.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from mmc import graficos

A = 60.0  # radio interior, mm
B = 90.0  # radio exterior, mm
P = 60.0  # presión del aceite, MPa
E = 210e3  # módulo de elasticidad, MPa
NU = 0.3  # coeficiente de Poisson
SIGMA_ADM = 140.0  # tensión circunferencial admisible, MPa

VALORES_LIBRO = {
    "sigma_rr_a": -60.0,
    "sigma_tt_a": 156.0,
    "sigma_tt_b": 96.0,
    "resultante_N_mm": 3600.0,
    "sigma_tt_media": 120.0,
    "pared_delgada": 150.0,
    "b_min": 94.9,
    "u_a": 0.0497,
}


def tensiones_lame(r):
    """Devuelve (srr, stt) en el radio r, en MPa, con presión exterior nula."""
    coeficiente = P * A**2 / (B**2 - A**2)
    q = (B / r) ** 2
    return coeficiente * (1 - q), coeficiente * (1 + q)


def resolver():
    """Devuelve los resultados del problema."""
    srr_a, stt_a = tensiones_lame(A)
    _, stt_b = tensiones_lame(B)
    r = np.linspace(A, B, 2001)
    resultante = float(simpson(tensiones_lame(r)[1], x=r))
    return {
        "sigma_rr_a": srr_a,
        "sigma_tt_a": stt_a,
        "sigma_tt_b": stt_b,
        "resultante_N_mm": resultante,
        "empuje_N_mm": P * A,
        "sigma_tt_media": resultante / (B - A),
        "pared_delgada": P * (A + B) / 2 / (B - A),
        "b_min": A * np.sqrt((SIGMA_ADM + P) / (SIGMA_ADM - P)),
        "u_a": A * (stt_a - NU * srr_a) / E,
    }


def graficar():
    """Dibuja srr y stt a través de la pared y guarda la figura."""
    datos = resolver()
    r = np.linspace(A, B, 201)
    srr, stt = tensiones_lame(r)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r, stt, color="tab:blue", linewidth=2, label=r"$\sigma_{\theta\theta}$")
    ax.plot(r, srr, color="tab:orange", linewidth=2, label=r"$\sigma_{rr}$")
    ax.plot(r, srr + stt, color="0.4", linestyle=":", label=r"$\sigma_{rr}+\sigma_{\theta\theta}$ (uniforme)")
    ax.axhline(datos["sigma_tt_media"], color="tab:blue", linestyle="--", linewidth=1, label=r"media de $\sigma_{\theta\theta}$")
    ax.axhline(datos["pared_delgada"], color="tab:green", linestyle="--", linewidth=1, label="fórmula de pared delgada")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Radio $r$ (mm)")
    ax.set_ylabel("Tensión (MPa)")
    ax.set_xlim(A, B)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ruta = graficos.guardar(fig, "p4_14")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
