#!/usr/bin/env python3
"""Problema 5.7: parámetros de Mohr-Coulomb por regresión de ensayos triaxiales.

Convenio de este problema: compresión positiva, el habitual en geotecnia. Las gráficas
comprueban el ajuste en el plano (sigma_3, sigma_1) y la envolvente tangente a los círculos.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos

SIGMA_3 = np.array([50.0, 100.0, 200.0])  # presión de confinamiento, kPa
SIGMA_1 = np.array([150.0, 250.0, 450.0])  # tensión principal mayor en rotura, kPa

VALORES_LIBRO = {"A": 2.00, "B_kPa": 50.0, "phi_grados": 19.5, "c_kPa": 17.7, "R2": 1.00}


def resolver():
    """Devuelve los resultados del problema."""
    a, b = np.polyfit(SIGMA_3, SIGMA_1, 1)
    residuos = SIGMA_1 - (a * SIGMA_3 + b)
    r2 = 1 - np.sum(residuos**2) / np.sum((SIGMA_1 - SIGMA_1.mean()) ** 2)
    phi = np.arcsin((a - 1) / (a + 1))
    return {
        "A": float(a),
        "B_kPa": float(b),
        "phi_grados": float(np.degrees(phi)),
        "c_kPa": float(b / (2 * np.sqrt(a))),
        "R2": float(r2),
    }


def graficar():
    """Dibuja la regresión y los círculos de rotura con su envolvente."""
    datos = resolver()
    a, b, c = datos["A"], datos["B_kPa"], datos["c_kPa"]
    phi = np.radians(datos["phi_grados"])
    fig, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.linspace(0, 220, 50)
    ax_r.plot(x, a * x + b, color="tab:blue", linewidth=2, label=rf"$\sigma_1 = {a:.2f}\,\sigma_3 + {b:.1f}$")
    ax_r.plot(SIGMA_3, SIGMA_1, "o", color="tab:red", markersize=8, label="Ensayos")
    ax_r.set_xlabel(r"$\sigma_3$ (kPa)")
    ax_r.set_ylabel(r"$\sigma_1$ en rotura (kPa)")
    ax_r.set_xlim(0, 220)
    ax_r.set_ylim(bottom=0)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend()

    for s1, s3 in zip(SIGMA_1, SIGMA_3):
        graficos.circulos_mohr([s1, s3, s3], "kPa", ax=ax_m)
    centros, radios = (SIGMA_1 + SIGMA_3) / 2, (SIGMA_1 - SIGMA_3) / 2
    ax_m.plot(centros - radios * np.sin(phi), radios * np.cos(phi), "o", color="tab:red", markersize=6, label="Puntos de tangencia")
    sigma = np.linspace(0, 480, 2)
    ax_m.plot(sigma, c + sigma * np.tan(phi), color="tab:orange", linewidth=2, label=rf"Envolvente: $c = {c:.1f}$ kPa, $\phi = {datos['phi_grados']:.1f}^\circ$")
    ax_m.set_xlim(0, 480)
    ax_m.set_ylim(0, 205)
    ax_m.set_xlabel(r"$\sigma_n$ (kPa), compresión positiva")
    ax_m.legend(loc="upper left", fontsize=8)

    ruta = graficos.guardar(fig, "p5_07")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
