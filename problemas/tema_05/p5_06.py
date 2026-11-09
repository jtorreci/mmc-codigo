#!/usr/bin/env python3
"""Problema 5.6: tensiones principales y criterios de plastificación de un estado general.

El polinomio característico no tiene solución sencilla a mano, de modo que sus raíces se
calculan numéricamente. Los círculos de Mohr muestran por qué los criterios discrepan.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, tensores

SIGMA = tensores.tensor_simetrico(100.0, 80.0, -60.0, s12=50.0, s23=30.0)  # MPa
SIGMA_Y = 200.0  # límite elástico, MPa

VALORES_LIBRO = {
    "I1": 120.0,
    "I2": -6200.0,
    "I3": -420000.0,
    "sigma_1": 142.8,
    "sigma_2": 44.0,
    "sigma_3": -66.8,
    "sigma_eq_vm": 182.0,
    "sigma_eq_tresca": 209.6,
    "fs_vm": 1.10,
    "fs_tresca": 0.954,
    "tau_max": 104.8,
}


def polinomio():
    """Devuelve los coeficientes de lambda^3 - I1 lambda^2 + I2 lambda - I3."""
    i1, i2, i3 = tensores.invariantes(SIGMA)
    return [1.0, -i1, i2, -i3]


def resolver():
    """Devuelve los resultados del problema."""
    i1, i2, i3 = tensores.invariantes(SIGMA)
    s1, s2, s3 = np.sort(np.roots(polinomio()).real)[::-1]
    vm = float(tensores.von_mises(SIGMA))
    tr = float(tensores.tresca(SIGMA))
    return {
        "I1": float(i1),
        "I2": float(i2),
        "I3": float(i3),
        "sigma_1": float(s1),
        "sigma_2": float(s2),
        "sigma_3": float(s3),
        "sigma_eq_vm": vm,
        "sigma_eq_tresca": tr,
        "fs_vm": SIGMA_Y / vm,
        "fs_tresca": SIGMA_Y / tr,
        "tau_max": float(tensores.tau_max(SIGMA)[0]),
    }


def graficar():
    """Dibuja el polinomio con sus raíces y los círculos de Mohr con el límite de Tresca."""
    datos = resolver()
    raices = [datos["sigma_1"], datos["sigma_2"], datos["sigma_3"]]
    fig, (ax_p, ax_m) = plt.subplots(1, 2, figsize=(12, 4.5))

    lam = np.linspace(-100, 180, 400)
    ax_p.plot(lam, np.polyval(polinomio(), lam) / 1e6, color="tab:blue", linewidth=2)
    ax_p.plot(raices, np.zeros(3), "o", color="tab:red", markersize=8, label="Tensiones principales")
    for raiz, desplazamiento in zip(raices, (-18, 18, -18)):  # cada etiqueta, al lado libre de la curva
        ax_p.annotate(f"{raiz:.1f}", (raiz, 0), textcoords="offset points", xytext=(desplazamiento, 8), ha="center")
    ax_p.axhline(0, color="0.4", linewidth=0.8)
    ax_p.set_xlabel(r"$\lambda$ (MPa)")
    ax_p.set_ylabel(r"$p(\lambda)$ ($10^6$ MPa$^3$)")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend()

    graficos.circulos_mohr(raices, "MPa", ax=ax_m)
    tau_y = SIGMA_Y / 2
    ax_m.axhline(tau_y, color="tab:red", linestyle="--", linewidth=1, label=rf"Límite de Tresca $\tau_y = {tau_y:.0f}$ MPa")
    ax_m.axhline(-tau_y, color="tab:red", linestyle="--", linewidth=1)
    centro = (datos["sigma_1"] + datos["sigma_3"]) / 2
    ax_m.plot(centro, datos["tau_max"], "o", color="tab:red", markersize=7, label=rf"$\tau_{{max}} = {datos['tau_max']:.1f}$ MPa")
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)

    ruta = graficos.guardar(fig, "p5_06")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Comprobación con los autovalores: {tensores.principales(SIGMA)[0].round(2)}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
