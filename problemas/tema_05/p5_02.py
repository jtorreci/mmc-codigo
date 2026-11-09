#!/usr/bin/env python3
"""Problema 5.2: criterios de Tresca y de Von Mises en tensión plana.

La gráfica compara ambas superficies en el plano sigma_3 = 0 y sitúa el estado dado:
el factor de seguridad es la distancia, medida sobre el radio, hasta cada superficie.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, tensores

ESTADO = (200.0, 100.0, 0.0)  # tensiones principales, MPa
SIGMA_Y = 250.0  # límite elástico, MPa

VALORES_LIBRO = {
    "sigma_eq_vm": 173.0,
    "sigma_eq_tresca": 200.0,
    "fs_vm": 1.44,
    "fs_tresca": 1.25,
    "relacion": 1.155,
}


def resolver():
    """Devuelve los resultados del problema."""
    vm = float(tensores.von_mises(ESTADO))
    tr = float(tensores.tresca(ESTADO))
    return {
        "sigma_eq_vm": vm,
        "sigma_eq_tresca": tr,
        "fs_vm": SIGMA_Y / vm,
        "fs_tresca": SIGMA_Y / tr,
        "relacion": tr / vm,
    }


def contorno(criterio):
    """Devuelve el contorno de plastificación escalando cada dirección del plano sigma_3 = 0."""
    fi = np.radians(np.arange(0.0, 360.5, 0.5))
    direcciones = np.column_stack((np.cos(fi), np.sin(fi), np.zeros_like(fi)))
    escala = np.array([SIGMA_Y / criterio(d) for d in direcciones])
    return escala * direcciones[:, 0], escala * direcciones[:, 1]


def graficar():
    """Dibuja la elipse de Von Mises, el hexágono de Tresca y el estado dado."""
    datos = resolver()
    s1, s2, _ = ESTADO
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(*contorno(tensores.von_mises), color="tab:blue", linewidth=2, label="Von Mises")
    ax.plot(*contorno(tensores.tresca), color="tab:orange", linewidth=2, label="Tresca")
    fs_vm, fs_tr = datos["fs_vm"], datos["fs_tresca"]
    ax.plot([0, fs_vm * s1], [0, fs_vm * s2], color="0.4", linestyle="--", linewidth=1)
    ax.plot(s1, s2, "o", color="tab:red", markersize=8, label=f"Estado ({s1:.0f}, {s2:.0f}) MPa")
    ax.plot(fs_tr * s1, fs_tr * s2, "s", color="tab:orange", markersize=8, label=f"Plastifica con Tresca: FS = {fs_tr:.2f}")
    ax.plot(fs_vm * s1, fs_vm * s2, "D", color="tab:blue", markersize=8, label=f"Plastifica con Von Mises: FS = {fs_vm:.2f}")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_1$ (MPa)")
    ax.set_ylabel(r"$\sigma_2$ (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    ruta = graficos.guardar(fig, "p5_02")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
