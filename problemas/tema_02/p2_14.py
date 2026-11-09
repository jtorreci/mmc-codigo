#!/usr/bin/env python3
"""Problema 2.14: compatibilidad e integración de un campo de deformaciones plano.

La malla deformada, amplificada, permite ver el desplazamiento que se ha reconstruido a partir
de las deformaciones y el peso del giro en la forma final del dominio.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos

A = 1e-3  # constante de epsilon_11, 1/m^2
B = 1e-3  # constante de epsilon_22, 1/m^2
C = 1e-3  # constante de epsilon_12, 1/m^2
LADO = 1.0  # lado del dominio cuadrado, m
AMPLIFICACION = 100.0  # factor de la deformada en la figura
DIVISIONES = 10  # divisiones de la malla por lado

VALORES_LIBRO = {"u1_mm": 0.500, "u2_mm": 0.833, "omega_rad": -5.00e-4}


def desplazamiento(x1, x2):
    """Devuelve (u1, u2) en m, integrados con u = 0 y giro nulo en el origen."""
    return A / 2 * x1**2 * x2, B * x1**2 * x2 - A / 6 * x1**3


def giro(x1, x2):
    """Devuelve el giro omega_12 en rad."""
    return A * x1**2 / 2 - B * x1 * x2


def resolver():
    """Devuelve los resultados del problema."""
    u1, u2 = desplazamiento(LADO, LADO)
    resultados = {
        # Condición de compatibilidad: d2e11/dx2^2 + d2e22/dx1^2 - 2 d2e12/dx1dx2 = 2B - 2C.
        "residuo_compatibilidad": 2 * B - 2 * C,
        "u1_mm": u1 * 1e3,
        "u2_mm": u2 * 1e3,
        "omega_rad": giro(LADO, LADO),
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja la malla inicial y la deformada amplificada."""
    r = resolver()
    fig, ax = plt.subplots(figsize=(6, 6))
    lineas = np.linspace(0, LADO, DIVISIONES + 1)
    continuo = np.linspace(0, LADO, 101)
    for valor in lineas:
        for x1, x2 in ((np.full_like(continuo, valor), continuo), (continuo, np.full_like(continuo, valor))):
            u1, u2 = desplazamiento(x1, x2)
            ax.plot(x1, x2, color="0.7", linestyle="--", linewidth=0.7)
            ax.plot(x1 + AMPLIFICACION * u1, x2 + AMPLIFICACION * u2, color="tab:blue", linewidth=1)
    ax.plot([], [], color="0.7", linestyle="--", linewidth=0.7, label="Dominio inicial")
    ax.plot([], [], color="tab:blue", linewidth=1, label=f"Deformada (x{AMPLIFICACION:.0f})")
    u1, u2 = desplazamiento(LADO, LADO)
    vertice = (LADO + AMPLIFICACION * u1, LADO + AMPLIFICACION * u2)
    ax.plot(*vertice, "o", color="black", markersize=5)
    ax.annotate(
        f"$u_1$ = {r['u1_mm']:.3f} mm\n$u_2$ = {r['u2_mm']:.3f} mm",
        vertice,
        textcoords="offset points",
        xytext=(-85, -36),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ (m)")
    ax.set_ylabel(r"$x_2$ (m)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ruta = graficos.guardar(fig, "p2_14")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:25s} {valor:.5g}")
    print(f"{'figura':25s} {graficar()}")


if __name__ == "__main__":
    main()
