#!/usr/bin/env python3
"""Problema 2.7: trayectoria, línea de corriente y línea de traza del humo de una chimenea.

En un flujo no estacionario las tres líneas son distintas; dibujarlas juntas muestra que el
penacho guarda la historia del viento y solo junto a la boca indica su dirección actual.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos

U = 5.0  # componente constante del viento, m/s
C = 0.1  # tasa de variación de la componente transversal, m/s^2
T_OBS = 20.0  # instante de observación, s
X1_COMPARACION = 50.0  # abscisa en la que se comparan las tres líneas, m

VALORES_LIBRO = {
    "x2_trayectoria": 5.00,
    "x2_corriente": 20.0,
    "x2_traza": 15.0,
    "x1_final": 100.0,
    "x2_final": 20.0,
    "pendiente_boca": 0.4,
    "aceleracion_2": 0.1,
}


def trayectoria(x1):
    """Devuelve x2 sobre la trayectoria de la partícula emitida en t = 0."""
    t = x1 / U
    return C * t**2 / 2


def linea_corriente(x1):
    """Devuelve x2 sobre la línea de corriente que pasa por la boca en T_OBS."""
    return C * T_OBS / U * x1


def linea_traza(x1):
    """Devuelve x2 sobre el penacho en T_OBS: la partícula salió de la boca en tau."""
    tau = T_OBS - x1 / U
    return C * (T_OBS**2 - tau**2) / 2


def resolver():
    """Devuelve los resultados del problema."""
    x1_final = U * T_OBS
    resultados = {
        "x2_trayectoria": trayectoria(X1_COMPARACION),
        "x2_corriente": linea_corriente(X1_COMPARACION),
        "x2_traza": linea_traza(X1_COMPARACION),
        "x1_final": x1_final,
        "x2_final": linea_traza(x1_final),
        "pendiente_boca": C * T_OBS / U,
        "aceleracion_2": C,
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja las tres líneas en el instante de observación."""
    r = resolver()
    x1 = np.linspace(0, r["x1_final"], 201)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x1, trayectoria(x1), color="tab:blue", label="Trayectoria (partícula de t = 0)")
    ax.plot(x1, linea_corriente(x1), color="tab:orange", label=f"Línea de corriente (t = {T_OBS:.0f} s)")
    ax.plot(x1, linea_traza(x1), color="tab:green", label=f"Línea de traza (t = {T_OBS:.0f} s)")
    ax.axvline(X1_COMPARACION, color="0.5", linestyle=":", linewidth=1)
    for clave in ("x2_trayectoria", "x2_corriente", "x2_traza"):
        ax.plot(X1_COMPARACION, r[clave], "o", color="black", markersize=5)
        ax.annotate(f"{r[clave]:.1f} m", (X1_COMPARACION, r[clave]), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel(r"$x_1$ (m)")
    ax.set_ylabel(r"$x_2$ (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ruta = graficos.guardar(fig, "p2_07")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
