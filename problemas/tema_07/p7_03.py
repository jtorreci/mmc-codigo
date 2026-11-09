#!/usr/bin/env python3
"""Problema 7.3: fluencia y recuperación de un elemento de Kelvin.

Dibujar por separado los dos escalones de Boltzmann y su suma muestra por qué la
descarga se resuelve sumando un escalón negativo, sin ningún salto de deformación.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, viscoelasticidad

E = 600.0  # módulo del resorte, MPa
ETA = 6e4  # viscosidad del amortiguador, MPa·s
SIGMA0 = 12.0  # tensión aplicada, MPa
T1 = 300.0  # instante de la descarga, s
T_MAX = 700.0  # fin de la gráfica, s

INSTANTES = (0.0, T1)  # instantes de los escalones, s
SALTOS = (SIGMA0, -SIGMA0)  # saltos de tensión, MPa

VALORES_LIBRO = {
    "tau_K": 100.0,
    "eps_inf": 0.0200,
    "eps_50": 0.00787,
    "eps_100": 0.0126,
    "t95": 300.0,
    "eps_t1": 0.0190,
    "eps_600": 9.46e-4,
}


def fluencia(t):
    """Devuelve la función de fluencia del elemento de Kelvin."""
    return viscoelasticidad.fluencia_kelvin(t, E, ETA)


def deformacion(t):
    """Devuelve la deformación por superposición de los dos escalones."""
    return viscoelasticidad.boltzmann(fluencia, t, INSTANTES, SALTOS)


def resolver():
    """Devuelve los resultados del problema."""
    tau = ETA / E
    return {
        "tau_K": tau,
        "eps_inf": SIGMA0 / E,
        "eps_50": float(deformacion(50.0)),
        "eps_100": float(deformacion(100.0)),
        "t95": float(tau * np.log(20)),
        "eps_t1": float(deformacion(T1)),
        "eps_600": float(deformacion(600.0)),
    }


def graficar():
    """Dibuja la tensión aplicada y la deformación con sus dos escalones."""
    t = np.linspace(0, T_MAX, 701)
    tension = viscoelasticidad.boltzmann(np.ones_like, t, INSTANTES, SALTOS)
    curvas = {
        "Suma: deformación real": deformacion(t),
        r"Escalón $+\sigma_0$ en $t = 0$": viscoelasticidad.boltzmann(fluencia, t, INSTANTES[:1], SALTOS[:1]),
        r"Escalón $-\sigma_0$ en $t_1$": viscoelasticidad.boltzmann(fluencia, t, INSTANTES[1:], SALTOS[1:]),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=(1, 2.5))
    graficos.historia_temporal(t, {"Tensión aplicada": tension}, "", r"$\sigma$ (MPa)", ax=ax1)
    graficos.historia_temporal(t, curvas, "$t$ (s)", r"$\varepsilon$ (adimensional)", ax=ax2)
    # Los escalones van a trazos para que se vea la suma, que coincide con el primero hasta t1.
    for linea in ax2.lines[1:]:
        linea.set_linestyle("--")
    ax2.legend()
    ax2.axhline(0, color="0.4", linewidth=0.8)
    ruta = graficos.guardar(fig, "p7_03")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
