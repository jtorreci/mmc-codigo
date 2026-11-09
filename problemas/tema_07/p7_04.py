#!/usr/bin/env python3
"""Problema 7.4: deformación bajo una carga escalonada con fluencia potencial.

La gráfica enseña los saltos instantáneos y la memoria del material: con la misma
tensión final, la deformación difiere de la que daría esa tensión aplicada desde el principio.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, viscoelasticidad

J0 = 2e-4  # flexibilidad instantánea, 1/MPa
TAU = 100.0  # parámetro de tiempo de la ley potencial, s
INSTANTES = (0.0, 200.0, 400.0)  # instantes de los escalones, s
SALTOS = (5.0, 3.0, -5.0)  # saltos de tensión, MPa
TENSION_FINAL = 3.0  # tensión del último tramo, MPa
T_MIN, T_MAX = -60.0, 620.0  # límites de la gráfica, s
DT_LIMITE = 1e-9  # separación para evaluar el límite por la izquierda, s

VALORES_LIBRO = {
    "eps_100": 2.00e-3,
    "eps_200_menos": 2.41e-3,
    "eps_200_mas": 3.01e-3,
    "eps_400_menos": 4.45e-3,
    "eps_400_mas": 3.45e-3,
    "eps_500": 2.88e-3,
    "eps_500_3MPa": 1.94e-3,
}


def fluencia(t):
    """Devuelve la función de fluencia potencial, que no está acotada."""
    return J0 * (1 + np.sqrt(np.asarray(t, dtype=float) / TAU))


def deformacion(t):
    """Devuelve la deformación por superposición de los escalones."""
    return viscoelasticidad.boltzmann(fluencia, t, INSTANTES, SALTOS)


def malla_con_saltos():
    """Devuelve instantes que incluyen ambos lados de cada salto y se agrupan tras él.

    La raíz cuadrada tiene pendiente infinita justo después de cada escalón.
    """
    bordes = (T_MIN, *INSTANTES, T_MAX)
    tramos = []
    for inicio, fin in zip(bordes[:-1], bordes[1:]):
        fraccion = np.linspace(0, 1, 150) ** 2
        tramos.append(inicio + (fin - DT_LIMITE - inicio) * fraccion)
    return np.concatenate(tramos)


def resolver():
    """Devuelve los resultados del problema."""
    return {
        "eps_100": float(deformacion(100.0)),
        "eps_200_menos": float(deformacion(200.0 - DT_LIMITE)),
        "eps_200_mas": float(deformacion(200.0)),
        "eps_400_menos": float(deformacion(400.0 - DT_LIMITE)),
        "eps_400_mas": float(deformacion(400.0)),
        "eps_500": float(deformacion(500.0)),
        "eps_500_3MPa": float(TENSION_FINAL * fluencia(500.0)),
    }


def graficar():
    """Dibuja la historia de tensión y la deformación que produce, con sus saltos."""
    t = malla_con_saltos()
    tension = viscoelasticidad.boltzmann(np.ones_like, t, INSTANTES, SALTOS)
    solo_final = viscoelasticidad.boltzmann(fluencia, t, (0.0,), (TENSION_FINAL,))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=(1, 2.5))
    graficos.historia_temporal(t, {"Tensión aplicada": tension}, "", r"$\sigma$ (MPa)", ax=ax1)
    curvas = {"Historia escalonada": deformacion(t) * 1e3, "3 MPa desde el principio": solo_final * 1e3}
    graficos.historia_temporal(t, curvas, "$t$ (s)", r"$\varepsilon$ ($\times 10^{-3}$)", ax=ax2)
    for instante in INSTANTES:
        ax1.axvline(instante, color="0.7", linewidth=0.8, linestyle=":")
        ax2.axvline(instante, color="0.7", linewidth=0.8, linestyle=":")
    ruta = graficos.guardar(fig, "p7_04")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
