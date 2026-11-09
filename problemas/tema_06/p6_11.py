#!/usr/bin/env python3
"""Problema 6.11: transición laminar-turbulento a partir de ocho ensayos.

Con ocho ensayos la tabla se calcula mejor por ordenador, y el diagrama f-Re en escala
logarítmica muestra de un vistazo dónde abandonan los puntos la ley laminar.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import fluidos, graficos

D = 0.010  # diámetro, m
L = 2.0  # distancia entre tomas, m
MU = 1e-3  # viscosidad dinámica, Pa·s
RHO = 1000.0  # densidad, kg/m^3
Q = np.array([5, 10, 15, 20, 30, 50, 100, 200]) * 1e-6  # caudales, m^3/s
DP = np.array([41, 83, 120, 215, 580, 1420, 4750, 15700], dtype=float)  # caídas de presión, Pa
UMBRAL_LAMINAR = 0.05  # desviación relativa máxima para considerar laminar un ensayo

VALORES_LIBRO = {
    "f_Q10": 0.0512,
    "Re_crit_min": 1910.0,
    "Re_crit_max": 2546.0,
    "pendiente_inicial": 1.02,
    "pendiente_final": 1.72,
}


def ensayos():
    """Devuelve Re, f experimental y las leyes laminar y de Colebrook de cada ensayo."""
    u = fluidos.velocidad_media(Q, D)
    re = fluidos.reynolds(u, D, MU / RHO)
    f = 2 * DP * D / (RHO * u**2 * L)
    return re, f, 64 / re, fluidos.colebrook(re)


def acotar_transicion(re, f, f_laminar):
    """Devuelve el último Re sobre la ley laminar y el primero que se separa de ella."""
    separado = np.abs(f / f_laminar - 1) > UMBRAL_LAMINAR
    primero = int(np.argmax(separado))
    return float(re[primero - 1]), float(re[primero])


def resolver():
    """Devuelve los resultados del problema."""
    re, f, f_laminar, _ = ensayos()
    re_min, re_max = acotar_transicion(re, f, f_laminar)
    return {
        "f_Q10": float(f[1]),
        "Re_crit_min": re_min,
        "Re_crit_max": re_max,
        "pendiente_inicial": float(np.log(DP[1] / DP[0]) / np.log(Q[1] / Q[0])),
        "pendiente_final": float(np.log(DP[-1] / DP[-2]) / np.log(Q[-1] / Q[-2])),
    }


def graficar():
    """Dibuja f frente a Re con las leyes de cada régimen y sombrea la transición."""
    re, f, _, _ = ensayos()
    resultados = resolver()
    re_laminar = np.geomspace(500, 3200, 50)
    re_turbulento = np.geomspace(2546, 30000, 50)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(resultados["Re_crit_min"], resultados["Re_crit_max"], color="0.85", label="Cota de la transición")
    ax.loglog(re_laminar, 64 / re_laminar, color="tab:blue", label="Laminar, $f = 64/\\mathrm{Re}$")
    ax.loglog(re_turbulento, fluidos.colebrook(re_turbulento), color="tab:orange", linestyle="--",
              label="Colebrook-White, tubo liso")
    ax.loglog(re, f, "o", color="tab:red", markersize=6, label="Ensayos")
    ax.set_xlabel("Número de Reynolds, Re (adimensional)")
    ax.set_ylabel("Factor de fricción, $f$ (adimensional)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ruta = graficos.guardar(fig, "p6_11")
    graficos.cerrar(fig)
    return ruta


def main():
    re, f, f_laminar, f_colebrook = ensayos()
    print(f"{'Q (mL/s)':>9s} {'Re':>8s} {'f':>8s} {'64/Re':>8s} {'cociente':>9s} {'Colebrook':>10s}")
    for q, numero, medido, laminar, turbulento in zip(Q, re, f, f_laminar, f_colebrook):
        print(f"{q * 1e6:9.0f} {numero:8.0f} {medido:8.4f} {laminar:8.4f} {medido / laminar:9.2f} {turbulento:10.4f}")
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
