#!/usr/bin/env python3
"""Problema 7.5: carga y descarga de un modelo de Zener.

La historia de deformación separa la respuesta instantánea de la retardada, y la curva
de J(t)·E(t) comprueba que fluencia y relajación no son funciones inversas entre sí.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, viscoelasticidad

E1 = 500.0  # módulo del resorte aislado, MPa
E2 = 800.0  # módulo del resorte del elemento de Kelvin, MPa
ETA = 8e4  # viscosidad del amortiguador, MPa·s
SIGMA0 = 5.0  # tensión aplicada, MPa
T1 = 300.0  # instante de la descarga, s
T_MAX = 700.0  # fin de la gráfica, s
DT_LIMITE = 1e-9  # separación para evaluar el límite por la izquierda, s

INSTANTES = (0.0, T1)  # instantes de los escalones, s
SALTOS = (SIGMA0, -SIGMA0)  # saltos de tensión, MPa
PARAMETROS = viscoelasticidad.parametros_zener(E1, E2, ETA)

VALORES_LIBRO = {
    "E0": 500.0,
    "Einf": 307.7,
    "tau_s": 100.0,
    "tau_e": 61.5,
    "eps_0_mas": 0.0100,
    "eps_equilibrio": 0.01625,
    "eps_t1_menos": 0.0159,
    "eps_t1_mas": 0.00594,
    "eps_600": 2.96e-4,
    "E_100": 345.6,
    "J_100": 2.79e-3,
    "J_por_E_100": 0.964,
}


def fluencia(t):
    """Devuelve la función de fluencia del modelo de Zener."""
    return viscoelasticidad.fluencia_zener(t, PARAMETROS.E0, PARAMETROS.Einf, PARAMETROS.tau_s)


def relajacion(t):
    """Devuelve el módulo de relajación del modelo de Zener."""
    return viscoelasticidad.relajacion_zener(t, PARAMETROS.E0, PARAMETROS.Einf, PARAMETROS.tau_s)


def deformacion(t):
    """Devuelve la deformación por superposición de los dos escalones."""
    return viscoelasticidad.boltzmann(fluencia, t, INSTANTES, SALTOS)


def resolver():
    """Devuelve los resultados del problema."""
    return {
        "E0": float(PARAMETROS.E0),
        "Einf": float(PARAMETROS.Einf),
        "tau_s": float(PARAMETROS.tau_s),
        "tau_e": float(PARAMETROS.tau_e),
        "eps_0_mas": float(deformacion(0.0)),
        "eps_equilibrio": float(SIGMA0 / PARAMETROS.Einf),
        "eps_t1_menos": float(deformacion(T1 - DT_LIMITE)),
        "eps_t1_mas": float(deformacion(T1)),
        "eps_600": float(deformacion(600.0)),
        "E_100": float(relajacion(100.0)),
        "J_100": float(fluencia(100.0)),
        "J_por_E_100": float(fluencia(100.0) * relajacion(100.0)),
    }


def graficar():
    """Dibuja la deformación con sus saltos y el producto J(t)·E(t)."""
    # Se incluyen los dos lados del salto para que la descarga se dibuje vertical.
    t = np.concatenate((np.linspace(0, T1 - DT_LIMITE, 300), np.linspace(T1, T_MAX, 400)))
    t_dualidad = np.linspace(0, 6 * PARAMETROS.tau_s, 400)
    equilibrio = SIGMA0 / PARAMETROS.Einf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    graficos.historia_temporal(t, {"Deformación": deformacion(t) * 100}, "$t$ (s)", r"$\varepsilon$ (%)", ax=ax1)
    ax1.axhline(equilibrio * 100, color="0.4", linestyle="--", label=r"Equilibrio $\sigma_0/E_\infty$")
    ax1.legend()
    producto = fluencia(t_dualidad) * relajacion(t_dualidad)
    graficos.historia_temporal(t_dualidad, {r"$J(t)\,E(t)$": producto}, "$t$ (s)", "$J\\,E$ (adimensional)", ax=ax2)
    ax2.axhline(1, color="0.4", linestyle="--", label="Valor 1")
    ax2.legend()
    fig.tight_layout()
    ruta = graficos.guardar(fig, "p7_05")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
