#!/usr/bin/env python3
"""Problema 7.1: ajuste de un modelo de Zener a un ensayo de fluencia.

La curva ajustada frente a las medidas, y su versión linealizada, permiten juzgar
si una sola exponencial basta para describir el ensayo.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, viscoelasticidad

SIGMA0 = 5.0  # tensión de ensayo, MPa
T_MIN = np.array([0.0, 10.0, 30.0, 60.0, 120.0])  # instantes de medida, min
DEFORMACION = np.array([0.10, 0.15, 0.22, 0.28, 0.33]) / 100  # deformaciones medidas, adimensional
DEFORMACION_INF = 0.35 / 100  # asíntota, adimensional
T_AJUSTE_MIN = 30.0  # medida empleada para obtener el tiempo de retardo, min

VALORES_LIBRO = {
    "E0": 5000.0,
    "Einf": 1429.0,
    "cociente_diferida": 2.50,
    "E2": 2000.0,
    "tau_s_min": 45.9,
    "eta_MPas": 5.51e6,
    "tau_e_min": 13.1,
    "tau_s_60min": 47.1,
}


def fraccion_pendiente(deformacion):
    """Devuelve la fracción de fluencia diferida que aún falta por producirse."""
    return (DEFORMACION_INF - deformacion) / (DEFORMACION_INF - DEFORMACION[0])


def tiempo_retardo(t_min):
    """Devuelve el tiempo de retardo en minutos, obtenido de la medida de t_min."""
    medida = DEFORMACION[T_MIN == t_min][0]
    return float(-t_min / np.log(fraccion_pendiente(medida)))


def modelo():
    """Devuelve los parámetros de Zener ajustados, con los tiempos en segundos."""
    e0 = SIGMA0 / DEFORMACION[0]
    einf = SIGMA0 / DEFORMACION_INF
    e2 = 1 / (1 / einf - 1 / e0)
    eta = tiempo_retardo(T_AJUSTE_MIN) * 60 * e2
    return viscoelasticidad.parametros_zener(e0, e2, eta), e2, eta


def resolver():
    """Devuelve los resultados del problema."""
    parametros, e2, eta = modelo()
    return {
        "E0": float(parametros.E0),
        "Einf": float(parametros.Einf),
        "cociente_diferida": float((DEFORMACION_INF - DEFORMACION[0]) / DEFORMACION[0]),
        "E2": float(e2),
        "tau_s_min": float(parametros.tau_s / 60),
        "eta_MPas": float(eta),
        "tau_e_min": float(parametros.tau_e / 60),
        "tau_s_60min": tiempo_retardo(60.0),
    }


def graficar():
    """Dibuja la curva ajustada y su linealización, cuya pendiente es -1/tau_s."""
    parametros, _, _ = modelo()
    t_min = np.linspace(0, 180, 200)
    ajuste = SIGMA0 * viscoelasticidad.fluencia_zener(t_min * 60, parametros.E0, parametros.Einf, parametros.tau_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    graficos.historia_temporal(t_min, {"Modelo de Zener": ajuste * 100}, "$t$ (min)", r"$\varepsilon$ (%)", ax=ax1)
    ax1.plot(T_MIN, DEFORMACION * 100, "o", color="tab:red", label="Medidas")
    ax1.axhline(DEFORMACION_INF * 100, color="0.4", linestyle="--", label="Asíntota")
    ax1.legend()

    medidas = T_MIN > 0
    tau_min = parametros.tau_s / 60
    ax2.plot(t_min, -t_min / tau_min, label=rf"Recta de pendiente $-1/\tau_\sigma$ ($\tau_\sigma$ = {tau_min:.1f} min)")
    ax2.plot(T_MIN[medidas], np.log(fraccion_pendiente(DEFORMACION[medidas])), "o", color="tab:red", label="Medidas")
    ax2.set_xlabel("$t$ (min)")
    ax2.set_ylabel(r"$\ln\,[(\varepsilon_\infty-\varepsilon)/(\varepsilon_\infty-\varepsilon_0)]$ (adimensional)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    ruta = graficos.guardar(fig, "p7_01")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
