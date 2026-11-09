#!/usr/bin/env python3
"""Problema 8.12: convergencia de una ménsula de triángulos de deformación constante.

El diagrama doble logarítmico convierte los errores en rectas cuya pendiente es la tasa
de convergencia, y permite comprobar que la flecha converge el doble de rápido que la energía.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos

# Unidades coherentes: kN y m.
L = 4.0  # longitud, m
CANTO = 1.0  # canto D, m
ESPESOR = 0.5  # espesor, m
E = 30e6  # módulo de elasticidad, kN/m^2
NU = 0.2  # coeficiente de Poisson
P = 500.0  # carga en el extremo, kN

N = np.array([2, 4, 8, 16, 32])  # divisiones del canto
ERROR_ENERGIA = np.array([0.668, 0.411, 0.221, 0.112, 0.0565])  # error relativo en la norma energética
FLECHA_MM = np.array([4.97, 7.38, 8.44, 8.75, 8.84])  # flecha de cada malla, mm
ERROR_OBJETIVO = 0.02  # error en energía buscado
TASA_FLECHA = 2  # tasa teórica de la flecha, empleada en la extrapolación

VALORES_LIBRO = {
    "v_exacta_mm": 8.87,
    "v_cortante_mm": 0.333,
    "cortante_pct": 3.76,
    "tasa_energia_final": 0.99,
    "tasa_flecha_inicial": 1.39,
    "tasa_flecha_final": 2.13,
    "v_richardson_mm": 8.85,
    "n_necesario": 90.4,
}


def flecha_exacta():
    """Devuelve las flechas total y debida al cortante, en mm."""
    inercia = ESPESOR * CANTO**3 / 12
    ei = E * inercia
    flexion = P * L**3 / (3 * ei)
    cortante = (4 + 5 * NU) * P * L * CANTO**2 / (24 * ei)
    return (flexion + cortante) * 1e3, cortante * 1e3


def error_flecha():
    """Devuelve el error relativo de la flecha en cada malla."""
    return 1 - FLECHA_MM / flecha_exacta()[0]


def tasas(errores):
    """Devuelve la tasa observada entre mallas consecutivas, con h dividido por dos."""
    return np.log2(errores[:-1] / errores[1:])


def resolver():
    """Devuelve los resultados del problema."""
    v_exacta, v_cortante = flecha_exacta()
    tasas_energia = tasas(ERROR_ENERGIA)
    tasas_flecha = tasas(error_flecha())
    v8, v16 = FLECHA_MM[2], FLECHA_MM[3]
    return {
        "v_exacta_mm": v_exacta,
        "v_cortante_mm": v_cortante,
        "cortante_pct": 100 * v_cortante / v_exacta,
        "tasa_energia_inicial": float(tasas_energia[0]),
        "tasa_energia_final": float(tasas_energia[-1]),
        "tasa_flecha_inicial": float(tasas_flecha[0]),
        "tasa_flecha_final": float(tasas_flecha[-1]),
        "v_richardson_mm": float(v16 + (v16 - v8) / (2**TASA_FLECHA - 1)),
        "n_necesario": float(N[-1] * ERROR_ENERGIA[-1] / ERROR_OBJETIVO),
    }


def graficar():
    """Dibuja los errores frente a h con las pendientes teóricas de referencia."""
    h = CANTO / N
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(h, ERROR_ENERGIA, "o-", color="tab:blue", label="Error en la norma energética")
    ax.loglog(h, error_flecha(), "s-", color="tab:red", label="Error de la flecha")
    ax.loglog(h, ERROR_ENERGIA**2, "x", color="0.3", markersize=8, label="Cuadrado del error en energía")
    # Las referencias se anclan en la malla más fina para compararlas con las últimas tasas.
    ax.loglog(h, ERROR_ENERGIA[-1] * (h / h[-1]), "--", color="tab:blue", alpha=0.6, label="Pendiente 1")
    ax.loglog(h, error_flecha()[-1] * (h / h[-1]) ** 2, "--", color="tab:red", alpha=0.6, label="Pendiente 2")
    ax.set_xticks(h, [f"1/{n}" for n in N])
    ax.minorticks_off()
    ax.set_xlabel("Tamaño de malla, $h$ (m)")
    ax.set_ylabel("Error relativo (adimensional)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ruta = graficos.guardar(fig, "p8_12")
    graficos.cerrar(fig)
    return ruta


def main():
    print(f"{'n':>4s} {'e_E':>8s} {'e_v':>8s} {'e_E^2':>8s}")
    for n, e_energia, e_flecha in zip(N, ERROR_ENERGIA, error_flecha()):
        print(f"{n:4d} {e_energia:8.4f} {e_flecha:8.4f} {e_energia**2:8.4f}")
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura guardada en {graficar()}")


if __name__ == "__main__":
    main()
