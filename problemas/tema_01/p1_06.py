#!/usr/bin/env python3
"""Problema 1.6: círculo de Mohr de un estado plano y plano de una soldadura.

La figura sitúa las caras X e Y y el plano S sobre el círculo de la chapa, y muestra por qué
la tangencial máxima del punto no está en el plano de la chapa, sino en el círculo mayor.
"""
import numpy as np

from mmc import graficos, tensores

SIGMA_XX = 80.0  # MPa
SIGMA_YY = 40.0  # MPa
SIGMA_XY = 20.0  # MPa
THETA_SOLDADURA = 60.0  # ángulo de la normal desde el eje x, grados

SIGMA = tensores.tensor_simetrico(SIGMA_XX, SIGMA_YY, 0.0, s12=SIGMA_XY)  # MPa

VALORES_LIBRO = {
    "centro": 60.0,
    "radio": 28.28,
    "sigma_1": 88.3,
    "sigma_2": 31.7,
    "theta_p": 22.5,
    "sigma_n_S": 67.3,
    "tau_n_S": -27.3,
    "tau_max": 44.1,
}


def componentes_plano(theta):
    """Devuelve (sigma_n, tau_n) del plano de normal a theta grados del eje x.

    La tangencial lleva signo: es positiva según la normal girada 90 grados antihorario.
    """
    angulo = np.radians(theta)
    normal = (np.cos(angulo), np.sin(angulo), 0.0)
    tangente = np.array([-np.sin(angulo), np.cos(angulo), 0.0])
    t, sigma_n, _ = tensores.componentes_intrinsecas(SIGMA, normal)
    return sigma_n, t @ tangente


def resolver():
    """Devuelve los resultados del problema."""
    valores, vectores = tensores.principales(SIGMA)
    sigma_n, tau_n = componentes_plano(THETA_SOLDADURA)
    resultados = {
        "centro": (SIGMA_XX + SIGMA_YY) / 2,
        "radio": (valores[0] - valores[1]) / 2,
        "sigma_1": valores[0],
        "sigma_2": valores[1],
        "sigma_3": valores[2],
        "theta_p": np.degrees(np.arctan2(vectores[1, 0], vectores[0, 0])) % 180,
        "sigma_n_S": sigma_n,
        "tau_n_S": tau_n,
        "tau_max": tensores.tau_max(SIGMA)[0],
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja los círculos de Mohr con las caras de la chapa y el plano de la soldadura."""
    r = resolver()
    fig, ax = graficos.circulos_mohr([r["sigma_1"], r["sigma_2"], r["sigma_3"]], unidad="MPa")
    angulo = np.linspace(0, 2 * np.pi, 361)
    ax.plot(
        r["centro"] + r["radio"] * np.cos(angulo),
        r["radio"] * np.sin(angulo),
        color="tab:orange",
        linewidth=2,
        label="Planos que contienen a z",
    )
    ax.plot([SIGMA_XX, SIGMA_YY], [SIGMA_XY, -SIGMA_XY], color="tab:orange", linewidth=1)
    puntos = {
        "X": (SIGMA_XX, SIGMA_XY),
        "Y": (SIGMA_YY, -SIGMA_XY),
        "S": (r["sigma_n_S"], r["tau_n_S"]),
        r"$\tau_{\max}$": (r["sigma_1"] / 2, r["tau_max"]),
    }
    for nombre, (x, y) in puntos.items():
        ax.plot(x, y, "o", color="black", markersize=5, zorder=3)
        desplazado = (-6, -16) if y == r["tau_max"] else (6, 6)
        ax.annotate(nombre, (x, y), textcoords="offset points", xytext=desplazado)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ruta = graficos.guardar(fig, "p1_06")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
