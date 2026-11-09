#!/usr/bin/env python3
"""Problema 3.11: plasticidad uniaxial con endurecimiento isótropo lineal.

La curva de carga y descarga muestra de dónde sale la deformación permanente y por qué el
nuevo límite elástico coincide con la tensión máxima alcanzada.
"""
import matplotlib.pyplot as plt

from mmc import graficos

E = 200e3  # módulo de Young, MPa
SIGMA_Y0 = 300.0  # límite elástico inicial, MPa
H = 20e3  # módulo de endurecimiento plástico, MPa
SIGMA_MAX = 450.0  # tensión máxima, MPa

VALORES_LIBRO = {
    "eps_p": 0.00750,
    "eps_e": 0.00225,
    "eps_max": 0.00975,
    "sigma_y": 450.0,
    "Et_GPa": 18.2,
    "H_tangente_GPa": 22.2,
    "eps_p_tangente": 0.00675,
}


def resolver():
    """Devuelve los resultados del problema."""
    eps_p = (SIGMA_MAX - SIGMA_Y0) / H
    eps_e = SIGMA_MAX / E
    # Lectura alternativa del libro: los 20 GPa como pendiente tangente de la curva.
    h_tangente = E * H / (E - H)
    resultados = {
        "eps_p": eps_p,
        "eps_e": eps_e,
        "eps_max": eps_p + eps_e,
        "sigma_y": SIGMA_Y0 + H * eps_p,
        "Et_GPa": E * H / (E + H) / 1e3,
        "H_tangente_GPa": h_tangente / 1e3,
        "eps_p_tangente": (SIGMA_MAX - SIGMA_Y0) / h_tangente,
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja la carga, la descarga elástica y la deformación permanente."""
    r = resolver()
    eps_y = SIGMA_Y0 / E
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([0, eps_y, r["eps_max"]], [0, SIGMA_Y0, SIGMA_MAX], color="tab:blue", label="Carga")
    ax.plot([r["eps_max"], r["eps_p"]], [SIGMA_MAX, 0], color="tab:orange", linestyle="--", label="Descarga y recarga")
    puntos = {
        rf"$\sigma_{{y0}}$ = {SIGMA_Y0:.0f} MPa": (eps_y, SIGMA_Y0, (8, -12)),
        rf"$\sigma_{{\max}}$ = {SIGMA_MAX:.0f} MPa": (r["eps_max"], SIGMA_MAX, (-92, 6)),
        rf"$\varepsilon^p$ = {r['eps_p']:.5f}": (r["eps_p"], 0.0, (-74, 6)),
    }
    for nombre, (x, y, desplazamiento) in puntos.items():
        ax.plot(x, y, "o", color="black", markersize=5)
        ax.annotate(nombre, (x, y), textcoords="offset points", xytext=desplazamiento)
    ax.set_xlabel(r"Deformación $\varepsilon$")
    ax.set_ylabel(r"Tensión $\sigma$ (MPa)")
    ax.set_ylim(top=500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ruta = graficos.guardar(fig, "p3_11")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
