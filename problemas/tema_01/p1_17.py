#!/usr/bin/env python3
"""Problema 1.17: probeta de suelo en un ensayo triaxial verdadero.

Los tres círculos de Mohr muestran que un suelo comprimido en todas las direcciones soporta,
sin embargo, una tangencial importante, que es la que puede romperlo.
"""
from mmc import graficos, tensores

SIGMAS = (-200.0, -300.0, -400.0)  # tensiones principales, kPa (compresión negativa)

# J3 no se compara: vale cero y la tolerancia relativa no se aplica a un valor nulo.
VALORES_LIBRO = {
    "I1": -900.0,
    "I2": 260000.0,
    "I3": -2.40e7,
    "sigma_m": -300.0,
    "p": 300.0,
    "J2": 10000.0,
    "tau_max": 100.0,
    "sigma_n": -300.0,
    "xi": -519.6,
    "rho": 141.4,
    "tau_oct": 81.6,
}


def resolver():
    """Devuelve los resultados del problema."""
    i1, i2, i3 = tensores.invariantes(SIGMAS)
    j2, j3 = tensores.invariantes_desviador(SIGMAS)
    t_max, sigma_n = tensores.tau_max(SIGMAS)
    xi, rho, _ = tensores.haigh_westergaard(SIGMAS)
    resultados = {
        "I1": i1,
        "I2": i2,
        "I3": i3,
        "sigma_m": i1 / 3,
        "p": -i1 / 3,
        "J2": j2,
        "J3": j3,
        "tau_max": t_max,
        "sigma_n": sigma_n,
        "xi": xi,
        "rho": rho,
        "tau_oct": tensores.octaedricas(SIGMAS)[1],
    }
    return {clave: float(valor) for clave, valor in resultados.items()}


def graficar():
    """Dibuja los círculos de Mohr y marca el plano de tangencial máxima."""
    r = resolver()
    fig, ax = graficos.circulos_mohr(SIGMAS, unidad="kPa")
    ax.plot(r["sigma_n"], r["tau_max"], "o", color="black", markersize=5, zorder=3)
    ax.annotate(
        rf"$\tau_{{\max}}$ = {r['tau_max']:.0f} kPa",
        (r["sigma_n"], r["tau_max"]),
        textcoords="offset points",
        xytext=(8, 2),
    )
    for nombre, valor in zip((r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"), SIGMAS):
        ax.annotate(nombre, (valor, 0), textcoords="offset points", xytext=(-16 if valor == max(SIGMAS) else 2, 5))
    ruta = graficos.guardar(fig, "p1_17")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"{'figura':20s} {graficar()}")


if __name__ == "__main__":
    main()
