#!/usr/bin/env python3
"""Problema 4.15: agujero circular en el alma de una viga armada.

La solución de Kirsch es uniaxial, de modo que los estados biaxial y de corte puro se
obtienen por superposición. Las gráficas muestran dónde se concentra la tensión en el
borde y con qué rapidez se amortigua al alejarse del agujero.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import elasticidad, graficos

S1 = 100.0  # tensión lejana según x1 en la zona de flexión, MPa
S2 = 50.0  # tensión lejana según x2 en la zona de flexión, MPa
TAU = 60.0  # corte puro lejano en la zona de cortante, MPa
FY = 275.0  # límite elástico, MPa
A = 1.0  # radio del agujero; los resultados solo dependen de r/a

VALORES_LIBRO = {
    "sigma_tt_90": 250.0,
    "sigma_tt_0": 50.0,
    "sigma11_2a_90": 123.0,
    "sigma_max_cortante": 240.0,
    "fraccion_fy": 0.87,
}


def stt_biaxial(r, theta):
    """Tensión circunferencial con S1 según x1 más S2 según x2, que es Kirsch girada 90 grados."""
    return elasticidad.kirsch(S1, r, theta, A)[1] + elasticidad.kirsch(S2, r, theta - np.pi / 2, A)[1]


def stt_cortante(r, theta):
    """Tensión circunferencial con corte puro: tracción TAU a 45 grados y compresión a 135 grados."""
    traccion = elasticidad.kirsch(TAU, r, theta - np.pi / 4, A)[1]
    compresion = elasticidad.kirsch(-TAU, r, theta - 3 * np.pi / 4, A)[1]
    return traccion + compresion


def resolver():
    """Devuelve los resultados del problema."""
    borde = np.linspace(0, np.pi, 721)
    maximo_cortante = float(np.max(np.abs(stt_cortante(A, borde))))
    return {
        "sigma_tt_90": float(stt_biaxial(A, np.pi / 2)),
        "sigma_tt_0": float(stt_biaxial(A, 0.0)),
        # En theta = 90 grados la dirección circunferencial coincide con x1.
        "sigma11_2a_90": float(stt_biaxial(2 * A, np.pi / 2)),
        "sigma_max_cortante": maximo_cortante,
        "fraccion_fy": maximo_cortante / FY,
    }


def graficar():
    """Dibuja la tensión en el borde y su amortiguamiento radial, y guarda la figura."""
    fig, (ax_borde, ax_radial) = plt.subplots(1, 2, figsize=(11, 4.5))

    grados = np.linspace(0, 180, 361)
    theta = np.radians(grados)
    ax_borde.plot(grados, stt_biaxial(A, theta), color="tab:blue", linewidth=2, label="Biaxial")
    ax_borde.plot(grados, stt_cortante(A, theta), color="tab:orange", linewidth=2, label="Corte puro")
    ax_borde.axhline(FY, color="0.4", linestyle="--", linewidth=1, label=r"$\pm f_y$")
    ax_borde.axhline(-FY, color="0.4", linestyle="--", linewidth=1)
    ax_borde.axhline(0, color="0.4", linewidth=0.8)
    ax_borde.set_xlabel(r"Ángulo $\theta$ desde $x_1$ (grados)")
    ax_borde.set_ylabel(r"$\sigma_{\theta\theta}$ en el borde (MPa)")
    ax_borde.set_xticks(range(0, 181, 45))
    ax_borde.grid(True, alpha=0.3)
    ax_borde.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)

    r = np.linspace(A, 6 * A, 251)
    ax_radial.plot(r / A, stt_biaxial(r, np.pi / 2), color="tab:blue", linewidth=2, label=r"Biaxial, $\theta = 90^\circ$")
    ax_radial.plot(r / A, stt_cortante(r, 3 * np.pi / 4), color="tab:orange", linewidth=2, label=r"Corte puro, $\theta = 135^\circ$")
    ax_radial.axhline(S1, color="tab:blue", linestyle=":", linewidth=1, label="Valor lejano, biaxial")
    ax_radial.axhline(TAU, color="tab:orange", linestyle=":", linewidth=1, label="Valor lejano, corte puro")
    ax_radial.set_xlabel("Distancia relativa $r/a$")
    ax_radial.set_ylabel(r"$\sigma_{\theta\theta}$ (MPa)")
    ax_radial.grid(True, alpha=0.3)
    ax_radial.legend()

    ruta = graficos.guardar(fig, "p4_15")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
