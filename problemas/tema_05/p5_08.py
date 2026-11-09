#!/usr/bin/env python3
"""Problema 5.8: ajustes de Drucker-Prager al cono de Mohr-Coulomb.

Convenio de este problema: compresión positiva. La sección desviadora muestra que cada
cono solo coincide con el hexágono en un meridiano y cuánto se separa en los demás.
"""
import matplotlib.pyplot as plt
import numpy as np

from mmc import graficos, tensores

C = 25.0  # cohesión, kPa
PHI = np.radians(30.0)  # ángulo de fricción
SIGMA_3 = 100.0  # tensión menor de los estados de comprobación, kPa

VALORES_LIBRO = {
    "alpha_compresion": 0.231,
    "k_compresion_kPa": 30.0,
    "alpha_extension": 0.165,
    "k_extension_kPa": 21.4,
    "sigma_fallo_kPa": 386.6,
    "I1_kPa": 586.6,
    "f_extension_cono_compresion_kPa": -66.2,
    "f_compresion_cono_extension_kPa": 47.3,
}


def ajuste(signo):
    """Devuelve (alpha, k); signo = +1 ajusta el meridiano de compresión y -1 el de extensión."""
    denominador = np.sqrt(3) * (3 - signo * np.sin(PHI))
    return 2 * np.sin(PHI) / denominador, 6 * C * np.cos(PHI) / denominador


def funcion_f(estado, alpha, k):
    """Evalúa f = sqrt(J2) - alpha I1 - k; los tensores de entrada son de compresión positiva."""
    i1 = tensores.invariantes(estado)[0]
    j2 = tensores.invariantes_desviador(estado)[0]
    return float(np.sqrt(j2) - alpha * i1 - k)


def tension_de_fallo():
    """Devuelve la tensión mayor de rotura de Mohr-Coulomb con la tensión menor SIGMA_3."""
    kp = (1 + np.sin(PHI)) / (1 - np.sin(PHI))
    return SIGMA_3 * kp + 2 * C * np.sqrt(kp)


def resolver():
    """Devuelve los resultados del problema."""
    alpha_c, k_c = ajuste(+1)
    alpha_e, k_e = ajuste(-1)
    s1 = tension_de_fallo()
    compresion = (s1, SIGMA_3, SIGMA_3)
    extension = (s1, s1, SIGMA_3)
    i1 = float(tensores.invariantes(compresion)[0])
    return {
        "alpha_compresion": float(alpha_c),
        "k_compresion_kPa": float(k_c),
        "alpha_extension": float(alpha_e),
        "k_extension_kPa": float(k_e),
        "sigma_fallo_kPa": float(s1),
        "I1_kPa": i1,
        "f_compresion_cono_compresion_kPa": funcion_f(compresion, alpha_c, k_c),
        "f_extension_cono_extension_kPa": funcion_f(extension, alpha_e, k_e),
        "f_extension_cono_compresion_kPa": funcion_f(extension, alpha_c, k_c),
        "f_compresion_cono_extension_kPa": funcion_f(compresion, alpha_e, k_e),
        "radio_compresion_kPa": float(np.sqrt(2) * (alpha_c * i1 + k_c)),
        "radio_extension_kPa": float(np.sqrt(2) * (alpha_e * i1 + k_e)),
    }


def hexagono_mohr_coulomb(i1):
    """Devuelve el contorno de Mohr-Coulomb en la sección desviadora de primer invariante i1.

    Cada dirección de la sección es un desviador unitario d. El estado m + t d alcanza el
    criterio sigma_max = Kp sigma_min + 2 c sqrt(Kp) para un único valor de t, que es el radio.
    """
    kp = (1 + np.sin(PHI)) / (1 - np.sin(PHI))
    m = i1 / 3
    beta = np.radians(np.arange(0.0, 361.0, 1.0))
    ejes = np.radians([90.0, 210.0, 330.0])  # posición de sigma_1, sigma_2 y sigma_3 en la sección
    d = np.sqrt(2 / 3) * np.cos(beta[:, None] - ejes)
    radio = (2 * C * np.sqrt(kp) + (kp - 1) * m) / (d.max(axis=1) - kp * d.min(axis=1))
    return radio * np.cos(beta), radio * np.sin(beta)


def graficar():
    """Dibuja el hexágono de Mohr-Coulomb y los dos círculos de Drucker-Prager."""
    datos = resolver()
    radio_c, radio_e = datos["radio_compresion_kPa"], datos["radio_extension_kPa"]
    fi = np.linspace(0, 2 * np.pi, 361)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for angulo, nombre in zip((90, 210, 330), (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$")):
        x, y = 280 * np.cos(np.radians(angulo)), 280 * np.sin(np.radians(angulo))
        ax.plot([0, x], [0, y], color="0.6", linestyle="--", linewidth=0.8)
        ax.text(1.05 * x, 1.05 * y, nombre, ha="center", va="center")
    ax.plot(*hexagono_mohr_coulomb(datos["I1_kPa"]), color="0.15", linewidth=2, label="Mohr-Coulomb")
    ax.plot(radio_c * np.cos(fi), radio_c * np.sin(fi), color="tab:blue", linewidth=2, label="Drucker-Prager, ajuste en compresión")
    ax.plot(radio_e * np.cos(fi), radio_e * np.sin(fi), color="tab:orange", linewidth=2, label="Drucker-Prager, ajuste en extensión")
    ax.plot(0, radio_c, "o", color="tab:red", markersize=8, label=f"({datos['sigma_fallo_kPa']:.1f}, {SIGMA_3:.0f}, {SIGMA_3:.0f}) kPa")
    ax.set_aspect("equal")
    ax.set_xlim(-320, 320)
    ax.set_ylim(-320, 320)
    ax.set_xlabel(f"Sección desviadora con $I_1 = {datos['I1_kPa']:.1f}$ kPa, coordenada horizontal (kPa)")
    ax.set_ylabel("Coordenada vertical (kPa)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    ruta = graficos.guardar(fig, "p5_08")
    graficos.cerrar(fig)
    return ruta


def main():
    for clave, valor in resolver().items():
        print(f"{clave:36s} {valor:.5g}")
    print(f"Figura: {graficar()}")


if __name__ == "__main__":
    main()
