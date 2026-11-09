#!/usr/bin/env python3
"""Problema 6.4: caudal de un tubo liso con la pérdida de presión impuesta.

Con el caudal como incógnita, Re y f dependen de la velocidad buscada, de modo que
Colebrook-White y Darcy-Weisbach deben iterarse juntas hasta que la velocidad se estabiliza.
"""
import numpy as np

from mmc import fluidos

D = 0.002  # diámetro, m
L = 1.0  # longitud, m
DP = 2e5  # pérdida de presión impuesta, Pa
MU = 1e-3  # viscosidad dinámica, Pa·s
RHO = 1000.0  # densidad, kg/m^3
RE_LAMINAR = 2300.0  # límite del régimen laminar
F_INICIAL = 0.03  # factor de fricción de partida

M3S_A_LH = 3.6e6  # conversión de m^3/s a L/h

VALORES_LIBRO = {
    "Q_HP_Lh": 283.0,
    "u_HP": 25.0,
    "Re_HP": 5.00e4,
    "dp_max_kPa": 9.20,
    "f": 0.0307,
    "u": 5.10,
    "Q_Lh": 57.7,
    "Q_4mm_Lh": 378.0,
    "factor": 6.55,
    "factor_HP": 16.0,
}


def iterar_caudal(diametro, tol=1e-10, max_iter=100):
    """Devuelve (f, u, Re) de un tubo liso iterando entre Colebrook y Darcy-Weisbach."""
    nu = MU / RHO
    f = F_INICIAL
    for _ in range(max_iter):
        u = np.sqrt(2 * DP * diametro / (RHO * f * L))
        re = fluidos.reynolds(u, diametro, nu)
        nuevo = float(fluidos.colebrook(re))
        if abs(nuevo - f) <= tol:
            return nuevo, float(u), float(re)
        f = nuevo
    raise RuntimeError("La iteración del caudal no ha convergido.")


def caudal_laminar(diametro):
    """Devuelve el caudal de Hagen-Poiseuille: la caída de presión es lineal en Q."""
    return DP / fluidos.hagen_poiseuille(MU, L, diametro, 1.0)


def caudal_medio(u, diametro):
    """Devuelve el caudal correspondiente a una velocidad media."""
    return u * np.pi * diametro**2 / 4


def resolver():
    """Devuelve los resultados del problema."""
    q_hp = caudal_laminar(D)
    u_hp = q_hp / (np.pi * D**2 / 4)
    f, u, re = iterar_caudal(D)
    q = caudal_medio(u, D)
    f_doble, u_doble, re_doble = iterar_caudal(2 * D)
    q_doble = caudal_medio(u_doble, 2 * D)
    return {
        "Q_HP_Lh": float(q_hp * M3S_A_LH),
        "u_HP": float(u_hp),
        "u_max_HP": float(2 * u_hp),
        "Re_HP": float(fluidos.reynolds(u_hp, D, MU / RHO)),
        "dp_max_kPa": 32 * MU**2 * L * RE_LAMINAR / (RHO * D**3) / 1e3,
        "f": f,
        "u": u,
        "Re": re,
        "Q_Lh": float(q * M3S_A_LH),
        "dp_comprobacion_kPa": float(fluidos.darcy_weisbach(f, L, D, RHO, u) / 1e3),
        "f_4mm": f_doble,
        "u_4mm": u_doble,
        "Re_4mm": re_doble,
        "Q_4mm_Lh": float(q_doble * M3S_A_LH),
        "factor": float(q_doble / q),
        "factor_HP": float(caudal_laminar(2 * D) / q_hp),
    }


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")


if __name__ == "__main__":
    main()
