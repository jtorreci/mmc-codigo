#!/usr/bin/env python3
"""Problema 6.6: pérdida de carga en una tubería de acero comercial.

Compara la ley laminar, fuera de su dominio, con Darcy-Weisbach y Colebrook-White.
Colebrook-White es implícita en f, de modo que se resuelve por iteración.
"""
from mmc import fluidos

D = 0.100  # diámetro, m
L = 500.0  # longitud, m
RUGOSIDAD = 0.045e-3  # rugosidad de la pared, m
RHO = 1000.0  # densidad, kg/m^3
NU = 1e-6  # viscosidad cinemática, m^2/s
Q = 0.02  # caudal, m^3/s

VALORES_LIBRO = {"u": 2.55, "Re": 2.55e5, "f": 0.01816, "dp_turbulento_kPa": 294.0}


def resolver():
    """Devuelve los resultados del problema."""
    u = fluidos.velocidad_media(Q, D)
    re = fluidos.reynolds(u, D, NU)
    f = float(fluidos.colebrook(re, RUGOSIDAD / D))
    return {
        "u": u,
        "Re": re,
        "f": f,
        "dp_laminar_kPa": fluidos.hagen_poiseuille(RHO * NU, L, D, Q) / 1e3,
        "dp_turbulento_kPa": fluidos.darcy_weisbach(f, L, D, RHO, u) / 1e3,
    }


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")


if __name__ == "__main__":
    main()
