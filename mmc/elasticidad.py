"""Relaciones de elasticidad lineal isótropa."""
from typing import NamedTuple

import numpy as np


class Constantes(NamedTuple):
    """Constantes derivadas: cortante, volumétrico, Lamé y módulo edométrico."""

    G: float
    K: float
    lam: float
    M: float


def constantes_elasticas(E, nu):
    """Devuelve las constantes derivadas a partir de E y del coeficiente de Poisson."""
    G = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return Constantes(G=G, K=E / (3 * (1 - 2 * nu)), lam=lam, M=lam + 2 * G)


def hooke(eps, E, nu):
    """Devuelve la tensión de la deformación: sigma = lam tr(eps) I + 2 G eps."""
    c = constantes_elasticas(E, nu)
    deformacion = np.asarray(eps, dtype=float)
    return c.lam * np.trace(deformacion) * np.eye(3) + 2 * c.G * deformacion


def hooke_inversa(sigma, E, nu):
    """Devuelve la deformación de la tensión: eps = [(1 + nu) sigma - nu tr(sigma) I] / E."""
    tension = np.asarray(sigma, dtype=float)
    return ((1 + nu) * tension - nu * np.trace(tension) * np.eye(3)) / E


def kirsch(s, r, theta, a=1.0):
    """Devuelve (srr, stt, srt) en una placa con taladro de radio a.

    La tracción lejana s actúa según x1 y theta se mide desde x1, en radianes.
    r y theta admiten vectores. El estado biaxial se obtiene por superposición.
    """
    radio = np.asarray(r, dtype=float)
    if np.any(radio < a):
        raise ValueError("El radio no puede ser menor que el del taladro.")
    q2 = (a / radio) ** 2
    q4 = q2**2
    doble = 2 * np.asarray(theta, dtype=float)
    srr = s / 2 * (1 - q2) + s / 2 * (1 - 4 * q2 + 3 * q4) * np.cos(doble)
    stt = s / 2 * (1 + q2) - s / 2 * (1 + 3 * q4) * np.cos(doble)
    srt = -s / 2 * (1 + 2 * q2 - 3 * q4) * np.sin(doble)
    return srr, stt, srt
