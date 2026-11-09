"""Flujo en tuberías de sección circular."""
import numpy as np


def velocidad_media(Q, D):
    """Devuelve la velocidad media a partir del caudal y el diámetro."""
    return 4 * Q / (np.pi * D**2)


def reynolds(u, D, nu):
    """Devuelve el número de Reynolds con la viscosidad cinemática nu."""
    return u * D / nu


def colebrook(re, rugosidad_relativa=0.0, tol=1e-12, max_iter=200):
    """Resuelve el factor de fricción de Darcy con Colebrook-White por punto fijo.

    re admite vectores. rugosidad_relativa es epsilon/D; con 0 se obtiene el tubo liso.
    """
    numero = np.asarray(re, dtype=float)
    if np.any(numero <= 0):
        raise ValueError("El número de Reynolds debe ser positivo.")
    f = np.full_like(numero, 0.02)
    for _ in range(max_iter):
        nuevo = (-2 * np.log10(rugosidad_relativa / 3.7 + 2.51 / (numero * np.sqrt(f)))) ** -2
        if np.all(np.abs(nuevo - f) <= tol):
            return nuevo
        f = nuevo
    raise RuntimeError("Colebrook-White no ha convergido.")


def darcy_weisbach(f, L, D, rho, u):
    """Devuelve la pérdida de presión de Darcy-Weisbach."""
    return f * L / D * rho * u**2 / 2


def hagen_poiseuille(mu, L, D, Q):
    """Devuelve la pérdida de presión laminar con la viscosidad dinámica mu."""
    return 128 * mu * L * Q / (np.pi * D**4)
