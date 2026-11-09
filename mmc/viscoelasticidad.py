"""Modelos reológicos lineales y superposición de Boltzmann."""
from typing import NamedTuple

import numpy as np


class ParametrosZener(NamedTuple):
    """Módulos instantáneo y de equilibrio, y tiempos de retardo y de relajación."""

    E0: float
    Einf: float
    tau_s: float
    tau_e: float


def parametros_zener(E1, E2, eta):
    """Devuelve los parámetros de un resorte E1 en serie con un Kelvin (E2, eta)."""
    einf = E1 * E2 / (E1 + E2)
    tau_s = eta / E2
    return ParametrosZener(E0=E1, Einf=einf, tau_s=tau_s, tau_e=tau_s * einf / E1)


def relajacion_maxwell(t, sigma0, tau):
    """Devuelve la tensión de un Maxwell con deformación impuesta en t = 0."""
    return sigma0 * np.exp(-np.asarray(t, dtype=float) / tau)


def fluencia_kelvin(t, E, eta):
    """Devuelve la función de fluencia J(t) de un elemento de Kelvin."""
    return (1 - np.exp(-np.asarray(t, dtype=float) * E / eta)) / E


def fluencia_zener(t, E0, Einf, tau_s):
    """Devuelve la función de fluencia J(t) del modelo de Zener."""
    return 1 / Einf - (1 / Einf - 1 / E0) * np.exp(-np.asarray(t, dtype=float) / tau_s)


def relajacion_zener(t, E0, Einf, tau_s):
    """Devuelve el módulo de relajación E(t) del modelo de Zener."""
    tau_e = tau_s * Einf / E0
    return Einf + (E0 - Einf) * np.exp(-np.asarray(t, dtype=float) / tau_e)


def boltzmann(respuesta, t, instantes, saltos):
    """Superpone la respuesta unitaria a una historia de saltos.

    respuesta: función de fluencia o de relajación, evaluada en el tiempo desde cada salto.
    instantes y saltos: momento y magnitud de cada escalón de la carga.
    En el instante exacto de un salto devuelve el valor posterior al salto.
    """
    tiempo = np.asarray(t, dtype=float)
    total = np.zeros_like(tiempo)
    for instante, salto in zip(instantes, saltos):
        activo = tiempo >= instante
        total = total + np.where(activo, salto * respuesta(np.where(activo, tiempo - instante, 0.0)), 0.0)
    return total
