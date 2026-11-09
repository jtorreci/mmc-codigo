"""Operaciones con tensores simétricos de orden dos en tres dimensiones."""
import numpy as np


def _como_tensor(S):
    """Devuelve un tensor 3x3 simétrico; un vector de tres valores se toma como diagonal."""
    A = np.asarray(S, dtype=float)
    if A.shape == (3,):
        return np.diag(A)
    if A.shape != (3, 3):
        raise ValueError("Se esperaba un tensor 3x3 o tres valores principales.")
    if not np.allclose(A, A.T):
        raise ValueError("El tensor debe ser simétrico.")
    return A


def tensor_simetrico(s11, s22, s33, s12=0.0, s13=0.0, s23=0.0):
    """Construye el tensor 3x3 a partir de sus seis componentes."""
    return np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]], dtype=float)


def invariantes(S):
    """Devuelve los invariantes (I1, I2, I3) del tensor."""
    A = _como_tensor(S)
    i1 = np.trace(A)
    i2 = 0.5 * (i1**2 - np.trace(A @ A))
    return i1, i2, np.linalg.det(A)


def desviador(S):
    """Devuelve la parte desviadora del tensor."""
    A = _como_tensor(S)
    return A - np.trace(A) / 3 * np.eye(3)


def invariantes_desviador(S):
    """Devuelve (J2, J3), los invariantes del desviador."""
    s = desviador(S)
    return 0.5 * np.trace(s @ s), np.linalg.det(s)


def principales(S):
    """Devuelve los valores principales de mayor a menor y sus direcciones.

    Las direcciones son las columnas de la matriz y forman un triedro directo.
    """
    valores, vectores = np.linalg.eigh(_como_tensor(S))
    valores, vectores = valores[::-1], vectores[:, ::-1]
    if np.linalg.det(vectores) < 0:
        vectores[:, 2] = -vectores[:, 2]
    return valores, vectores


def tau_max(S):
    """Devuelve la tangencial máxima y la tensión normal en los planos donde actúa."""
    valores, _ = principales(S)
    return (valores[0] - valores[2]) / 2, (valores[0] + valores[2]) / 2


def octaedricas(S):
    """Devuelve las tensiones normal y tangencial octaédricas."""
    j2, _ = invariantes_desviador(S)
    return invariantes(S)[0] / 3, np.sqrt(2 * j2 / 3)


def haigh_westergaard(S):
    """Devuelve (xi, rho, theta), con el ángulo de Lode theta en radianes entre 0 y pi/3."""
    i1 = invariantes(S)[0]
    j2, j3 = invariantes_desviador(S)
    if np.isclose(j2, 0.0):
        return i1 / np.sqrt(3), 0.0, 0.0
    coseno = 1.5 * np.sqrt(3) * j3 / j2**1.5
    return i1 / np.sqrt(3), np.sqrt(2 * j2), np.arccos(np.clip(coseno, -1.0, 1.0)) / 3


def von_mises(S):
    """Devuelve la tensión equivalente de Von Mises, de un tensor o de sus principales."""
    return np.sqrt(3 * invariantes_desviador(S)[0])


def tresca(S):
    """Devuelve la tensión equivalente de Tresca, de un tensor o de sus principales."""
    valores, _ = principales(S)
    return valores[0] - valores[2]


def componentes_intrinsecas(S, n):
    """Devuelve (t, sigma_n, tau) en el plano de normal n, que no hace falta normalizar."""
    normal = np.asarray(n, dtype=float)
    modulo = np.linalg.norm(normal)
    if modulo == 0:
        raise ValueError("La normal no puede ser nula.")
    normal = normal / modulo
    t = _como_tensor(S) @ normal
    sigma_n = t @ normal
    return t, sigma_n, np.sqrt(max(t @ t - sigma_n**2, 0.0))
