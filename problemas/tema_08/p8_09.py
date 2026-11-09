#!/usr/bin/env python3
"""Problema 8.9: cercha Warren de una pasarela con elementos de barra.

Ensamblar y resolver a mano un sistema de 7 x 7 no aporta nada; el ordenador da
desplazamientos, axiles y reacciones, y deja al alumno las comprobaciones de equilibrio.
"""
import numpy as np

# Unidades coherentes: kN y m.
E = 210e6  # módulo de elasticidad, kN/m^2
A = 1500e-6  # área de la sección, m^2
EA = E * A  # rigidez axil, kN
NODOS = np.array([[0, 0], [4, 0], [8, 0], [2, 2], [6, 2]], dtype=float)  # coordenadas 1 a 5, m
BARRAS = [(1, 2), (2, 3), (4, 5), (1, 4), (4, 2), (2, 5), (5, 3)]  # nodos de cada barra
P = 60.0  # carga vertical descendente en el nodo 2, kN
H = 10.0  # carga horizontal hacia la derecha en el nodo 4, kN
# Grados de libertad prescritos: u1, v1 y v3 (u de un nodo n es 2n-2 y v, 2n-1).
GDL_FIJOS = [0, 1, 5]

VALORES_LIBRO = {
    "u2_mm": 0.476,
    "v2_mm": -2.35,
    "u3_mm": 0.889,
    "u4_mm": 0.918,
    "v4_mm": -1.41,
    "u5_mm": 0.0925,
    "v5_mm": -1.38,
    "N12": 37.5,
    "N23": 32.5,
    "N45": -65.0,
    "N14": -38.9,
    "N42": 38.9,
    "N25": 46.0,
    "N53": -46.0,
    "R1x": -10.0,
    "R1y": 27.5,
    "R3y": 32.5,
    "delta2_mm": 2.35,
    "trabajo_kNm": 0.0750,
}


def gdl(nodo):
    """Devuelve los dos grados de libertad de un nodo numerado desde 1."""
    return np.array([2 * nodo - 2, 2 * nodo - 1])


def geometria(barra):
    """Devuelve la longitud de una barra y el vector g de la matriz K^e = (EA/L) g g^T."""
    i, j = barra
    delta = NODOS[j - 1] - NODOS[i - 1]
    longitud = np.linalg.norm(delta)
    c, s = delta / longitud
    return longitud, np.array([-c, -s, c, s])


def ensamblar():
    """Devuelve la matriz de rigidez global de la cercha, en kN/m."""
    k = np.zeros((2 * len(NODOS), 2 * len(NODOS)))
    for barra in BARRAS:
        longitud, g = geometria(barra)
        indices = np.concatenate((gdl(barra[0]), gdl(barra[1])))
        k[np.ix_(indices, indices)] += EA / longitud * np.outer(g, g)
    return k


def resolver_cholesky(k, f):
    """Resuelve el sistema reducido, simétrico y definido positivo."""
    l = np.linalg.cholesky(k)
    return np.linalg.solve(l.T, np.linalg.solve(l, f))


def desplazamientos(k, f):
    """Devuelve el vector global de desplazamientos con los apoyos eliminados."""
    libres = np.setdiff1d(np.arange(k.shape[0]), GDL_FIJOS)
    u = np.zeros(k.shape[0])
    u[libres] = resolver_cholesky(k[np.ix_(libres, libres)], f[libres])
    return u


def axiles(u):
    """Devuelve el axil de cada barra, positivo en tracción, y sus longitudes."""
    resultado = []
    longitudes = []
    for barra in BARRAS:
        longitud, g = geometria(barra)
        indices = np.concatenate((gdl(barra[0]), gdl(barra[1])))
        resultado.append(EA / longitud * g @ u[indices])
        longitudes.append(longitud)
    return np.array(resultado), np.array(longitudes)


def resolver():
    """Devuelve los resultados del problema, con desplazamientos en mm y fuerzas en kN."""
    k = ensamblar()
    f = np.zeros(k.shape[0])
    f[gdl(2)[1]] = -P
    f[gdl(4)[0]] = H
    u = desplazamientos(k, f)
    n, longitudes = axiles(u)
    reacciones = k @ u - f

    # Método de la carga unitaria: una fuerza unidad descendente en el nodo 2.
    f_unidad = np.zeros(k.shape[0])
    f_unidad[gdl(2)[1]] = -1.0
    n_unidad, _ = axiles(desplazamientos(k, f_unidad))

    resultados = {
        "b_mas_r": float(len(BARRAS) + len(GDL_FIJOS)),
        "dos_n": float(2 * len(NODOS)),
        "delta2_mm": float(np.sum(n * n_unidad * longitudes / EA) * 1e3),
        "trabajo_kNm": float((P * abs(u[gdl(2)[1]]) + H * u[gdl(4)[0]]) / 2),
        "energia_kNm": float(np.sum(n**2 * longitudes / (2 * EA))),
        "R1x": float(reacciones[0]),
        "R1y": float(reacciones[1]),
        "R3y": float(reacciones[5]),
    }
    for nodo in (2, 3, 4, 5):
        ejes = gdl(nodo)
        if ejes[0] not in GDL_FIJOS:
            resultados[f"u{nodo}_mm"] = float(u[ejes[0]] * 1e3)
        if ejes[1] not in GDL_FIJOS:
            resultados[f"v{nodo}_mm"] = float(u[ejes[1]] * 1e3)
    for barra, axil in zip(BARRAS, n):
        resultados[f"N{barra[0]}{barra[1]}"] = float(axil)
    return resultados


def main():
    for clave, valor in resolver().items():
        print(f"{clave:20s} {valor:.5g}")


if __name__ == "__main__":
    main()
