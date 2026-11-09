"""Gráficos repetidos en los problemas."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def circulos_mohr(principales, unidad="MPa", ax=None):
    """Dibuja los tres círculos de Mohr y sombrea la región de estados admisibles.

    principales: las tres tensiones principales, en cualquier orden.
    Devuelve (fig, ax); sobre ax pueden añadirse los puntos de cada problema.
    """
    valores = np.sort(np.asarray(principales, dtype=float))[::-1]
    if valores.shape != (3,):
        raise ValueError("Se esperaban tres tensiones principales.")
    fig, ax = plt.subplots(figsize=(7, 4.5)) if ax is None else (ax.figure, ax)
    angulo = np.linspace(0, 2 * np.pi, 361)
    pares = [(valores[0], valores[2]), (valores[0], valores[1]), (valores[1], valores[2])]
    for mayor, menor in pares:
        centro, radio = (mayor + menor) / 2, (mayor - menor) / 2
        ax.plot(centro + radio * np.cos(angulo), radio * np.sin(angulo), color="tab:blue")
    _sombrear_region(ax, valores, angulo)
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$\sigma_n$ ({unidad})")
    ax.set_ylabel(rf"$\tau$ ({unidad})")
    return fig, ax


def _sombrear_region(ax, valores, angulo):
    """Rellena el círculo mayor y vuelve a cubrir el interior de los dos menores.

    Un círculo menor que coincide con el mayor no se cubre: la región admisible es entonces
    todo el círculo mayor.
    """
    diametro = valores[0] - valores[2]

    def rellenar(mayor, menor, color, opacidad):
        centro, radio = (mayor + menor) / 2, (mayor - menor) / 2
        ax.fill(centro + radio * np.cos(angulo), radio * np.sin(angulo), color=color, alpha=opacidad, linewidth=0, zorder=0)

    rellenar(valores[0], valores[2], "tab:blue", 0.15)
    for mayor, menor in [(valores[0], valores[1]), (valores[1], valores[2])]:
        if not (np.isclose(mayor - menor, diametro) or np.isclose(mayor - menor, 0.0)):
            rellenar(mayor, menor, "white", 1.0)


def historia_temporal(t, curvas, etiqueta_x, etiqueta_y, ax=None):
    """Dibuja una o varias historias temporales.

    curvas: diccionario {leyenda: valores}. Devuelve (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(7, 4)) if ax is None else (ax.figure, ax)
    for leyenda, valores in curvas.items():
        ax.plot(t, valores, label=leyenda)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def guardar(fig, nombre, carpeta=None):
    """Guarda la figura como PNG y devuelve su ruta.

    Sin carpeta usa la variable de entorno MMC_SALIDA y, si no existe, ./salida.
    """
    destino = Path(carpeta or os.environ.get("MMC_SALIDA", "salida"))
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{nombre}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


def cerrar(fig):
    """Libera la memoria de la figura."""
    plt.close(fig)
