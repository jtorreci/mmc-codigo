"""Funciones de uso repetido en los problemas de Mecánica de Medios Continuos.

Convenio de signos: tracción positiva, salvo que el problema indique otra cosa.
Las funciones no fijan unidades: el resultado sale en las unidades de los datos.
"""
from mmc import elasticidad, fluidos, tensores, viscoelasticidad

__all__ = ["elasticidad", "fluidos", "tensores", "viscoelasticidad"]
