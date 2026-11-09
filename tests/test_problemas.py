"""Comprueba que cada script de problema corre y reproduce los valores del libro."""
import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ESPERADOS = {
    "p0_08", "p1_06", "p1_17", "p1_18", "p1_22", "p2_07", "p2_09", "p2_14", "p3_06", "p3_07",
    "p3_08", "p3_11", "p4_14", "p4_15", "p4_16", "p5_02", "p5_06", "p5_07", "p5_08", "p6_04",
    "p6_06", "p6_11", "p7_01", "p7_02", "p7_03", "p7_04", "p7_05", "p8_09", "p8_12",
}


def scripts():
    return sorted((RAIZ / "problemas").glob("tema_*/p*_*.py"))


class Problemas(unittest.TestCase):
    def test_existe_un_script_por_cada_problema_seleccionado(self):
        self.assertEqual({ruta.stem for ruta in scripts()}, ESPERADOS)

    def test_cada_script_reproduce_los_valores_del_libro(self):
        for ruta in scripts():
            with self.subTest(ruta.stem):
                modulo = importlib.import_module(f"problemas.{ruta.parent.name}.{ruta.stem}")
                resultados = modulo.resolver()
                tolerancia = getattr(modulo, "TOLERANCIA", 5e-3)
                self.assertTrue(modulo.VALORES_LIBRO)
                for clave, esperado in modulo.VALORES_LIBRO.items():
                    self.assertLessEqual(
                        abs(resultados[clave] - esperado),
                        tolerancia * abs(esperado),
                        f"{clave}: {resultados[clave]} frente a {esperado}",
                    )

    def test_cada_script_se_ejecuta_y_genera_sus_figuras(self):
        for ruta in scripts():
            with self.subTest(ruta.stem), tempfile.TemporaryDirectory() as carpeta:
                entorno = {**os.environ, "MPLBACKEND": "Agg", "MMC_SALIDA": carpeta}
                proceso = subprocess.run(
                    [sys.executable, "-m", f"problemas.{ruta.parent.name}.{ruta.stem}"],
                    cwd=RAIZ, env=entorno, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(proceso.returncode, 0, proceso.stderr)
                self.assertTrue(proceso.stdout.strip())
                modulo = importlib.import_module(f"problemas.{ruta.parent.name}.{ruta.stem}")
                if hasattr(modulo, "graficar"):
                    self.assertTrue(list(Path(carpeta).glob(f"{ruta.stem}*.png")))


if __name__ == "__main__":
    unittest.main()
