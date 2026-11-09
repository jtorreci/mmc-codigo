"""Ejecuta en orden los ejemplos de Python del tutorial."""
import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TUTORIAL = Path(__file__).resolve().parents[1] / "TUTORIAL.md"


class Tutorial(unittest.TestCase):
    def test_los_ejemplos_se_ejecutan_y_generan_figuras(self):
        bloques = re.findall(r"```python\n(.*?)```", TUTORIAL.read_text(encoding="utf-8"), re.S)
        self.assertGreaterEqual(len(bloques), 7)
        espacio = {}
        with tempfile.TemporaryDirectory() as carpeta:
            with mock.patch.dict(os.environ, {"MMC_SALIDA": carpeta, "MPLBACKEND": "Agg"}):
                with contextlib.redirect_stdout(io.StringIO()) as salida:
                    for bloque in bloques:
                        exec(bloque, espacio)
            self.assertTrue((Path(carpeta) / "mohr.png").exists())
            self.assertTrue((Path(carpeta) / "historia.png").exists())
        self.assertIn("181.96", salida.getvalue())


if __name__ == "__main__":
    unittest.main()
