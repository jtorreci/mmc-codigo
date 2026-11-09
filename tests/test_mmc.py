"""Pruebas de la librería con resultados publicados en el libro de problemas."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmc import elasticidad, fluidos, graficos, tensores, viscoelasticidad

# Problema 1.18 (MPa)
S_118 = tensores.tensor_simetrico(150, 100, 50, s12=40, s13=30, s23=20)
# Problema 1.17 (kPa)
S_117 = np.diag([-200.0, -300.0, -400.0])


class Relativo(unittest.TestCase):
    def cerca(self, calculado, esperado, tol=5e-3):
        self.assertLessEqual(abs(calculado - esperado), tol * abs(esperado), f"{calculado} != {esperado}")


class Tensores(Relativo):
    def test_tensor_simetrico_coloca_cada_componente(self):
        self.assertEqual(S_118[0, 1], 40)
        self.assertEqual(S_118[2, 0], 30)
        self.assertEqual(S_118[1, 2], 20)
        self.assertTrue(np.array_equal(S_118, S_118.T))

    def test_invariantes_p118(self):
        i1, i2, i3 = tensores.invariantes(S_118)
        self.cerca(i1, 300)
        self.cerca(i2, 24600)
        self.cerca(i3, np.linalg.det(S_118))

    def test_principales_ordena_de_mayor_a_menor_p118(self):
        valores, vectores = tensores.principales(S_118)
        self.cerca(valores[0], 181.96)
        self.assertTrue(np.all(np.diff(valores) <= 0))
        self.assertTrue(np.allclose(vectores.T @ vectores, np.eye(3)))
        self.cerca(np.linalg.det(vectores), 1.0)
        self.assertTrue(np.allclose(S_118 @ vectores[:, 0], valores[0] * vectores[:, 0]))

    def test_desviador_tiene_traza_nula(self):
        self.assertAlmostEqual(np.trace(tensores.desviador(S_118)), 0.0)

    def test_invariantes_desviador_p118(self):
        j2, _ = tensores.invariantes_desviador(S_118)
        self.cerca(j2, 5400)

    def test_tau_max_p117(self):
        tau, sigma_n = tensores.tau_max(S_117)
        self.cerca(tau, 100)
        self.cerca(sigma_n, -300)

    def test_haigh_westergaard_p117(self):
        xi, rho, _ = tensores.haigh_westergaard(S_117)
        self.cerca(xi, -900 / math.sqrt(3))
        self.cerca(rho, 141.4)

    def test_lode_de_estados_conocidos(self):
        traccion = np.diag([1.0, 0.0, 0.0])
        compresion = np.diag([0.0, 0.0, -1.0])
        cortante = np.diag([1.0, 0.0, -1.0])
        self.assertAlmostEqual(tensores.haigh_westergaard(traccion)[2], 0.0, places=7)
        self.assertAlmostEqual(tensores.haigh_westergaard(compresion)[2], math.pi / 3, places=7)
        self.assertAlmostEqual(tensores.haigh_westergaard(cortante)[2], math.pi / 6, places=7)

    def test_octaedricas_p117(self):
        sigma, tau = tensores.octaedricas(S_117)
        self.cerca(sigma, -300)
        self.cerca(tau, math.sqrt(2 * 10000 / 3))

    def test_von_mises_y_tresca_aceptan_principales_p51(self):
        self.cerca(tensores.von_mises([200, 100, 0]), 173.2)
        self.cerca(tensores.tresca([200, 100, 0]), 200)

    def test_von_mises_acepta_tensor(self):
        self.cerca(tensores.von_mises(S_118), math.sqrt(3 * 5400))

    def test_componentes_intrinsecas(self):
        t, sigma_n, tau = tensores.componentes_intrinsecas(np.diag([10.0, 0.0, 0.0]), [1, 1, 0])
        self.assertTrue(np.allclose(t, [10 / math.sqrt(2), 0, 0]))
        self.cerca(sigma_n, 5)
        self.cerca(tau, 5)

    def test_componentes_intrinsecas_rechaza_normal_nula(self):
        with self.assertRaises(ValueError):
            tensores.componentes_intrinsecas(S_118, [0, 0, 0])

    def test_tensor_no_simetrico_se_rechaza(self):
        with self.assertRaises(ValueError):
            tensores.principales(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


class Elasticidad(Relativo):
    def test_constantes(self):
        c = elasticidad.constantes_elasticas(200e3, 0.3)
        self.cerca(c.G, 76923.08)
        self.cerca(c.K, 166666.7)
        self.cerca(c.lam, 115384.6)
        self.cerca(c.M, c.lam + 2 * c.G)

    def test_hooke_y_su_inversa_son_reciprocas(self):
        eps = tensores.tensor_simetrico(1e-3, -2e-4, 3e-4, s12=5e-4)
        sigma = elasticidad.hooke(eps, 200e3, 0.3)
        self.assertTrue(np.allclose(elasticidad.hooke_inversa(sigma, 200e3, 0.3), eps))

    def test_kirsch_p122(self):
        _, stt, _ = elasticidad.kirsch(100.0, 1.0, math.pi / 2, a=1.0)
        self.cerca(stt, 300)
        _, stt, _ = elasticidad.kirsch(100.0, 2.0, math.pi / 2, a=1.0)
        self.cerca(stt, 121.9)

    def test_kirsch_borde_libre(self):
        srr, _, srt = elasticidad.kirsch(100.0, 1.0, 0.7, a=1.0)
        self.assertAlmostEqual(srr, 0.0)
        self.assertAlmostEqual(srt, 0.0)

    def test_kirsch_tiende_al_campo_lejano(self):
        srr, stt, _ = elasticidad.kirsch(100.0, 1e4, 0.0, a=1.0)
        self.cerca(srr, 100)
        self.assertAlmostEqual(stt, 0.0, places=3)

    def test_kirsch_rechaza_radio_interior(self):
        with self.assertRaises(ValueError):
            elasticidad.kirsch(100.0, 0.5, 0.0, a=1.0)

    def test_kirsch_admite_vectores(self):
        _, stt, _ = elasticidad.kirsch(100.0, np.array([1.0, 2.0]), math.pi / 2, a=1.0)
        self.assertEqual(stt.shape, (2,))


class Viscoelasticidad(Relativo):
    def test_parametros_zener_p75(self):
        p = viscoelasticidad.parametros_zener(500, 800, 8e4)
        self.cerca(p.E0, 500)
        self.cerca(p.Einf, 500 * 800 / 1300)
        self.cerca(p.tau_s, 100)
        self.cerca(p.tau_e, 100 * (500 * 800 / 1300) / 500)

    def test_boltzmann_zener_carga_y_descarga_p75(self):
        p = viscoelasticidad.parametros_zener(500, 800, 8e4)
        j = lambda t: viscoelasticidad.fluencia_zener(t, p.E0, p.Einf, p.tau_s)
        eps = lambda t: viscoelasticidad.boltzmann(j, t, [0.0, 300.0], [5.0, -5.0])
        self.cerca(float(eps(299.999999)), 0.0159, tol=2e-2)
        self.cerca(float(eps(600.0)), 2.96e-4, tol=5e-3)

    def test_boltzmann_es_nula_antes_del_primer_salto(self):
        j = lambda t: viscoelasticidad.fluencia_kelvin(t, 600, 6e4)
        self.assertEqual(float(viscoelasticidad.boltzmann(j, -1.0, [0.0], [12.0])), 0.0)

    def test_kelvin_p73(self):
        j = lambda t: viscoelasticidad.fluencia_kelvin(t, 600, 6e4)
        self.cerca(float(viscoelasticidad.boltzmann(j, 600.0, [0.0, 300.0], [12.0, -12.0])), 9.46e-4)

    def test_kelvin_parte_de_cero(self):
        self.assertEqual(float(viscoelasticidad.fluencia_kelvin(0.0, 600, 6e4)), 0.0)

    def test_dualidad_zener(self):
        self.cerca(float(viscoelasticidad.relajacion_zener(0.0, 500, 300, 100)), 500)
        self.cerca(float(viscoelasticidad.relajacion_zener(1e6, 500, 300, 100)), 300)

    def test_relajacion_maxwell(self):
        self.cerca(float(viscoelasticidad.relajacion_maxwell(10.0, 1.0, 10.0)), 0.368)


class Fluidos(Relativo):
    def test_velocidad_y_reynolds_p66(self):
        u = fluidos.velocidad_media(0.02, 0.1)
        self.cerca(u, 2.546)
        self.cerca(fluidos.reynolds(u, 0.1, 1e-6), 2.546e5)

    def test_colebrook_p66(self):
        self.cerca(float(fluidos.colebrook(2.546e5, 0.045 / 100)), 0.01816)

    def test_colebrook_admite_vectores(self):
        f = fluidos.colebrook(np.array([1e4, 1e5, 1e6]))
        self.assertEqual(f.shape, (3,))
        self.assertTrue(np.all(np.diff(f) < 0))

    def test_colebrook_rechaza_reynolds_no_positivo(self):
        with self.assertRaises(ValueError):
            fluidos.colebrook(0.0)

    def test_darcy_weisbach_p66(self):
        self.cerca(fluidos.darcy_weisbach(0.01816, 500, 0.1, 1000, 2.546), 294e3)

    def test_hagen_poiseuille(self):
        esperado = 128 * 1e-3 * 500 * 0.02 / (math.pi * 0.1**4)
        self.cerca(fluidos.hagen_poiseuille(1e-3, 500, 0.1, 0.02), esperado)


class Graficos(unittest.TestCase):
    def test_circulos_mohr_dibuja_tres_circulos(self):
        fig, ax = graficos.circulos_mohr([3.0, 1.0, -1.0])
        circulos = [linea for linea in ax.lines if len(linea.get_xdata()) > 2]
        self.assertEqual(len(circulos), 3)
        graficos.cerrar(fig)

    def test_circulos_mohr_conserva_el_sombreado_con_dos_principales_iguales(self):
        fig, ax = graficos.circulos_mohr([3.0, 1.0, 1.0])
        blancos = [p for p in ax.patches if p.get_facecolor()[:3] == (1.0, 1.0, 1.0)]
        self.assertEqual(blancos, [])
        graficos.cerrar(fig)

    def test_circulos_mohr_rechaza_otra_longitud(self):
        with self.assertRaises(ValueError):
            graficos.circulos_mohr([3.0, 1.0])

    def test_guardar_crea_el_png(self):
        fig, _ = graficos.historia_temporal([0, 1], {"a": [0, 1]}, "t", "y")
        with tempfile.TemporaryDirectory() as d:
            ruta = graficos.guardar(fig, "prueba", carpeta=d)
            self.assertEqual(Path(ruta).suffix, ".png")
            self.assertGreater(Path(ruta).stat().st_size, 0)
        graficos.cerrar(fig)


if __name__ == "__main__":
    unittest.main()
