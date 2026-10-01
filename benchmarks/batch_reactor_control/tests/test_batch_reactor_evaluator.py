import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad


TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch_reactor_evaluator", TASK / "evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluatorTests(unittest.TestCase):
    def test_constant_policy_against_integral_solution(self):
        temperature = 350.0
        k1 = 4000*np.exp(-2500/temperature)
        k2 = 620000*np.exp(-5000/temperature)
        expected_b = quad(lambda t: k1/(1+k1*t)**2*np.exp(-k2*(1-t)),
                          0.0, 1.0, epsabs=1e-12, epsrel=1e-12)[0]
        score = evaluator.grade_artifact({"temperature": [temperature]*501})
        self.assertAlmostEqual(score["final_B"], expected_b, places=10)
        self.assertAlmostEqual(score["final_A"], 1/(1+k1), places=10)
        self.assertLess(score["mass_balance_error"], 1e-10)

    def test_baseline_and_optimized_reference(self):
        initial = evaluator.evaluate(TASK / "initial_program.py")
        reference = evaluator.evaluate(TASK / "oracle" / "best_program.py")
        self.assertEqual(initial["validity"], 1.0)
        self.assertEqual(reference["validity"], 1.0)
        self.assertGreater(reference["final_B"], 0.6107)
        self.assertLess(reference["final_B"], 0.6109)
        self.assertGreater(reference["final_B"] - initial["final_B"], 0.02)

    def test_reject_malformed_and_out_of_bound_controls(self):
        invalid = [None, {}, {"temperature": [350.0]*500},
                   {"temperature": [float("nan")]*501},
                   {"temperature": [float("inf")]*501},
                   {"temperature": [True]*501},
                   {"temperature": ["350"]*501},
                   {"temperature": [None]*501},
                   {"temperature": tuple([350.0]*501)},
                   {"temperature": [10**400]*501},
                   {"temperature": [297.999]*501},
                   {"temperature": [398.001]*501},
                   {"temperature": [350.0]*501, "score": 1.0}]
        for artifact in invalid:
            with self.subTest(artifact=artifact), self.assertRaises((ValueError, TypeError)):
                evaluator.grade_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
