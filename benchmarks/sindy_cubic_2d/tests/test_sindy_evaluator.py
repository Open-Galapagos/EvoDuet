import importlib.util
from pathlib import Path
import unittest

import numpy as np


TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sindy_cubic_evaluator", TASK / "evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluatorTests(unittest.TestCase):
    def test_exact_published_model(self):
        coefficients = np.zeros((2, 10))
        coefficients[:, [6, 9]] = [[-0.1, 2.0], [-2.0, -0.1]]
        score = evaluator.grade_artifact({"coefficients": coefficients.tolist()})
        self.assertLess(score["vector_field_nrmse"], 1e-14)
        self.assertGreater(score["combined_score"], 0.999999)
        self.assertEqual(score["active_coefficients"], 4)

    def test_baseline_and_fitted_reference(self):
        initial = evaluator.evaluate(TASK / "initial_program.py")
        reference = evaluator.evaluate(TASK / "oracle" / "best_program.py")
        self.assertEqual(initial["validity"], 1.0)
        self.assertEqual(reference["validity"], 1.0)
        self.assertGreater(reference["combined_score"], 0.99)
        self.assertGreater(reference["combined_score"] - initial["combined_score"], 0.3)

    def test_reject_malformed_or_nonfinite_models(self):
        invalid = [None, {}, {"coefficients": [[0.0]]},
                   {"coefficients": [[float("nan")]*10]*2},
                   {"coefficients": [[float("inf")]*10]*2},
                   {"coefficients": [[False]*10]*2},
                   {"coefficients": [["0"]*10]*2},
                   {"coefficients": [[None]*10]*2},
                   {"coefficients": [[0.0]*10, None]},
                   {"coefficients": tuple([[0.0]*10]*2)},
                   {"coefficients": [[10**400]*10]*2},
                   {"coefficients": [[1e7]*10]*2},
                   {"coefficients": [[0.0]*10]*2, "score": 1.0}]
        for artifact in invalid:
            with self.subTest(artifact=artifact), self.assertRaises((ValueError, TypeError)):
                evaluator.grade_artifact(artifact)

    def test_reject_runaway_dynamics(self):
        coefficients = np.zeros((2, 10))
        coefficients[0, 1] = 1000.0
        with self.assertRaises(ValueError):
            evaluator.grade_artifact({"coefficients": coefficients.tolist()})


if __name__ == "__main__":
    unittest.main()
