"""Offline regression checks for the public SRSD srsd_feynman_i_27_6 port."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent))
import _srsd_feynman_runtime as runtime

spec = importlib.util.spec_from_file_location("srsd_feynman_i_27_6_evaluator", TASK_DIR / "evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class SRSDRegressionTests(unittest.TestCase):
    def grade(self, expression, split="val"):
        x, y = runtime.load_split(TASK_DIR, split)
        return runtime.solve({"artifact": {"expression": expression}, "x": x.tolist(),
                              "y": y.tolist(), "split": split,
                              "reference_expression": evaluator.REFERENCE_EXPRESSION})

    def test_pinned_files_and_original_splits(self):
        provenance = json.loads((TASK_DIR / "data_provenance.json").read_text())
        for entry in provenance["files"]:
            with self.subTest(file=entry["file"]):
                self.assertEqual(hashlib.sha256((TASK_DIR / entry["file"]).read_bytes()).hexdigest(),
                                 entry["sha256"])
        for split, count in [("train", 8000), ("val", 1000), ("test", 1000)]:
            x, y = runtime.load_split(TASK_DIR, split)
            self.assertEqual(x.shape, (count, 3))
            self.assertEqual(y.shape, (count,))
            self.assertTrue(np.all(y != 0))
        payload = evaluator.make_payload()
        self.assertEqual(set(payload), {"x_train", "y_train", "variables", "variable_descriptions"})
        self.assertEqual(len(payload["x_train"]), 8000)
        self.assertEqual(len(payload["y_train"]), 8000)

    def test_reference_and_equivalent_expression(self):
        for expression in ["1/(1/x0+x1/x2)", "x0*x2/(x2+x0*x1)"]:
            for split in ["val", "test"]:
                result = self.grade(expression, split)
                self.assertEqual(result["validity"], 1.0)
                self.assertLess(result["mean_squared_relative_error"], 1e-25)
                if split == "test":
                    self.assertEqual(result["normalized_edit_distance"], 0.0)
                else:
                    self.assertNotIn("normalized_edit_distance", result)

    def test_relative_loss_is_scale_sensitive_and_distinct_from_ned(self):
        target = np.array([1e-30, -4e-28, 3e-25])
        self.assertEqual(runtime.mean_squared_relative_error(np.zeros(3), target), 1.0)
        result = self.grade("2/(1/x0+x1/x2)", "test")
        self.assertAlmostEqual(result["mean_squared_relative_error"], 1.0, places=12)
        self.assertAlmostEqual(result["combined_score"], 0.5, places=12)
        # Scaling inserts one additional constant node in this particular tree.
        self.assertEqual(result["normalized_edit_distance"], 0.1)
        # Existing numeric constants share a label, irrespective of magnitude.
        self.assertEqual(runtime.normalized_edit_distance(
            runtime.to_sympy(runtime.parse_expression("2*x0")),
            runtime.to_sympy(runtime.parse_expression("3*x0"))), 0.0)
        with self.assertRaises(ValueError):
            runtime.mean_squared_relative_error(np.zeros(1), np.zeros(1))

    def test_malformed_artifacts_and_restricted_syntax(self):
        x, y = runtime.load_split(TASK_DIR, "val")
        base = {"x": x[:3].tolist(), "y": y[:3].tolist(), "split": "val"}
        for artifact in [None, False, [], "x0", {"expression": "x0", "combined_score": 1},
                         {"expression": None}, {"expression": False}, {"expression": 1}]:
            with self.subTest(artifact=artifact), self.assertRaises((ValueError, TypeError)):
                runtime.solve(dict(base, artifact=artifact))
        for expression in ["True*x0", "None", "__import__('os').system('id')", "x0.__class__",
                           "x0[0]", "x3", "sin(x0, x1)", "cos(x0=x0)", "x0 if x1 else x2",
                           "x0**x1", "x0**9", "x0+1e999", "x0+" * 1000 + "x0"]:
            with self.subTest(expression=expression), self.assertRaises((ValueError, SyntaxError)):
                runtime.parse_expression(expression)

    def test_constant_singular_and_complex_models_are_rejected(self):
        expressions = ["0", "1", "x0-x0", "x0/(x1-x1)",
                       "(1/(1/x0+x1/x2))+(-1)**0.5",
                       "(1/(1/x0+x1/x2))+0*((-1)**0.5)"]
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises((ValueError, FloatingPointError)):
                self.grade(expression)

    def test_seed_and_reference_execute_in_isolated_runner(self):
        seed = evaluator.evaluate(str(TASK_DIR / "initial_program.py"))
        reference = evaluator.evaluate(str(TASK_DIR / "oracle" / "best_program.py"))
        final = evaluator.evaluate_final(str(TASK_DIR / "oracle" / "best_program.py"))
        self.assertEqual(seed["validity"], 1.0, seed)
        self.assertEqual(reference["validity"], 1.0, reference)
        self.assertGreater(reference["combined_score"], seed["combined_score"] + 0.3)
        self.assertEqual(final["validity"], 1.0, final)
        self.assertEqual(final["normalized_edit_distance"], 0.0)
        self.assertLess(final["mean_squared_relative_error"], 1e-25)

    def test_candidate_cannot_supply_its_own_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.py"
            candidate.write_text("def solve(payload):\n    return {'combined_score': 1.0}\n")
            result = evaluator.evaluate(str(candidate))
        self.assertEqual(result["validity"], 0.0)
        self.assertEqual(result["combined_score"], 0.0)

    def test_oracle_text_is_retrieved_excerpt_and_scaffold_matches(self):
        document = json.loads((TASK_DIR / "oracle" / "web_search.json").read_text())[0]["response"]["results"][0]
        raw = (TASK_DIR / "oracle" / "source_excerpt.txt").read_text()
        self.assertEqual(document["raw_content"], raw)
        self.assertTrue(document["raw_content_truncated"])
        provenance = json.loads((TASK_DIR / "oracle" / "provenance.json").read_text())
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
                         provenance["documents"][0]["raw_content_sha256"])
        scaffold = []
        for file in [TASK_DIR / "initial_program.py", TASK_DIR / "oracle" / "best_program.py"]:
            before, rest = file.read_text().split("# EVOLVE-BLOCK-START")
            _, after = rest.split("# EVOLVE-BLOCK-END")
            scaffold.append((before, after))
        self.assertEqual(*scaffold)

    def test_exact_tree_distance_against_independent_reference_cases(self):
        data = json.loads((TASK_DIR / "source" / "tree_distance_reference_cases.json").read_text())
        self.assertEqual(data["provenance"]["mismatches"], 0)
        self.assertEqual(data["provenance"]["cases_checked"], 256)
        for i, case in enumerate(data["cases"]):
            with self.subTest(case=i):
                self.assertEqual(runtime.ordered_tree_distance(case["a"], case["b"]), case["distance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
