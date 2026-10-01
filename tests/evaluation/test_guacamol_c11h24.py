"""Native GuacaMol formula scores, unique-structure aggregation and task artifacts."""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
import pytest
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "benchmarks/guacamol_c11h24"


def load(path):
    spec = importlib.util.spec_from_file_location("guacamol_c11h24_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluator():
    return load(TASK / "evaluator.py")


@pytest.fixture(scope="module")
def reference(evaluator):
    return evaluator.run_candidate(TASK / "oracle/best_program.py", evaluator.task_payload())


def test_baseline_and_reference_reach_measured_native_scores(evaluator, reference):
    initial = evaluator.evaluate(TASK / "initial_program.py")
    best = evaluator.score_smiles(reference)
    assert initial["unique_molecule_count"] == 5
    assert initial["combined_score"] == pytest.approx(5 / 159)
    assert best["combined_score"] == 1
    assert best["unique_molecule_count"] == best["exact_formula_count"] == 159
    assert best["invalid_smiles_count"] == best["missing_count"] == 0


def test_canonical_duplicates_and_isotope_labels_do_not_raise_score(evaluator):
    structures = ["C" * 11, "C(CCCCCCCCCC)", "[13CH3]CCCCCCCCCC"]
    result = evaluator.score_smiles(structures)
    assert result["unique_molecule_count"] == 1
    assert result["duplicate_count"] == 2
    assert result["combined_score"] == pytest.approx(1 / 159)


def test_stereoisomers_are_one_structure(evaluator):
    result = evaluator.score_smiles(["CC[C@H](C)CCCCCCC", "CC[C@@H](C)CCCCCCC"])
    assert result["exact_formula_count"] == 1
    assert result["duplicate_count"] == 1
    assert result["combined_score"] == pytest.approx(1 / 159)


def test_invalid_smiles_are_removed_without_discarding_good_structures(evaluator):
    result = evaluator.score_smiles(["C" * 11, "not-a-molecule", "C1CC", "C(C)(C)(C)(C)C"])
    assert result["invalid_smiles_count"] == 3
    assert result["validity"] == 1
    assert result["combined_score"] == pytest.approx(1 / 159)


@pytest.mark.parametrize("bad", [None, 159, True, "CCCC", {}, [True], [1.0], [[]], [float("nan")]])
def test_non_string_artifacts_are_rejected(evaluator, bad):
    with pytest.raises(ValueError):
        evaluator.score_smiles(bad)


def test_empty_and_all_invalid_lists_have_zero_scores(evaluator):
    for structures in ([], ["bad-smiles", "C1CC"]):
        result = evaluator.score_smiles(structures)
        assert result["combined_score"] == result["validity"] == 0
        assert result["missing_count"] == 159


def test_resource_limits_are_enforced(evaluator):
    with pytest.raises(ValueError):
        evaluator.score_smiles(["C"] * (evaluator.MAX_SUBMITTED_SMILES + 1))
    with pytest.raises(ValueError):
        evaluator.score_smiles(["C" * (evaluator.MAX_SMILES_CHARS + 1)])


def test_formula_mismatch_receives_native_gaussian_partial_credit(evaluator):
    result = evaluator.score_smiles(["C" * 10])  # C10H22, total atoms 32.
    expected = (math.exp(-0.5) * math.exp(-2) * math.exp(-1.125)) ** (1 / 3) / 159
    assert 0 < result["combined_score"] < 1 / 159
    assert result["combined_score"] == pytest.approx(expected, rel=1e-14)
    assert result["exact_formula_count"] == 0


def test_no_added_connectivity_or_extra_element_constraints(evaluator):
    disconnected = evaluator.score_smiles(["C.CCCCCCCCCC"])
    oxygenated = evaluator.score_smiles(["C" * 11 + "O"])
    assert disconnected["combined_score"] == pytest.approx(math.exp(-2.5) ** (1 / 3) / 159)
    assert oxygenated["combined_score"] == pytest.approx(math.exp(-0.125) ** (1 / 3) / 159)


def test_more_than_159_candidates_uses_top_159(evaluator, reference):
    result = evaluator.score_smiles(reference + ["C", "CC", "CCC"])
    assert result["unique_molecule_count"] == 162
    assert result["combined_score"] == 1
    assert result["missing_count"] == 0


def test_returned_score_claim_cannot_replace_structure_artifact(evaluator, tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def solve(payload):\n    return {'combined_score': 1.0}\n")
    result = evaluator.evaluate(candidate)
    assert result["combined_score"] == result["validity"] == 0


def test_native_objective_matches_fetched_upstream_definitions(evaluator, reference):
    """Execute the fetched mathematical/chemistry definitions, without its old deps.

    Only upstream AST definitions needed by this scorer are selected. The base
    class's score method delegates to raw_score, as upstream does for valid input.
    This avoids importing obsolete scipy.histogram used elsewhere in GuacaMol.
    """
    class MoleculewiseScoringFunction:
        corrupt_score = -1.0

        def score(self, smiles):
            return self.raw_score(smiles)

    class RdkitScoringFunction:
        def __init__(self, descriptor, score_modifier):
            self.descriptor = descriptor
            self.score_modifier = score_modifier

        def score(self, smiles):
            return self.score_modifier(self.descriptor(Chem.MolFromSmiles(smiles)))

    namespace = {
        "Chem": Chem, "Mol": Chem.Mol, "np": np, "re": re,
        "List": list, "Tuple": tuple, "Dict": dict, "Optional": __import__("typing").Optional,
        "Callable": __import__("typing").Callable, "Iterable": __import__("typing").Iterable,
        "MoleculewiseScoringFunction": MoleculewiseScoringFunction,
        "RdkitScoringFunction": RdkitScoringFunction, "ScoreModifier": object,
        "remove_duplicates": lambda items: list(dict.fromkeys(items)),
    }
    definitions = {
        "utils/math.py": {"geometric_mean", "arithmetic_mean"},
        "utils/descriptors.py": {"num_atoms", "AtomCounter"},
        "utils/chemistry.py": {"parse_molecular_formula", "canonicalize", "canonicalize_list"},
        "score_modifier.py": {"GaussianModifier"},
        "common_scoring_functions.py": {"IsomerScoringFunction"},
        "goal_directed_score_contributions.py": {
            "ScoreContributionSpecification", "uniform_specification", "compute_global_score"
        },
    }
    for filename, names in definitions.items():
        tree = ast.parse((TASK / "source/upstream" / filename).read_text())
        selected = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(selected) == len(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)

    upstream = namespace["IsomerScoringFunction"]("C11H24")
    cases = [
        reference,
        ["C" * 10, "C.CCCCCCCCCC", "C" * 11 + "O"],
        ["CC[C@H](C)CCCCCCC", "CC[C@@H](C)CCCCCCC", "[13CH3]CCCCCCCCCC"],
        [""],  # Native canonicalize retains RDKit's zero-atom molecule.
        [],
    ]
    for structures in cases:
        canonical = namespace["canonicalize_list"](structures, include_stereocenters=False)
        scores = [upstream.score(smiles) for smiles in canonical]
        scores += [0.0] * max(0, 159 - len(scores))
        expected, _ = namespace["compute_global_score"](namespace["uniform_specification"](159), scores)
        actual = evaluator.score_smiles(structures)["combined_score"]
        assert actual == pytest.approx(expected, rel=1e-13, abs=0)




def test_scaffold_is_identical_and_payload_has_no_structure_table(evaluator):
    def outside(path):
        text = path.read_text()
        before, rest = text.split("# EVOLVE-BLOCK-START")
        _, after = rest.split("# EVOLVE-BLOCK-END")
        return before, after

    assert outside(TASK / "initial_program.py") == outside(TASK / "oracle/best_program.py")
    payload = evaluator.task_payload()
    assert "smiles" not in payload and "best" not in payload and "oracle" not in payload
    payload["target_atom_counts"]["C"] = 0
    assert evaluator.task_payload()["target_atom_counts"]["C"] == 11
