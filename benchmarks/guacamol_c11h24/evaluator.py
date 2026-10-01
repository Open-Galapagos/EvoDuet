"""Independently reproduce GuacaMol v2's C11H24 objective on returned SMILES.

The formula score, nonstereo canonicalization and top-159 aggregation follow
BenevolentAI/guacamol (MIT, 2018); pinned source snapshots and license are in
source/upstream/. The isolated candidate never provides its own score.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from rdkit import Chem, rdBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _public_optimization_runtime import run_candidate


TARGET_COUNT = 159
# Adapter resource limits. These do not change any native correctly sized output.
MAX_SUBMITTED_SMILES = 10000
MAX_SMILES_CHARS = 4096


def task_payload():
    """Fixed public benchmark definition, without a structure list or reference."""
    return {
        "molecular_formula": "C11H24",
        "target_atom_counts": {"C": 11, "H": 24},
        "target_total_atoms": 35,
        "element_count_sigma": 1.0,
        "total_atom_count_sigma": 2.0,
        "number_molecules": TARGET_COUNT,
        "include_stereocenters": False,
        "max_submitted_smiles": MAX_SUBMITTED_SMILES,
        "max_smiles_chars": MAX_SMILES_CHARS,
    }


def molecule_score(molecule):
    """GuacaMol IsomerScoringFunction('C11H24', mean_function='geometric')."""
    with_hydrogens = Chem.AddHs(molecule)
    carbon_count = sum(atom.GetSymbol() == "C" for atom in molecule.GetAtoms())
    hydrogen_count = sum(atom.GetSymbol() == "H" for atom in with_hydrogens.GetAtoms())
    total_count = with_hydrogens.GetNumAtoms()
    # Multiply the three Gaussian contributions before taking their cube root,
    # as upstream geometric_mean does (including its floating-point underflow).
    contributions = (
        math.exp(-0.5 * (carbon_count - 11) ** 2),
        math.exp(-0.5 * (hydrogen_count - 24) ** 2),
        math.exp(-0.5 * ((total_count - 35) / 2.0) ** 2),
    )
    score = math.prod(contributions) ** (1.0 / 3.0)
    exact_formula = carbon_count == 11 and hydrogen_count == 24 and total_count == 35
    return score, exact_formula


def score_smiles(structures):
    """Filter unparseable SMILES, deduplicate, then compute the native top-159 mean.

GuacaMol does not impose extra connectivity, charge, or synthesis constraints.
It also canonicalizes with isomericSmiles=False, which discards isotope labels
as well as stereochemistry. Scoring reparses that canonical representation.
"""
    if not isinstance(structures, list):
        raise ValueError("solve(payload) must return a list of SMILES strings")
    if len(structures) > MAX_SUBMITTED_SMILES:
        raise ValueError(f"At most {MAX_SUBMITTED_SMILES} SMILES may be submitted")
    if any(not isinstance(smiles, str) for smiles in structures):
        raise ValueError("Every list entry must be a SMILES string")
    if any(len(smiles) > MAX_SMILES_CHARS for smiles in structures):
        raise ValueError(f"Each SMILES must be at most {MAX_SMILES_CHARS} characters")

    canonicalized = set()
    valid_entries = 0
    # Invalid candidate strings are expected during search; keep stderr bounded.
    with rdBase.BlockLogs():
        for smiles in structures:
            try:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    continue
                canonical = Chem.MolToSmiles(molecule, isomericSmiles=False)
            except (ValueError, RuntimeError):
                continue
            valid_entries += 1
            canonicalized.add(canonical)

        scores = []
        exact_formula_count = 0
        for canonical in sorted(canonicalized):
            molecule = Chem.MolFromSmiles(canonical)
            score, exact_formula = molecule_score(molecule)
            scores.append(score)
            exact_formula_count += int(exact_formula)

    scores.sort(reverse=True)
    # The denominator remains 159 even for fewer structures; this is zero padding.
    top_159 = sum(scores[:TARGET_COUNT]) / TARGET_COUNT
    return {
        "combined_score": top_159,
        "top_159": top_159,
        "validity": float(bool(canonicalized)),
        "submitted_count": len(structures),
        "valid_smiles_count": valid_entries,
        "unique_molecule_count": len(canonicalized),
        "duplicate_count": valid_entries - len(canonicalized),
        "invalid_smiles_count": len(structures) - valid_entries,
        "exact_formula_count": exact_formula_count,
        "missing_count": max(0, TARGET_COUNT - len(canonicalized)),
    }


def evaluate(program_path):
    try:
        structures = run_candidate(program_path, task_payload(), timeout=30)
        return score_smiles(structures)
    except (OSError, ValueError, TypeError, RuntimeError, TimeoutError, ArithmeticError) as error:
        return {"combined_score": 0.0, "validity": 0.0, "error": str(error)}


evaluate_final = evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
