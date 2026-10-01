"""Generate SMILES for GuacaMol v2's fixed C11H24 isomer-generation task.

Implement solve(payload), returning a JSON-compatible list of SMILES strings.
The evaluator canonicalizes without stereochemistry and scores the top 159
distinct structures, padding missing structures with zero scores.
"""


# EVOLVE-BLOCK-START
def solve(payload):
    """Explore only the straight chain and singly methyl-branched chains."""
    carbons = payload["target_atom_counts"]["C"]
    backbone = carbons - 1
    structures = ["C" * carbons]
    for position in range(2, backbone):
        structures.append("C" * position + "(C)" + "C" * (backbone - position))
    return structures
# EVOLVE-BLOCK-END
