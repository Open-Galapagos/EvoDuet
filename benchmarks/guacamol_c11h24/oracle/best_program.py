"""Generate SMILES for GuacaMol v2's fixed C11H24 isomer-generation task.

Implement solve(payload), returning a JSON-compatible list of SMILES strings.
The evaluator canonicalizes without stereochemistry and scores the top 159
distinct structures, padding missing structures with zero scores.
"""


# EVOLVE-BLOCK-START
def solve(payload):
    """Enumerate carbon trees with valence at most four, once per topology."""
    import networkx as nx
    from rdkit import Chem

    carbons = payload["target_atom_counts"]["C"]
    structures = set()
    for tree in nx.nonisomorphic_trees(carbons):
        if any(degree > 4 for _, degree in tree.degree()):
            continue
        molecule = Chem.RWMol()
        for _ in range(carbons):
            molecule.AddAtom(Chem.Atom("C"))
        for first, second in tree.edges():
            molecule.AddBond(first, second, Chem.BondType.SINGLE)
        molecule = molecule.GetMol()
        Chem.SanitizeMol(molecule)
        structures.add(Chem.MolToSmiles(molecule, isomericSmiles=False))
    return sorted(structures)
# EVOLVE-BLOCK-END
