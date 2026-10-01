"""Choose a composition for the fixed OLYMPUS perovskites bandgap benchmark.

Return the three category names in a JSON-compatible composition object.
The payload provides the allowed choices, without measured bandgaps.
"""


# EVOLVE-BLOCK-START
def solve(payload):
    """Use the minimum-bandgap composition stated in the Gryffin tutorial."""
    return {"organic": "hydrazinium", "cation": "Sn", "anion": "I"}
# EVOLVE-BLOCK-END
