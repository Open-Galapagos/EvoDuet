"""Choose a composition for the fixed OLYMPUS perovskites bandgap benchmark.

Return the three category names in a JSON-compatible composition object.
The payload provides the allowed choices, without measured bandgaps.
"""


# EVOLVE-BLOCK-START
def solve(payload):
    """A valid untuned baseline: select the first option for each component."""
    return {parameter["name"]: parameter["options"][0] for parameter in payload["parameters"]}
# EVOLVE-BLOCK-END
