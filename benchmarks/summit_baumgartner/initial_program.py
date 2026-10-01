"""Propose conditions for Summit's fixed aniline C-N cross-coupling emulator."""


# EVOLVE-BLOCK-START
def solve(payload):
    """Use a fixed catalyst/base pair and the midpoints of the numeric bounds."""
    return {
        "catalyst": "tBuXPhos",
        "base": "DBU",
        **{name: (bounds[0] + bounds[1]) / 2
           for name, bounds in payload["bounds"].items()},
    }
# EVOLVE-BLOCK-END
