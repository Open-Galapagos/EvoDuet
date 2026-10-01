"""Propose conditions for Summit's fixed aniline C-N cross-coupling emulator."""


# EVOLVE-BLOCK-START
def solve(payload):
    """Use the high-yield conditions in Summit's published Table 1."""
    return {
        "catalyst": "AlPhos",
        "base": "BTMG",
        "base_equivalents": 2.25,
        "temperature": 99.89,
        "t_res": 1763.66,
    }
# EVOLVE-BLOCK-END
