"""Produce a bounded temperature policy for the published batch reactor."""


# EVOLVE-BLOCK-START
def solve(payload):
    return {"temperature": [350.0] * len(payload["time"])}
# EVOLVE-BLOCK-END
