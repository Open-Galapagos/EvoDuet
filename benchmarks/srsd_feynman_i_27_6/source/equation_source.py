class FeynmanICh27Eq6(KnownEquation):
    """
    - Equation: I.27.6
    - Raw: 1 / (1 / d1 + n / d2)
    - Num. Vars: 3
    - Vars:
        - x[0]: d1 (float, positive)
        - x[1]: n (float, positive)
        - x[2]: d2 (float, positive)
    - Constraints:
        - x[0] != 0
        - x[2] != 0
    """
    _eq_name = 'feynman-i.27.6'

    def __init__(self, sampling_objs=None):
        if sampling_objs is None:
            sampling_objs = [
                DefaultSampling(1.0e-3, 1.0e-1, uses_negative=False),
                DefaultSampling(1.0e-1, 1.0e1, uses_negative=False),
                DefaultSampling(1.0e-3, 1.0e-1, uses_negative=False)
            ]

        super().__init__(num_vars=3, sampling_objs=sampling_objs)
        x = self.x
        self.sympy_eq = 1 / (1 / x[0] + x[1] / x[2])

    def eq_func(self, x):
        return 1 / (1 / x[0] + x[1] / x[2])
