import random


class Material:
    """Aggregate material: name, density and grain shape.

    short_ratio and long_ratio are (low, high) ranges given as multiples of
    the sieve size, which is also the middle dimension of the grain. Short
    must stay at or below 1 and long at or above 1, so the sieve size is
    always the middle one.
    """

    def __init__(self, name, density, short_ratio, long_ratio):
        self.name = name
        self.density = density
        self.short_ratio = short_ratio
        self.long_ratio = long_ratio

    def sample_dimensions(self, length):
        """
        Return (length, depth, height) from the sieve size and aspect ratios.

        length is the sieve size (middle dimension), depth is the long axis
        and height is the short axis.
        """
        depth = length * random.uniform(*self.long_ratio)
        height = length * random.uniform(*self.short_ratio)

        return length, depth, height