import torch
import numpy as np
import sys
import copy

def find_prng_stats(prng, max_val):
    """docstring for find_prng_std"""
    if type(prng) is ParallelNumpyXorshift32:
        prng_c = copy.deepcopy(prng) # Copy PRNG to avoid changing its state
        #ParallelNumpyXorshift32(seed, dim)
        t = prng.next()
        #t = tensor1D(prng, 1)

        #t = torch.tensor(data)

        # Generate a good amount of elements
        ELEMENTS = 10000
        while t.numel() < ELEMENTS:
            temp = prng.next()
            t = torch.concat((t, temp))

        # Map the generated values to the range given
        t = apply_range(t, max_val)

        t = t.to(torch.float)
        print(t)
        print(f"mean: {t.mean().item()}")
        print(f"std: {t.std().item()}")
        return t.mean().item(), t.std().item()
    else:
        raise RuntimeError(f"find_prng_stats() not implemented for {type(prng)}")

class Xorshift32:
    def __init__(self, seed):
        if seed == 0:
            raise ValueError("Seed must be non-zero for Xorshift.")
        self.state = seed & 0xFFFFFFFF  # Ensure 32-bit state

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF  # Shift left by 13 bits
        x ^= (x >> 17) & 0xFFFFFFFF  # Shift right by 17 bits
        x ^= (x << 5) & 0xFFFFFFFF   # Shift left by 5 bits
        self.state = x
        return x

class NumpyXorshift32:
    def __init__(self, seed):
        if seed == 0:
            raise ValueError("Seed must be non-zero for Xorshift.")
        self.state = np.array([seed], dtype=np.int32)  # Ensure 32-bit state

    def next(self):
        x = np.copy(self.state)
        x ^= np.bitwise_left_shift(x, 13) # Shift left by 13 bits
        #x ^= np.bitwise_right_shift(x, 17) # Shift left by 13 bits
        x ^= np.bitwise_right_shift(x, 17) # Shift left by 13 bits
        x ^= np.bitwise_left_shift(x, 5) # Shift left by 13 bits
        #x ^= (x << 13) & 0xFFFFFFFF  # Shift left by 13 bits
        #x ^= (x >> 17) & 0xFFFFFFFF  # Shift right by 17 bits
        #x ^= (x << 5) & 0xFFFFFFFF   # Shift left by 5 bits
        self.state = x
        return torch.from_numpy(x)

class ParallelNumpyXorshift32:
    """
    Generates multiple seeds
    """
    def __init__(self, seed, numel):
        if seed == 0:
            raise ValueError("Seed must be non-zero for Xorshift.")
        #self.state = np.array([seed], dtype=np.int32)  # Ensure 32-bit state
        self._numel = numel
        # This parameter is not being used. Maybe I just copied and pasted from
        # NumpyXorshift32 when I was creating this class and it possibly can be
        # removed.
        self._seed = seed
        self.state = np.random.randint(np.iinfo(np.uint32).max, size=(self._numel), dtype=np.uint32)
        zeros = self.state == 0
        if zeros.any():
            self.state[zeros] = self.state[zeros]+1

    def next(self):
        x = np.copy(self.state)
        x ^= np.bitwise_left_shift(x, 13) # Shift left by 13 bits
        x ^= np.bitwise_right_shift(x, 17) # Shift left by 13 bits
        x ^= np.bitwise_left_shift(x, 5) # Shift left by 13 bits
        self.state = x
        return torch.from_numpy(x)

# Create a function to map raw randoms into [-max, max]
class CachedPRNG(object):
    """docstring for CachedPRNG"""
    def __init__(self):
        super(CachedPRNG, self).__init__()
        self._data = None
        self._cache_el = 10000000
        self._ptr = 0

    def _generate(self, prng):
        """docstring for _generate"""
        t = torch.zeros(self._cache_el)
        t = tensor1D(prng, t.shape)
        return t

    def tensor1D(self, prng, req_el):
        """docstring for tensor1D"""
        if self._data is None:
            self._data = self._generate(prng)

        available_el = self._data.shape.numel() - self._ptr
        if available_el >= req_el:
            ret = self._data[self._ptr: self._ptr+req_el]
            self._ptr = self._ptr + req_el
            return ret
        else:
            self._new_data = self._generate(prng)
            ret = torch.zeros(req_el) # Create a buffer tensor
            ret[0:available_el] = self._data[self._ptr:]
            missing_el = req_el - available_el
            ret[available_el:] = self._new_data[0:missing_el]
            self._ptr = missing_el
            self._data = self._new_data
            return ret

def tensor1D(prng, shape):
    """Return a 1D tensor generated with the given PRNG"""
    data = [prng.next() for i in range(shape.numel())]
    t = torch.tensor(data)
    return t.view(shape)

def apply_range(t, max):
    """Map the random values in 't' to [-max, max-1]"""
    m = max*2
    t = t.to(torch.int32)
    t = t%m
    t = t-max
    return t

def main():
    """docstring for main"""
    seed = 120321
    dim = 1000
    #prng = Xorshift32(seed)
    prng = NumpyXorshift32(seed)
    data = [prng.next() for i in range(dim)]

    t = torch.tensor(data)

    max = 32
    ##t = t%32
    #t = t%max
    #t = t-max/2
    t = apply_range(t, max)
    t = t.to(torch.float)
    #print(t)
    print(f"mean: {t.mean().item()}")
    print(f"std: {t.std().item()}")

def main_cached():
    """docstring for main"""
    seed = 120321
    dim = 10
    #prng = Xorshift32(seed)
    prng = NumpyXorshift32(seed)
    t = prng.next()

    #t = torch.tensor(data)

    max = 32
    ##t = t%32
    #t = t%max
    #t = t-max/2
    t = apply_range(t, max)
    t = t.to(torch.float)
    #print(t)
    print(f"mean: {t.mean().item()}")
    print(f"std: {t.std().item()}")

def main_parallel():
    """docstring for main"""
    #seed = 120321
    #dim = 10000
    ##prng = Xorshift32(seed)
    #prng = ParallelNumpyXorshift32(seed, dim)
    #t = prng.next()
    #print(t)
    ##t = tensor1D(prng, 1)

    ##t = torch.tensor(data)

    #max = 32
    #max = 16
    ###t = t%32
    ##t = t%max
    ##t = t-max/2
    #t = apply_range(t, max)
    #t = t.to(torch.float)
    ##print(t)
    #print(f"mean: {t.mean().item()}")
    #print(f"std: {t.std().item()}")

    # Find stats with find_prng_stats()
    seed = 120321
    dim = 1000
    max = 32
    prng = ParallelNumpyXorshift32(seed, dim)
    mean, std = find_prng_stats(prng, max)
    print(f'mean: {mean} | std: {std}')


if __name__ == '__main__':
    #main()
    #main_cached()
    main_parallel()
