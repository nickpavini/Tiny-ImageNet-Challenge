import numpy as np
from tqdm import tqdm

# shuffle 2 or 3 numpy arrays in parallel
def unsisonShuffle(a, b, c = None, p = None):
    if (p is None):
        p = np.random.permutation(len(a)) # create permutation

    # swap ith variable with p[i]th permutation
    if c is None:
        assert (len(a) == len(b)), (len(a), len(b))
        for i in tqdm(range(len(a)), desc='Shuffling in unison'):
            temp = [a[i], b[i]]
            a[i], b[i] = a[p[i]], b[p[i]]
            a[p[i]], b[p[i]] = temp[0], temp[1]
        return a, b
    else:
        assert (len(a) == len(b) and len(a) == len(c)), (len(a), len(b), len(c))
        for i in tqdm(range(len(a)), desc='Shuffling in unison'):
            temp = [a[i], b[i], c[i]] #
            a[i], b[i], c[i] = a[p[i]], b[p[i]], c[p[i]]
            a[p[i]], b[p[i]], c[p[i]] = temp[0], temp[1], temp[2]
        return a, b, c
