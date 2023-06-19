import tensorflow as tf


def _dim_make_positive(dim, n_dims, add_one=False):
    if dim < 0:
        dim = n_dims + dim
        if add_one:
            dim += 1
    return dim


def _dims_make_positive(dims, n_dims, add_one=False):
    return [_dim_make_positive(d, n_dims, add_one) for d in dims]


def dim_pytorch2keras(dim, num_dims):
    if dim is None:
        return dim

    dim = _dim_make_positive(dim, num_dims)
    if dim == 0:
        return dim
    elif dim == 1:
        return num_dims - 1
    else:
        return dim - 1


def perm_identity(n_dims):
    return [i for i in range(n_dims)]


def perm_invert(perm):
    res = [None] * len(perm)
    for i, j in enumerate(perm):
        res[j] = i
    return res


def perm_compose(perm_next, perm_prev):
    assert len(perm_next) == len(perm_prev)
    res = [None] * len(perm_prev)
    for i2, i1 in enumerate(perm_next):
        res[i2] = perm_prev[i1]
    return res


def perm_keras2pytorch(n_dims):
    dims = [i for i in range(n_dims)]
    # Sic!
    return [dim_pytorch2keras(d, n_dims) for d in dims]


def perm_pytorch2keras(n_dims):
    return perm_invert(perm_keras2pytorch(n_dims))


def dims_pytorch2keras(dims, num_dims):
    return [dim_pytorch2keras(d, num_dims) for d in dims]


def permute_pytorch2keras(xs):
    return perm_compose(perm_pytorch2keras(len(xs)), xs)


def permute_keras2pytorch(xs):
    return perm_compose(perm_keras2pytorch(len(xs)), xs)


def is_identity_perm(perm):
    return all(i == p for i, p in enumerate(perm))


# FIXME: find a better place for these

def _permute(perm):
    if is_identity_perm(perm):
        return lambda x: x
    else:
        def func(x):
            return tf.transpose(x, perm)
        return func


def _flatten(iterable):
    if isinstance(iterable[0], (list, tuple)) and len(iterable) == 1:
        return iterable[0]
    else:
        return iterable


def _ensure_iterable(iterable):
    if isinstance(iterable, (list, tuple)):
        if len(iterable) == 1 and isinstance(iterable[0], (list, tuple)):
            return iterable[0]
        return iterable
    else:
        return [iterable]


def _ensure_tuple(iterable):
    if isinstance(iterable, tuple):
        return iterable
    else:
        return iterable,