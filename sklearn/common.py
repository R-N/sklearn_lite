import numbers
import numpy as np
xp=np

def _make_indexable(iterable):
    if hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)

def indexable(*iterables):
    result = [_make_indexable(X) for X in iterables]
    return result

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def _find_matching_floating_dtype(*arrays):
    dtyped_arrays = [a for a in arrays if hasattr(a, "dtype")]
    floating_dtypes = [
        a.dtype for a in dtyped_arrays if xp.isdtype(a.dtype, "real floating")
    ]
    if floating_dtypes:
        # Return the floating dtype with the highest precision:
        return xp.result_type(*floating_dtypes)

    return xp.asarray(0.0).dtype

def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric", xp=xp):

    if y_true.ndim == 1:
        y_true = xp.reshape(y_true, (-1, 1))

    if y_pred.ndim == 1:
        y_pred = xp.reshape(y_pred, (-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output ({0}!={1})".format(
                y_true.shape[1], y_pred.shape[1]
            )
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                "Allowed 'multioutput' string values are {}. "
                "You provided multioutput={!r}".format(
                    allowed_multioutput_str, multioutput
                )
            )
    elif multioutput is not None:
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif n_outputs != multioutput.shape[0]:
            raise ValueError(
                "There must be equally many custom weights "
                f"({multioutput.shape[0]}) as outputs ({n_outputs})."
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput

def _weighted_percentile(array, sample_weight, percentile=50):
    n_dim = array.ndim
    if n_dim == 0:
        return array[()]
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    # When sample_weight 1D, repeat for each array.shape[1]
    if array.shape != sample_weight.shape and array.shape[0] == sample_weight.shape[0]:
        sample_weight = np.tile(sample_weight, (array.shape[1], 1)).T
    sorted_idx = np.argsort(array, axis=0)
    sorted_weights = np.take_along_axis(sample_weight, sorted_idx, axis=0)

    # Find index of median prediction for each sample
    weight_cdf = np.cumsum(sorted_weights, axis=0)
    adjusted_percentile = percentile / 100 * weight_cdf[-1]

    # For percentile=0, ignore leading observations with sample_weight=0. GH20528
    mask = adjusted_percentile == 0
    adjusted_percentile[mask] = np.nextafter(
        adjusted_percentile[mask], adjusted_percentile[mask] + 1
    )

    percentile_idx = np.array(
        [
            np.searchsorted(weight_cdf[:, i], adjusted_percentile[i])
            for i in range(weight_cdf.shape[1])
        ]
    )
    percentile_idx = np.array(percentile_idx)
    # In rare cases, percentile_idx equals to sorted_idx.shape[0]
    max_idx = sorted_idx.shape[0] - 1
    percentile_idx = np.apply_along_axis(
        lambda x: np.clip(x, 0, max_idx), axis=0, arr=percentile_idx
    )

    col_index = np.arange(array.shape[1])
    percentile_in_sorted = sorted_idx[percentile_idx, col_index]
    percentile = array[percentile_in_sorted, col_index]
    return percentile[0] if n_dim == 1 else percentile


def column_or_1d(y, *, dtype=None, warn=False):
    shape = y.shape
    if len(shape) == 1:
        return xp.asarray(xp.reshape(y, (-1,)), order="C", xp=xp)
    if len(shape) == 2 and shape[1] == 1:
        return xp.asarray(xp.reshape(y, (-1,)), order="C", xp=xp)

    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(shape)
    )
